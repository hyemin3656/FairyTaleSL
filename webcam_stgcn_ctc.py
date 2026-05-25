import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose

from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules

from test_stgcn_ctc import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_LABEL_MAP,
    NUM_HAND,
    NUM_POSE,
    build_data_batch,
    build_keypoint_sample,
    ensure_tvc,
    load_label_map,
    map_gloss_ids,
    nan_to_zero_with_score,
    pad_or_trim_time,
    resolve_device,
    synchronize,
    to_int_list,
)

try:
    import mediapipe as mp
except ImportError:
    mp = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "realtime_outputs"

FPS = 30
USE_ROI = False
ROI_CROP_SIZE = 256

LEFT_WRIST_IDX = 16
RIGHT_WRIST_IDX = 15
POSE_POINTS = 33
HAND_POINTS = 21

mp_pose = mp.solutions.pose if mp is not None else None
mp_hands = mp.solutions.hands if mp is not None else None
mp_drawing = mp.solutions.drawing_utils if mp is not None else None


def require_mediapipe():
    if mp is None:
        raise ImportError(
            "mediapipe is required for webcam keypoint extraction. "
            "Install it in the same environment, e.g. "
            "`pip install mediapipe==0.10.13`.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture webcam keypoints with the same MediaPipe rules used for "
            "training, crop by hand-detection rule, save npy files, and run "
            "ST-GCN-BiLSTM-CTC inference."))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on. Use 'auto' to prefer CUDA when available.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--start-ratio", type=float, default=0.8)
    parser.add_argument("--end-ratio", type=float, default=0.8)
    parser.add_argument("--use-roi", action="store_true", default=USE_ROI)
    parser.add_argument("--roi-crop-size", type=int, default=ROI_CROP_SIZE)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-cropped-frames", type=int, default=1)

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.cls_head.dropout=0.0.")
    return parser.parse_args()


def landmarks_to_array(landmarks, num_points, image_w, image_h):
    """
    return shape: (num_points, 4)
    columns: x_pixel, y_pixel, z, visibility/presence
    """
    arr = np.full((num_points, 4), np.nan, dtype=np.float32)

    if landmarks is None:
        return arr

    for i, lm in enumerate(landmarks.landmark):
        score = getattr(lm, "visibility", np.nan)
        arr[i] = [lm.x * image_w, lm.y * image_h, lm.z, score]

    return arr


def detect_hand_in_roi(
    image_bgr,
    hands_roi_model,
    wrist_x_px,
    wrist_y_px,
    crop_size=256,
):
    """
    Pose wrist coordinate centered crop_size x crop_size ROI hand detection.
    Returns a hand array in full-image pixel coordinates and the crop box.
    """
    if np.isnan(wrist_x_px) or np.isnan(wrist_y_px):
        return None, None

    h, w = image_bgr.shape[:2]
    cx = int(round(wrist_x_px))
    cy = int(round(wrist_y_px))
    half = crop_size // 2

    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)

    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return None, None

    crop_bgr = image_bgr[y0:y1, x0:x1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb.flags.writeable = False

    roi_result = hands_roi_model.process(crop_rgb)

    if not roi_result.multi_hand_landmarks:
        return None, (x0, y0, x1, y1)

    hand_landmarks = roi_result.multi_hand_landmarks[0]
    crop_h, crop_w = crop_bgr.shape[:2]

    hand_arr = np.full((HAND_POINTS, 4), np.nan, dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        hand_arr[i] = [
            lm.x * crop_w + x0,
            lm.y * crop_h + y0,
            lm.z,
            1.0,
        ]

    return hand_arr, (x0, y0, x1, y1)


def draw_hand_array(overlay, hand_arr, color, thickness=2, radius=2):
    if hand_arr is None or np.isnan(hand_arr[:, :2]).all():
        return

    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        x1, y1 = hand_arr[start_idx, :2]
        x2, y2 = hand_arr[end_idx, :2]
        if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
            cv2.line(
                overlay,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                thickness,
            )

    for x, y in hand_arr[:, :2]:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(overlay, (int(round(x)), int(round(y))), radius, color, -1)


@dataclass
class ExtractedFrame:
    frame_idx: int
    image_bgr: np.ndarray
    pose: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    left_hand_detected: bool
    right_hand_detected: bool
    mean_pose_visibility: float
    roi_rescued_count: int
    elapsed_ms: float

    @property
    def any_hand_detected(self):
        return self.left_hand_detected or self.right_hand_detected


class MediaPipeKeypointExtractor:
    def __init__(self, use_roi=USE_ROI, roi_crop_size=ROI_CROP_SIZE):
        require_mediapipe()
        self.use_roi = use_roi
        self.roi_crop_size = roi_crop_size
        self.pose_model = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.hands_model = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.hands_roi_model = None
        if use_roi:
            self.hands_roi_model = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=2,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )

    def close(self):
        self.pose_model.close()
        self.hands_model.close()
        if self.hands_roi_model is not None:
            self.hands_roi_model.close()

    def process_frame(self, image_bgr, frame_idx):
        start = time.perf_counter()
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        pose_result = self.pose_model.process(image_rgb)
        hands_result = self.hands_model.process(image_rgb)

        pose_arr = landmarks_to_array(pose_result.pose_landmarks, POSE_POINTS, w, h)
        pose_detected = pose_result.pose_landmarks is not None
        if pose_detected:
            mean_pose_visibility = np.nanmean(pose_arr[:, 3])
        else:
            mean_pose_visibility = 0.0

        left_hand_arr = np.full((HAND_POINTS, 4), np.nan, dtype=np.float32)
        right_hand_arr = np.full((HAND_POINTS, 4), np.nan, dtype=np.float32)
        left_hand_detected = False
        right_hand_detected = False
        left_hand_roi_rescued = False
        right_hand_roi_rescued = False
        left_roi_box = None
        right_roi_box = None
        best_left_dist = np.inf
        best_right_dist = np.inf

        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for hand_landmarks, handedness in zip(
                hands_result.multi_hand_landmarks,
                hands_result.multi_handedness,
            ):
                label = handedness.classification[0].label
                hand_arr = landmarks_to_array(hand_landmarks, HAND_POINTS, w, h)

                if pose_detected:
                    hand_wrist = hand_arr[0, :2]
                    left_wrist = pose_arr[LEFT_WRIST_IDX, :2]
                    right_wrist = pose_arr[RIGHT_WRIST_IDX, :2]
                    if (
                        not np.any(np.isnan(hand_wrist))
                        and (
                            not np.any(np.isnan(left_wrist))
                            or not np.any(np.isnan(right_wrist))
                        )
                    ):
                        dist_left = np.inf
                        dist_right = np.inf
                        if not np.any(np.isnan(left_wrist)):
                            dist_left = np.linalg.norm(hand_wrist - left_wrist)
                        if not np.any(np.isnan(right_wrist)):
                            dist_right = np.linalg.norm(hand_wrist - right_wrist)

                        if dist_left < dist_right:
                            if dist_left < best_left_dist:
                                best_left_dist = dist_left
                                left_hand_arr = hand_arr
                                left_hand_detected = True
                        else:
                            if dist_right < best_right_dist:
                                best_right_dist = dist_right
                                right_hand_arr = hand_arr
                                right_hand_detected = True
                    else:
                        if label == "Left":
                            left_hand_arr = hand_arr
                            left_hand_detected = True
                        else:
                            right_hand_arr = hand_arr
                            right_hand_detected = True
                else:
                    if label == "Left":
                        left_hand_arr = hand_arr
                        left_hand_detected = True
                    else:
                        right_hand_arr = hand_arr
                        right_hand_detected = True

        if self.use_roi and self.hands_roi_model is not None and pose_detected:
            if not left_hand_detected:
                wrist_x, wrist_y = pose_arr[LEFT_WRIST_IDX, :2]
                roi_hand_arr, left_roi_box = detect_hand_in_roi(
                    image_bgr,
                    self.hands_roi_model,
                    wrist_x,
                    wrist_y,
                    crop_size=self.roi_crop_size,
                )
                if roi_hand_arr is not None:
                    left_hand_arr = roi_hand_arr
                    left_hand_detected = True
                    left_hand_roi_rescued = True

            if not right_hand_detected:
                wrist_x, wrist_y = pose_arr[RIGHT_WRIST_IDX, :2]
                roi_hand_arr, right_roi_box = detect_hand_in_roi(
                    image_bgr,
                    self.hands_roi_model,
                    wrist_x,
                    wrist_y,
                    crop_size=self.roi_crop_size,
                )
                if roi_hand_arr is not None:
                    right_hand_arr = roi_hand_arr
                    right_hand_detected = True
                    right_hand_roi_rescued = True

        elapsed_ms = (time.perf_counter() - start) * 1000
        extracted = ExtractedFrame(
            frame_idx=frame_idx,
            image_bgr=image_bgr.copy(),
            pose=pose_arr,
            left_hand=left_hand_arr,
            right_hand=right_hand_arr,
            left_hand_detected=left_hand_detected,
            right_hand_detected=right_hand_detected,
            mean_pose_visibility=float(mean_pose_visibility),
            roi_rescued_count=int(left_hand_roi_rescued) + int(right_hand_roi_rescued),
            elapsed_ms=elapsed_ms,
        )
        return extracted, pose_result, hands_result, left_roi_box, right_roi_box


def draw_overlay(
    image_bgr,
    pose_result,
    hands_result,
    extracted,
    left_roi_box=None,
    right_roi_box=None,
    status="",
):
    overlay = image_bgr.copy()
    pose_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    left_hand_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    right_hand_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            overlay,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            pose_style,
            pose_style,
        )

    if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
        for hand_landmarks, handedness in zip(
            hands_result.multi_hand_landmarks,
            hands_result.multi_handedness,
        ):
            label = handedness.classification[0].label
            hand_style = left_hand_style if label == "Left" else right_hand_style
            mp_drawing.draw_landmarks(
                overlay,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_style,
                hand_style,
            )

    if left_roi_box is not None:
        x0, y0, x1, y1 = left_roi_box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(
            overlay,
            "Left ROI",
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
    if right_roi_box is not None:
        x0, y0, x1, y1 = right_roi_box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            "Right ROI",
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    cv2.putText(
        overlay,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"hand={int(extracted.any_hand_detected)} mp={extracted.elapsed_ms:.1f}ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return overlay


class OnlineHandCropper:
    def __init__(self, window, start_ratio, end_ratio):
        self.window = window
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.reset()

    def reset(self):
        self.frames = []
        self.start_idx = None
        self.last_detected_idx = None
        self.done = False

    @property
    def started(self):
        return self.start_idx is not None

    def update(self, extracted):
        if self.done:
            return None

        self.frames.append(extracted)
        detected = np.array([f.any_hand_detected for f in self.frames], dtype=bool)
        n = len(self.frames)

        if self.start_idx is None:
            if n >= self.window and detected[-self.window:].mean() >= self.start_ratio:
                self.start_idx = n - self.window
                self.last_detected_idx = self.start_idx
                for idx in range(self.start_idx, n):
                    if detected[idx]:
                        self.last_detected_idx = idx
            return None

        if extracted.any_hand_detected:
            self.last_detected_idx = n - 1

        if n >= self.window and (~detected[-self.window:]).mean() >= self.end_ratio:
            end_idx = self.last_detected_idx
            self.done = True
            return self.frames[self.start_idx:end_idx + 1]

        return None


def save_segment(segment, output_root, sample_name, fps):
    sample_dir = Path(output_root).expanduser().resolve() / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = sample_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    video_path = sample_dir / "segment.mp4"

    pose_np = np.stack([f.pose for f in segment]).astype(np.float32)
    left_hand_np = np.stack([f.left_hand for f in segment]).astype(np.float32)
    right_hand_np = np.stack([f.right_hand for f in segment]).astype(np.float32)

    np.save(sample_dir / "pose_33.npy", pose_np)
    np.save(sample_dir / "left_hand_21.npy", left_hand_np)
    np.save(sample_dir / "right_hand_21.npy", right_hand_np)

    frame_files = []
    for item in segment:
        frame_path = frame_dir / f"frame_{item.frame_idx:06d}.jpg"
        if not cv2.imwrite(str(frame_path), item.image_bgr):
            raise RuntimeError(f"Failed to save frame image: {frame_path}")
        frame_files.append(frame_path.relative_to(sample_dir).as_posix())

    first_frame = segment[0].image_bgr
    frame_h, frame_w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create mp4 video: {video_path}")

    try:
        for item in segment:
            frame = item.image_bgr
            if frame.shape[:2] != (frame_h, frame_w):
                frame = cv2.resize(frame, (frame_w, frame_h))
            writer.write(frame)
    finally:
        writer.release()

    summary_path = sample_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            "video,frame_idx,frame_file,left_hand_detected,right_hand_detected,"
            "roi_rescued_count,mean_pose_visibility,mediapipe_ms\n")
        for item, frame_file in zip(segment, frame_files):
            f.write(
                f"{sample_name},{item.frame_idx},{frame_file},"
                f"{item.left_hand_detected},"
                f"{item.right_hand_detected},{item.roi_rescued_count},"
                f"{item.mean_pose_visibility},{item.elapsed_ms}\n")

    return sample_dir, video_path, pose_np, left_hand_np, right_hand_np


def predict_from_npys(
    model,
    pipeline,
    pose_np,
    left_hand_np,
    right_hand_np,
    label_map,
    device,
    sample_name,
):
    start = time.perf_counter()
    pose = ensure_tvc(pose_np, NUM_POSE, "pose_np")
    total_frames = pose.shape[0]
    left = ensure_tvc(left_hand_np, NUM_HAND, "left_hand_np")
    left = pad_or_trim_time(left, total_frames)
    right = ensure_tvc(right_hand_np, NUM_HAND, "right_hand_np")
    right = pad_or_trim_time(right, total_frames)

    pose, pose_score = nan_to_zero_with_score(pose)
    left, left_score = nan_to_zero_with_score(left)
    right, right_score = nan_to_zero_with_score(right)

    keypoint = np.concatenate([pose, left, right], axis=1)
    keypoint_score = np.concatenate([pose_score, left_score, right_score], axis=1)
    sample = build_keypoint_sample(keypoint, keypoint_score, sample_name)
    build_ms = (time.perf_counter() - start) * 1000

    with torch.no_grad():
        start = time.perf_counter()
        data, data_batch = build_data_batch(sample, pipeline)
        result = model.test_step(data_batch)[0]
        synchronize(device)
        predict_ms = (time.perf_counter() - start) * 1000

    gloss_ids = to_int_list(result.pred_gloss)
    gloss_labels = map_gloss_ids(gloss_ids, label_map)
    return {
        "input_shape": tuple(data["inputs"].shape),
        "keypoint_build_ms": build_ms,
        "preprocess_prediction_ms": predict_ms,
        "gloss_ids": gloss_ids,
        "gloss_labels": gloss_labels,
    }


def load_model(config_path, checkpoint_path, device, cfg_options=None):
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    model = init_recognizer(
        cfg,
        checkpoint=str(Path(checkpoint_path).expanduser().resolve()),
        device=device,
    )
    return model, Compose(cfg.test_pipeline)


def main():
    args = parse_args()
    require_mediapipe()
    device = resolve_device(args.device)
    window = args.window if args.window is not None else int(args.fps * 0.5)
    sample_base = args.sample_name or datetime.now().strftime("webcam_%Y%m%d_%H%M%S")

    model, pipeline = load_model(
        args.config,
        args.checkpoint,
        device,
        cfg_options=args.cfg_options,
    )
    label_map = load_label_map(args.label_map)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    extractor = MediaPipeKeypointExtractor(
        use_roi=args.use_roi,
        roi_crop_size=args.roi_crop_size,
    )
    cropper = OnlineHandCropper(
        window=window,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
    )

    print("Press q to quit. Waiting for hand-detection start window...")
    print(
        f"Crop rule: window={window}, start_ratio={args.start_ratio}, "
        f"end_ratio={args.end_ratio}, use_roi={args.use_roi}")

    frame_idx = 0
    prediction_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            extracted, pose_result, hands_result, left_roi_box, right_roi_box = (
                extractor.process_frame(frame, frame_idx)
            )
            segment = cropper.update(extracted)

            status = "recording" if cropper.started else "waiting"
            if args.display:
                overlay = draw_overlay(
                    frame,
                    pose_result,
                    hands_result,
                    extracted,
                    left_roi_box=left_roi_box,
                    right_roi_box=right_roi_box,
                    status=status,
                )
                cv2.imshow("FairyTaleSL realtime ST-GCN-CTC", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if segment is not None:
                if len(segment) < args.min_cropped_frames:
                    print(f"Skipped short segment: {len(segment)} frames")
                    cropper.reset()
                    continue

                sample_name = sample_base
                if args.continuous:
                    sample_name = f"{sample_base}_{prediction_count:03d}"

                save_start = time.perf_counter()
                sample_dir, video_path, pose_np, left_np, right_np = save_segment(
                    segment,
                    args.output_dir,
                    sample_name,
                    args.fps,
                )
                save_ms = (time.perf_counter() - save_start) * 1000

                pred = predict_from_npys(
                    model=model,
                    pipeline=pipeline,
                    pose_np=pose_np,
                    left_hand_np=left_np,
                    right_hand_np=right_np,
                    label_map=label_map,
                    device=device,
                    sample_name=sample_name,
                )
                mediapipe_ms = sum(item.elapsed_ms for item in segment)

                print("\nSegment saved:", sample_dir)
                print("Frame jpg dir:", sample_dir / "frames")
                print("Segment mp4:", video_path)
                print("Device:", device)
                print("Frames:", len(segment))
                print("Input shape:", pred["input_shape"])
                print("MediaPipe total: {:.3f} ms".format(mediapipe_ms))
                print("Save npy time: {:.3f} ms".format(save_ms))
                print("Keypoint build time: {:.3f} ms".format(pred["keypoint_build_ms"]))
                print(
                    "Preprocess + prediction time: {:.3f} ms".format(
                        pred["preprocess_prediction_ms"]))
                print("Pred gloss ids:", pred["gloss_ids"])
                print("Pred gloss labels:", pred["gloss_labels"])

                prediction_count += 1
                if not args.continuous:
                    break
                cropper.reset()

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        extractor.close()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
