import argparse
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


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
            "mediapipe is required. Install local dependencies with: "
            "pip install -r requirements-local.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture local webcam keypoints, crop by the training hand-detection "
            "rule, save pose/hand npy files, and upload them to the GPU server."))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-dir", default="./captured_segments")
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
        "--upload-target",
        default=None,
        help=(
            "Optional scp destination, e.g. "
            "ubuntu@YOUR_SERVER:/home/ubuntu/FairyTaleSL/realtime_inputs/"))
    parser.add_argument("--ssh-key", default=None, help="Optional SSH private key path.")
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("--scp-bin", default="scp")
    parser.add_argument("--no-upload", action="store_true")
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


def detect_hand_in_roi(image_bgr, hands_roi_model, wrist_x_px, wrist_y_px, crop_size=256):
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
        hand_arr[i] = [lm.x * crop_w + x0, lm.y * crop_h + y0, lm.z, 1.0]

    return hand_arr, (x0, y0, x1, y1)


@dataclass
class ExtractedFrame:
    frame_idx: int
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
        mean_pose_visibility = np.nanmean(pose_arr[:, 3])

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


def draw_overlay(image_bgr, pose_result, hands_result, extracted, left_roi_box, right_roi_box, status):
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

    for box, color, text in [
        (left_roi_box, (255, 0, 0), "Left ROI"),
        (right_roi_box, (0, 0, 255), "Right ROI"),
    ]:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay,
            text,
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
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


def save_segment(segment, output_root, sample_name):
    sample_dir = Path(output_root).expanduser().resolve() / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    pose_np = np.stack([f.pose for f in segment]).astype(np.float32)
    left_hand_np = np.stack([f.left_hand for f in segment]).astype(np.float32)
    right_hand_np = np.stack([f.right_hand for f in segment]).astype(np.float32)

    np.save(sample_dir / "pose_33.npy", pose_np)
    np.save(sample_dir / "left_hand_21.npy", left_hand_np)
    np.save(sample_dir / "right_hand_21.npy", right_hand_np)

    with (sample_dir / "summary.csv").open("w", encoding="utf-8") as f:
        f.write(
            "video,frame_idx,left_hand_detected,right_hand_detected,"
            "roi_rescued_count,mean_pose_visibility,mediapipe_ms\n")
        for item in segment:
            f.write(
                f"{sample_name},{item.frame_idx},{item.left_hand_detected},"
                f"{item.right_hand_detected},{item.roi_rescued_count},"
                f"{item.mean_pose_visibility},{item.elapsed_ms}\n")

    return sample_dir


def upload_segment(sample_dir, target, scp_bin="scp", ssh_key=None, ssh_port=None):
    cmd = [scp_bin, "-r"]
    if ssh_key:
        cmd.extend(["-i", str(Path(ssh_key).expanduser())])
    if ssh_port:
        cmd.extend(["-P", str(ssh_port)])
    cmd.extend([str(sample_dir), target])

    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return (time.perf_counter() - start) * 1000


def main():
    args = parse_args()
    require_mediapipe()

    window = args.window if args.window is not None else int(args.fps * 0.5)
    sample_base = args.sample_name or datetime.now().strftime("webcam_%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open local webcam index {args.camera}. Try another index, "
            "or check whether another app is using the camera.")

    extractor = MediaPipeKeypointExtractor(
        use_roi=args.use_roi,
        roi_crop_size=args.roi_crop_size,
    )
    cropper = OnlineHandCropper(window, args.start_ratio, args.end_ratio)

    print("Local webcam capture started. Press q to quit.")
    print(
        f"Crop rule: window={window}, start_ratio={args.start_ratio}, "
        f"end_ratio={args.end_ratio}, use_roi={args.use_roi}")
    if args.upload_target and not args.no_upload:
        print("Upload target:", args.upload_target)
    else:
        print("Upload disabled. Use --upload-target to send results to server.")

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
                    left_roi_box,
                    right_roi_box,
                    status,
                )
                cv2.imshow("FairyTaleSL local capture", overlay)
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
                sample_dir = save_segment(segment, args.output_dir, sample_name)
                save_ms = (time.perf_counter() - save_start) * 1000

                print("\nSaved segment:", sample_dir)
                print("Frames:", len(segment))
                print("MediaPipe total: {:.3f} ms".format(sum(f.elapsed_ms for f in segment)))
                print("Save npy time: {:.3f} ms".format(save_ms))

                if args.upload_target and not args.no_upload:
                    upload_ms = upload_segment(
                        sample_dir,
                        args.upload_target,
                        scp_bin=args.scp_bin,
                        ssh_key=args.ssh_key,
                        ssh_port=args.ssh_port,
                    )
                    print("Upload time: {:.3f} ms".format(upload_ms))
                    print("Uploaded to:", args.upload_target)

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
