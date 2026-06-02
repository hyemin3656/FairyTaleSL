import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
MMACTION_ROOT = WORKSPACE_ROOT / "mmaction2"
CHECKPOINT_ROOT = WORKSPACE_ROOT / "checkpoints"

if str(MMACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MMACTION_ROOT))

from mmaction.apis import init_recognizer  # noqa: E402
from mmaction.utils import register_all_modules  # noqa: E402


DEFAULT_CONFIG = (
    MMACTION_ROOT
    / "configs/skeleton/cnn1d/"
    / "cnn1d_8xb16-joint-u100-50e_mediapipe-sign-keypoint-3d_without_face.py"
)
DEFAULT_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "best_acc_top1_epoch_36.pth"
)
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"

NUM_POSE_FULL = 33
NUM_POSE_USED = 23
NUM_HAND = 21
NUM_NODE = NUM_POSE_USED + NUM_HAND + NUM_HAND
COORD_DIM = 2
MP_COORD_DIM = 4


def landmarks_to_array(landmarks, num_points):
    arr = np.zeros((num_points, MP_COORD_DIM), dtype=np.float32)
    if landmarks is None:
        return arr

    for i, lm in enumerate(landmarks.landmark[:num_points]):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
        arr[i, 3] = lm.visibility if hasattr(lm, "visibility") else 1.0
    return arr


def interpolate_short_gaps(arr, frame_level_detection, max_gap=5):
    # 검출되지 않은 프레임이 max_gap 이하로 이어지는 경우 선형 보간으로 채우고, 보간된 프레임의 score는 0.5로 설정
    # arr: [T, V, C], frame_level_detection: [T], max_gap: int
    out = arr.copy()
    detected = np.asarray(frame_level_detection, dtype=bool)
    if out.shape[0] != detected.shape[0]:
        raise ValueError(
            f"Detection length mismatch: arr={out.shape[0]}, detected={detected.shape[0]}"
        )

    if detected.all() or not detected.any():
        return out, detected.copy()

    frame_detected_mask = detected.copy()
    for j in range(out.shape[1]):
        detected_for_val = out[:, j, 3] > 0 # (T,)
        if not np.array_equal(detected_for_val, detected):
            detected_for_val = detected

        if detected_for_val.all() or not detected_for_val.any():
            continue

        for c in range(3): # x, y, z
            series = pd.Series(out[:, j, c])
            series[~detected_for_val] = np.nan
            interp = series.interpolate(
                method="linear",
                limit=max_gap, #연속된 NaN이 limit 이하일 때만 해당 구간 보간
                limit_direction="both", #앞쪽/뒤쪽 양방향으로 보간을 허용
                limit_area="inside", #앞뒤에 정상값이 모두 있는 내부 구간만 보간
            )
            out[:, j, c] = interp.fillna(0).values #보간 불가능한 구간은 0으로 채움

        interpolated = (~detected_for_val) & ((out[:, j, :3] != 0).any(axis=1))
        frame_detected_mask[interpolated] = True
        out[interpolated, j, 3] = 0.5

    return out, frame_detected_mask


class MediaPipeWebcamExtractor:
    # 웹캠 프레임에서 MediaPipe Holistic 모델을 사용해 포즈와 양손 keypoint를 추출하는 클래스
    # 모델 초기화, 프레임 처리, 키포인트 array 변환
    def __init__(
        self,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self.holistic.close()

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        started = time.perf_counter()
        results = self.holistic.process(rgb)
        keypoint_ms = (time.perf_counter() - started) * 1000

        pose_detected = results.pose_landmarks is not None
        left_detected = results.left_hand_landmarks is not None
        right_detected = results.right_hand_landmarks is not None

        return {
            "pose": landmarks_to_array(results.pose_landmarks, NUM_POSE_FULL),
            "left_hand": landmarks_to_array(results.left_hand_landmarks, NUM_HAND),
            "right_hand": landmarks_to_array(results.right_hand_landmarks, NUM_HAND),
            "pose_detected": pose_detected,
            "left_hand_detected": left_detected,
            "right_hand_detected": right_detected,
            "any_hand_detected": left_detected or right_detected,
        }, keypoint_ms


class RealtimeSegmenter:
    # 웹캠 스트림에서 시작구간과 종료구간을 찾아 모델에 입력할 구간을 반환
    def __init__(self, window, start_ratio, end_ratio, max_record_frames, min_frames):
        self.window = window
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.max_record_frames = max_record_frames
        self.min_frames = min_frames
        self.reset()

    def reset(self):
        self.state = "waiting"
        self.pre_start = deque(maxlen=self.window)
        self.current = []
        self.last_detected_pos = None

    def update(self, frame_data):
        detected = frame_data["any_hand_detected"]

        if self.state == "waiting":
            self.pre_start.append(frame_data)
            # ===== 시작 시점 찾기 =====
            # window 크기만큼 최근 프레임을 모은 뒤, 손이 감지된 프레임의 비율이 start_ratio 이상이면 window내의 첫 detected 프레임 부터 기록 시작
            if len(self.pre_start) == self.window:
                ratio = np.mean([item["any_hand_detected"] for item in self.pre_start])
                if ratio >= self.start_ratio:
                    buffered = list(self.pre_start)
                    first_detected = next(
                        i for i, item in enumerate(buffered) if item["any_hand_detected"]
                    )
                    self.current = buffered[first_detected:]
                    self.last_detected_pos = max(
                        i
                        for i, item in enumerate(self.current)
                        if item["any_hand_detected"]
                    )
                    self.state = "recording"
                    return None, "started"
            return None, "waiting"

        self.current.append(frame_data)
        if detected:
            self.last_detected_pos = len(self.current) - 1

        recent = self.current[-self.window:]
        # ===== 종료 시점 찾기 =====
        # window 크기만큼 최근 프레임을 모은 뒤, 손이 감지되지 않은 프레임의 비율이 end_ratio 이상이면 recording 종료, 기록된 프레임 중 마지막 detected 프레임까지 segment로 반환
        if len(recent) == self.window:
            undetected_ratio = np.mean(
                [not item["any_hand_detected"] for item in recent]
            )
            if undetected_ratio >= self.end_ratio:
                return self._finish(), "finished"

        if len(self.current) >= self.max_record_frames:
            return self._finish(), "timeout"

        return None, "recording"

    def _finish(self):
        if self.last_detected_pos is None:
            segment = []
        else:
            segment = self.current[: self.last_detected_pos + 1]
        self.reset()

        if len(segment) < self.min_frames:
            return None
        return segment


class CNN1DRealtimeRecognizer:
    def __init__(self, config, checkpoint, label_map, device, topk, max_gap, cfg_options=None):
        register_all_modules(init_default_scope=True)
        self.cfg = Config.fromfile(config)
        if cfg_options is not None:
            self.cfg.merge_from_dict(cfg_options)
        self.pipeline = Compose(self.cfg.test_pipeline)
        self.model = init_recognizer(
            self.cfg,
            checkpoint=str(Path(checkpoint).expanduser().resolve()),
            device=device,
        )
        self.device = device
        self.topk = topk
        self.max_gap = max_gap
        self.label_map = self._load_label_map(label_map)

    @staticmethod
    def _load_label_map(path):
        path = Path(path).expanduser()
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict_segment(self, segment):
        started = time.perf_counter()
        sample = build_mmaction_sample(segment, max_gap=self.max_gap)
        data_build_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        data = self.pipeline(sample)
        data_batch = pseudo_collate([data])
        pipeline_ms = (time.perf_counter() - started) * 1000

        with torch.no_grad():
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_started = time.perf_counter()
            result = self.model.test_step(data_batch)[0]
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - infer_started) * 1000

        return {
            "predictions": self._format_predictions(result.pred_score),
            "input_shape": tuple(data["inputs"].shape),
            "data_build_ms": data_build_ms,
            "pipeline_ms": pipeline_ms,
            "inference_ms": infer_ms,
            "frames": len(segment),
        }

    def _format_predictions(self, pred_score):
        topk = min(self.topk, pred_score.numel())
        scores, indices = pred_score.topk(topk)
        predictions = []
        for class_id, score in zip(indices.tolist(), scores.tolist()):
            predictions.append(
                {
                    "class_id": int(class_id),
                    "label": self.label_map.get(str(int(class_id)), str(int(class_id))),
                    "score": float(score),
                }
            )
        return predictions


def stack_segment_arrays(segment, key):
    return np.stack([frame_data[key] for frame_data in segment]).astype(np.float32)


def build_mmaction_sample(segment, max_gap=5):
    pose = stack_segment_arrays(segment, "pose")[:, :NUM_POSE_USED]
    left = stack_segment_arrays(segment, "left_hand")
    right = stack_segment_arrays(segment, "right_hand")

    pose_detected = np.asarray([item["pose_detected"] for item in segment], dtype=bool)
    left_detected = np.asarray(
        [item["left_hand_detected"] for item in segment], dtype=bool
    )
    right_detected = np.asarray(
        [item["right_hand_detected"] for item in segment], dtype=bool
    )

    pose, pose_detected = interpolate_short_gaps(pose, pose_detected, max_gap=max_gap)
    left, left_detected = interpolate_short_gaps(left, left_detected, max_gap=max_gap)
    right, right_detected = interpolate_short_gaps(
        right, right_detected, max_gap=max_gap
    )

    keypoint = np.concatenate([pose[..., :COORD_DIM], left[..., :COORD_DIM], right[..., :COORD_DIM]], axis=1)
    keypoint_score = np.concatenate(
        [
            np.ones((pose.shape[0], NUM_POSE_USED), dtype=np.float32),
            left[..., 3],
            right[..., 3],
        ],
        axis=1,
    )

    if np.isnan(keypoint).any() or np.isnan(keypoint_score).any():
        raise ValueError("NaN values found in keypoint or keypoint_score arrays")
    # keypoint = np.nan_to_num(keypoint, nan=0.0, posinf=0.0, neginf=0.0)
    # keypoint_score = np.nan_to_num(keypoint_score, nan=0.0, posinf=0.0, neginf=0.0)

    total_frames = keypoint.shape[0]
    if keypoint.shape != (total_frames, NUM_NODE, COORD_DIM):
        raise ValueError(f"Unexpected keypoint shape: {keypoint.shape}")
    if keypoint_score.shape != (total_frames, NUM_NODE):
        raise ValueError(f"Unexpected keypoint_score shape: {keypoint_score.shape}")

    return {
        "frame_dir": "webcam",
        "total_frames": total_frames,
        "label": -1,
        "keypoint": keypoint[None, ...].astype(np.float32),
        "keypoint_score": keypoint_score[None, ...].astype(np.float32),
    }


def draw_status(frame, state, fps, latest_result):
    color = (0, 220, 0) if state == "recording" else (255, 255, 255)
    cv2.putText(
        frame,
        f"{state} | {fps:.1f} FPS",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    if latest_result:
        y = 64
        for rank, pred in enumerate(latest_result["predictions"][:3], start=1):
            text = f"top{rank} {pred['label']} {pred['score']:.3f}"
            cv2.putText(
                frame,
                text,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime webcam test for MediaPipe keypoints + CNN1D MMACTION model."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--device", default="auto", help="Use auto, cpu, or a torch device such as cuda:0.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument("--start-ratio", type=float, default=0.8)
    parser.add_argument("--end-ratio", type=float, default=0.8)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-record-sec", type=float, default=5.0)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Reserved for parity with MMACTION scripts; edit config file for pipeline changes.",
    )
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is not available. Falling back from {device} to cpu.")
        return "cpu"
    return device

def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window = max(1, int(round(args.fps * args.window_sec))) 
    max_record_frames = max(window, int(round(args.fps * args.max_record_sec)))

    extractor = MediaPipeWebcamExtractor(model_complexity=args.model_complexity)
    recognizer = CNN1DRealtimeRecognizer(
        args.config,
        args.checkpoint,
        args.label_map,
        device,
        args.topk,
        args.max_gap,
        args.cfg_options,
    )
    segmenter = RealtimeSegmenter(
        window=window,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
        max_record_frames=max_record_frames,
        min_frames=args.min_frames,
    )

    latest_result = None
    fps_meter = deque(maxlen=30)
    print("Press q to quit. Waiting for hand motion...")

    try:
        keypoint_ms = []
        while True:
            loop_started = time.perf_counter()
            ret, frame = cap.read()
            captured_at = time.perf_counter()
            if not ret:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_data, keypoint_ms_per_frame = extractor.process(frame)
            frame_data["captured_at"] = captured_at
            keypoint_ms.append(keypoint_ms_per_frame)
            segment, event = segmenter.update(frame_data)

            if event == "started":
                print("Recording segment...")
            elif event in {"finished", "timeout"}:
                if segment is None:
                    print("Segment ignored: too short or no valid hand detection.")
                else:
                    latest_result = recognizer.predict_segment(segment)
                    prediction_returned_at = time.perf_counter()
                    last_detected_capture_to_infer_end_ms = (
                        prediction_returned_at - segment[-1]["captured_at"]
                    ) * 1000
                    best = latest_result["predictions"][0]
                    print(
                        f"{event}: frames={latest_result['frames']} "
                        f"input={latest_result['input_shape']} "
                        f"keypoint={np.mean(keypoint_ms):.2f}ms "
                        f"data_build={latest_result['data_build_ms']:.2f}ms "
                        f"pipeline={latest_result['pipeline_ms']:.2f}ms "
                        f"infer={latest_result['inference_ms']:.2f}ms "
                        f"last_detected_capture_to_infer_end="
                        f"{last_detected_capture_to_infer_end_ms:.2f}ms "
                        f"top1={best['label']}({best['class_id']}) "
                        f"score={best['score']:.4f}"
                    )

            fps_meter.append(1.0 / max(time.perf_counter() - loop_started, 1e-6))
            if args.show:
                fps = float(np.mean(fps_meter)) if fps_meter else 0.0
                draw_status(frame, segmenter.state, fps, latest_result)
                cv2.imshow("FairyTaleSL CNN1D realtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        extractor.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
