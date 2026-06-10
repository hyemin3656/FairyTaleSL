import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CHECKPOINT_ROOT = WORKSPACE_ROOT / "checkpoints"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.builder import build_model  # noqa: E402
from model.config_utils import load_config  # noqa: E402
from model.data import preprocess_keypoint_sample  # noqa: E402
from model.model import load_checkpoint  # noqa: E402

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_CONFIG = (
    WORKSPACE_ROOT
    / "FairyTaleSL/model/configs/cnn1d_mediapipe_sign_without_face.py"
)
DEFAULT_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "best.pth"
)
DEFAULT_LABEL_MAP = PROJECT_ROOT / "src/class_labels.json"
DEFAULT_SAVE_IMAGES_DIR = PROJECT_ROOT / "saved_inference_windows"

NUM_POSE_FULL = 33
NUM_POSE_USED = 23
NUM_HAND = 21
NUM_NODE = NUM_POSE_USED + NUM_HAND + NUM_HAND
COORD_DIM = 3
MP_COORD_DIM = 4
TOP1_SCORE_THRESHOLD = 0.5
KOREAN_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
]
_DISPLAY_FONT = None


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
    def __init__(
        self,
        window_sec,
        start_ratio,
        end_ratio,
        max_record_sec,
        min_frames,
        sequence_level_detection=False,
        sequence_window_frames=90,
        sequence_stride_frames=20,
    ):
        self.window_sec = window_sec
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.max_record_sec = max_record_sec
        self.min_frames = min_frames
        self.sequence_level_detection = sequence_level_detection
        self.sequence_window_frames = sequence_window_frames
        self.sequence_stride_frames = sequence_stride_frames
        self.reset()

    def reset(self):
        self.state = "waiting"
        self.pre_start = deque()
        self.current = []
        self.last_detected_pos = None
        self.next_sequence_start = 0
        self.sequence_window_count = 0

    def _timestamp(self, frame_data):
        return frame_data.get("captured_at", time.perf_counter())

    def _window_ready(self, frames, now=None):
        if not frames:
            return False
        if now is None:
            now = self._timestamp(frames[-1])
        return now - self._timestamp(frames[0]) >= self.window_sec

    def _prune_before_window(self, frames, now):
        cutoff = now - self.window_sec
        while len(frames) > 1 and self._timestamp(frames[1]) < cutoff:
            frames.popleft()

    def _recent_time_window(self, frames, now):
        cutoff = now - self.window_sec
        start = 0
        for index, item in enumerate(frames):
            if self._timestamp(item) >= cutoff:
                start = max(0, index - 1)
                break
        return frames[start:]

    def update(self, frame_data):
        detected = frame_data["any_hand_detected"]
        now = self._timestamp(frame_data)

        if self.state == "waiting":
            self.pre_start.append(frame_data)
            self._prune_before_window(self.pre_start, now)
            # ===== 시작 시점 찾기 =====
            # window 크기만큼 최근 프레임을 모은 뒤, 손이 감지된 프레임의 비율이 start_ratio 이상이면 window내의 첫 detected 프레임 부터 기록 시작
            if self._window_ready(self.pre_start, now):
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
                    if self.sequence_level_detection:
                        return self._next_sequence_windows(), "started"
                    return None, "started"
            return None, "waiting"

        self.current.append(frame_data)
        if detected:
            self.last_detected_pos = len(self.current) - 1

        recent = self._recent_time_window(self.current, now)
        # ===== 종료 시점 찾기 =====
        # window 크기만큼 최근 프레임을 모은 뒤, 손이 감지되지 않은 프레임의 비율이 end_ratio 이상이면 recording 종료, 기록된 프레임 중 마지막 detected 프레임까지 segment로 반환
        if self._window_ready(recent, now):
            undetected_ratio = np.mean(
                [not item["any_hand_detected"] for item in recent]
            )
            if undetected_ratio >= self.end_ratio:
                if self.sequence_level_detection:
                    windows = self._next_sequence_windows(final=True)
                    self.reset()
                    return windows, "finished"
                return self._finish(), "finished"

        record_duration = self._timestamp(self.current[-1]) - self._timestamp(self.current[0])
        if record_duration >= self.max_record_sec:
            if self.sequence_level_detection:
                windows = self._next_sequence_windows(final=True)
                self.reset()
                return windows, "timeout"
            return self._finish(), "timeout"

        if self.sequence_level_detection:
            return self._next_sequence_windows(), "recording"
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

    def _build_sequence_window(self, start, window_type="regular"):
        end = start + self.sequence_window_frames
        return {
            "index": self.sequence_window_count,
            "start": start,
            "end": end - 1,
            "segment": self.current[start:end],
            "window_type": window_type,
        }

    def _next_sequence_windows(self, final=False):
        if self.last_detected_pos is None:
            return []

        windows = []
        valid_frame_count = self.last_detected_pos + 1
        while self.next_sequence_start + self.sequence_window_frames <= valid_frame_count:
            start = self.next_sequence_start
            windows.append(self._build_sequence_window(start))
            self.sequence_window_count += 1
            self.next_sequence_start += self.sequence_stride_frames

        if final and valid_frame_count >= self.sequence_window_frames:
            final_start = valid_frame_count - self.sequence_window_frames
            if final_start >= self.next_sequence_start:
                windows.append(
                    self._build_sequence_window(final_start, window_type="final")
                )
                self.sequence_window_count += 1
                self.next_sequence_start = final_start + self.sequence_stride_frames

        return windows


class CNN1DRealtimeRecognizer:
    def __init__(self, config, checkpoint, label_map, device, topk, max_gap, cfg_options=None):
        if cfg_options is not None:
            print("--cfg-options is ignored for standalone FairyTaleSL/model inference.")
        self.cfg = load_config(config)
        self.device = device
        self.model = build_model(self.cfg).to(self.device)
        checkpoint_info = load_checkpoint(
            self.model,
            Path(checkpoint).expanduser().resolve(),
            map_location=self.device,
            strict=False,
        )
        self.model.eval()
        if checkpoint_info["missing_keys"] or checkpoint_info["unexpected_keys"]:
            print(
                "Checkpoint loaded with "
                f"missing_keys={len(checkpoint_info['missing_keys'])}, "
                f"unexpected_keys={len(checkpoint_info['unexpected_keys'])}"
            )
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
        sample = build_keypoint_sample(segment, max_gap=self.max_gap)
        data_build_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        keypoint = preprocess_keypoint_sample(
            sample,
            clip_len=self.cfg.CLIP_LEN,
            num_clips=getattr(self.cfg, "TEST_NUM_CLIPS", 1),
            test_mode=True,
            zero_pad_short=getattr(self.cfg, "ZERO_PAD_SHORT", False),
            input_mode=getattr(self.cfg, "INPUT_MODE", "xy"),
            keypoint_normalize=getattr(self.cfg, "KEYPOINT_NORMALIZE", None),
            short_sample_interpolation=getattr(self.cfg, "SHORT_SAMPLE_INTERPOLATION", None),
        )
        inputs = torch.from_numpy(keypoint[None]).to(self.device)
        pipeline_ms = (time.perf_counter() - started) * 1000

        with torch.no_grad():
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_started = time.perf_counter()
            scores = self.model.predict(inputs)[0]
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - infer_started) * 1000

        return {
            "predictions": self._format_predictions(scores),
            "input_shape": tuple(inputs.shape),
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


def build_keypoint_sample(segment, max_gap=5):
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


def _landmark_to_pixel(landmark, width, height, min_score=0.0):
    x, y = float(landmark[0]), float(landmark[1])
    score = float(landmark[3]) if landmark.shape[0] > 3 else 1.0
    if min_score is not None and score <= min_score:
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


def draw_landmark_group(frame, landmarks, connections, color, min_score=0.0):
    height, width = frame.shape[:2]
    points = [
        _landmark_to_pixel(landmark, width, height, min_score=min_score)
        for landmark in landmarks
    ]

    for start, end in connections:
        if start >= len(points) or end >= len(points):
            continue
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)

    for point in points:
        if point is not None:
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)


def draw_keypoints(frame, frame_data):
    if frame_data["pose_detected"]:
        draw_landmark_group(
            frame,
            frame_data["pose"],
            mp.solutions.pose.POSE_CONNECTIONS,
            (255, 0, 0),
            min_score=0.0,
        )
    if frame_data["left_hand_detected"]:
        draw_landmark_group(
            frame,
            frame_data["left_hand"],
            mp.solutions.hands.HAND_CONNECTIONS,
            (0, 255, 0),
            min_score=None,
        )
    if frame_data["right_hand_detected"]:
        draw_landmark_group(
            frame,
            frame_data["right_hand"],
            mp.solutions.hands.HAND_CONNECTIONS,
            (0, 0, 255),
            min_score=None,
        )
    return frame


def postprocess_sequence_top1(top1_results):
    gloss_sequence = []
    previous_label = None

    for result in top1_results:
        label = result["label"] if isinstance(result, dict) else str(result)
        if label == previous_label:
            continue
        gloss_sequence.append(label)
        previous_label = label

    return gloss_sequence


def save_window_images(item):
    save_dir = item.get("save_images_dir")
    if save_dir is None:
        return

    window_dir = (
        Path(save_dir)
        / f"sequence_{item['sequence_id']:04d}"
        / f"window_{item['index']:04d}"
    )
    window_dir.mkdir(parents=True, exist_ok=True)

    for offset, frame_data in enumerate(item["segment"]):
        image = frame_data.get("image")
        if image is None:
            continue
        frame_index = frame_data.get("frame_index", offset)
        cv2.imwrite(str(window_dir / f"frame_{frame_index:06d}.jpg"), image)


def prediction_worker(recognizer, request_queue, result_queue):
    while True:
        item = request_queue.get()
        try:
            if item is None:
                return

            result = recognizer.predict_segment(item["segment"])
            best = result["predictions"][0]
            if best["score"] > TOP1_SCORE_THRESHOLD:
                save_window_images(item)
            result_queue.put(
                {
                    "ok": True,
                    "item": item,
                    "result": result,
                    "prediction_returned_at": time.perf_counter(),
                }
            )
        except Exception as exc:
            result_queue.put({"ok": False, "item": item, "error": exc})
        finally:
            request_queue.task_done()


def drain_queue(items_queue):
    items = []
    while True:
        try:
            items.append(items_queue.get_nowait())
        except queue.Empty:
            return items


def get_display_font(size=22):
    global _DISPLAY_FONT
    if _DISPLAY_FONT is not None:
        return _DISPLAY_FONT
    if ImageFont is None:
        return None

    for font_path in KOREAN_FONT_CANDIDATES:
        if font_path.exists():
            _DISPLAY_FONT = ImageFont.truetype(str(font_path), size)
            return _DISPLAY_FONT

    return None


def draw_unicode_text(frame, text, position, color, font_size=22):
    font = get_display_font(font_size)
    if Image is None or ImageDraw is None or font is None:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return frame


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
            draw_unicode_text(
                frame,
                text,
                (16, y),
                (0, 255, 255),
                font_size=22,
            )
            y += 28
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime webcam test for MediaPipe keypoints + standalone FairyTaleSL CNN1D model."
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
    parser.add_argument("--max-record-sec", type=float, default=10.0)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument(
        "--sequence-level-detection",
        action="store_true",
        help="Run sliding-window recognition while recording instead of one recognition after the segment ends.",
    )
    parser.add_argument(
        "--sequence-window-frames",
        type=int,
        default=90,
        help="Number of frames per sliding-window model input.",
    )
    parser.add_argument(
        "--sequence-stride-frames",
        type=int,
        default=20,
        help="Frame interval between sliding-window starts.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save frames used for each inference window as jpg files.",
    )
    parser.add_argument(
        "--save-images-dir",
        default=str(DEFAULT_SAVE_IMAGES_DIR),
        help="Directory for saved inference window images.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        help="Ignored. Kept only for compatibility with old MMACTION realtime commands.",
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
    if args.sequence_window_frames <= 0:
        raise ValueError("--sequence-window-frames must be greater than 0")
    if args.sequence_stride_frames <= 0:
        raise ValueError("--sequence-stride-frames must be greater than 0")
    save_images_dir = Path(args.save_images_dir).expanduser()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        window_sec=args.window_sec,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
        max_record_sec=args.max_record_sec,
        min_frames=args.min_frames,
        sequence_level_detection=args.sequence_level_detection,
        sequence_window_frames=args.sequence_window_frames,
        sequence_stride_frames=args.sequence_stride_frames,
    )

    latest_result = None
    sequence_top1_results = {}
    sequence_pending_counts = {}
    sequence_end_events = {}
    current_sequence_id = 0
    non_sequence_window_index = 0
    fps_meter = deque(maxlen=30)
    frame_index = 0
    prediction_requests = None
    prediction_results = None
    prediction_thread = None
    if args.sequence_level_detection:
        prediction_requests = queue.Queue()
        prediction_results = queue.Queue()
        prediction_thread = threading.Thread(
            target=prediction_worker,
            args=(recognizer, prediction_requests, prediction_results),
            daemon=True,
        )
        prediction_thread.start()
    print("Press q to quit. Waiting for hand motion...")

    def print_finished_sequence(sequence_id, event_name):
        gloss_sequence = postprocess_sequence_top1(
            sequence_top1_results.get(sequence_id, [])
        )
        print(
            f"{event_name}: sequence ended. "
            f"gloss_sequence={' '.join(gloss_sequence)}"
        )
        sequence_top1_results.pop(sequence_id, None)
        sequence_pending_counts.pop(sequence_id, None)
        sequence_end_events.pop(sequence_id, None)

    def handle_prediction_results():
        nonlocal latest_result
        if prediction_results is None:
            return

        for message in drain_queue(prediction_results):
            item = message["item"]
            sequence_id = item["sequence_id"]
            sequence_pending_counts[sequence_id] = max(
                0, sequence_pending_counts.get(sequence_id, 0) - 1
            )

            if not message["ok"]:
                print(
                    f"{item['event']}: window={item['index']} "
                    f"inference failed: {message['error']}"
                )
                if (
                    sequence_pending_counts.get(sequence_id, 0) == 0
                    and sequence_id in sequence_end_events
                ):
                    print_finished_sequence(
                        sequence_id, sequence_end_events[sequence_id]
                    )
                continue

            result = message["result"]
            segment = item["segment"]
            best = result["predictions"][0]
            if best["score"] <= TOP1_SCORE_THRESHOLD:
                if (
                    sequence_pending_counts.get(sequence_id, 0) == 0
                    and sequence_id in sequence_end_events
                ):
                    print_finished_sequence(
                        sequence_id, sequence_end_events[sequence_id]
                    )
                continue

            latest_result = result
            sequence_top1_results.setdefault(sequence_id, []).append(best)
            last_detected_capture_to_infer_end_ms = (
                message["prediction_returned_at"] - segment[-1]["captured_at"]
            ) * 1000
            print(
                f"{item['event']}: window={item['index']} "
                f"type={item.get('window_type', 'regular')} "
                f"frames={result['frames']} "
                f"frame_range={segment[0]['frame_index']}-"
                f"{segment[-1]['frame_index']} "
                f"input={result['input_shape']} "
                f"keypoint={np.mean(keypoint_ms):.2f}ms "
                f"data_build={result['data_build_ms']:.2f}ms "
                f"pipeline={result['pipeline_ms']:.2f}ms "
                f"infer={result['inference_ms']:.2f}ms "
                f"last_detected_capture_to_infer_end="
                f"{last_detected_capture_to_infer_end_ms:.2f}ms "
                f"top1={best['label']}({best['class_id']}) "
                f"score={best['score']:.4f}"
            )
            if (
                sequence_pending_counts.get(sequence_id, 0) == 0
                and sequence_id in sequence_end_events
            ):
                print_finished_sequence(sequence_id, sequence_end_events[sequence_id])

    try:
        keypoint_ms = []
        while True:
            handle_prediction_results()
            loop_started = time.perf_counter()
            ret, frame = cap.read()
            captured_at = time.perf_counter()
            if not ret:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_data, keypoint_ms_per_frame = extractor.process(frame)
            frame_data["captured_at"] = captured_at
            frame_data["frame_index"] = frame_index
            if args.save_images:
                frame_data["image"] = frame.copy()
            frame_index += 1
            keypoint_ms.append(keypoint_ms_per_frame)
            segment_result, event = segmenter.update(frame_data)

            if event == "started":
                current_sequence_id += 1
                sequence_top1_results[current_sequence_id] = []
                sequence_pending_counts[current_sequence_id] = 0
                sequence_end_events.pop(current_sequence_id, None)
                print("Recording segment...")

            if args.sequence_level_detection:
                windows = segment_result or []
                for window_info in windows:
                    prediction_item = dict(window_info)
                    prediction_item["event"] = event
                    prediction_item["sequence_id"] = current_sequence_id
                    if args.save_images:
                        prediction_item["save_images_dir"] = save_images_dir
                    sequence_pending_counts[current_sequence_id] = (
                        sequence_pending_counts.get(current_sequence_id, 0) + 1
                    )
                    prediction_requests.put(prediction_item)
                if event in {"finished", "timeout"}:
                    sequence_end_events[current_sequence_id] = event
                    if sequence_pending_counts.get(current_sequence_id, 0) == 0:
                        print_finished_sequence(current_sequence_id, event)
                handle_prediction_results()
            elif event in {"finished", "timeout"}:
                segment = segment_result
                if segment is None:
                    print("Segment ignored: too short or no valid hand detection.")
                else:
                    result = recognizer.predict_segment(segment)
                    prediction_returned_at = time.perf_counter()
                    best = result["predictions"][0]
                    if best["score"] > TOP1_SCORE_THRESHOLD:
                        if args.save_images:
                            save_window_images(
                                {
                                    "save_images_dir": save_images_dir,
                                    "sequence_id": 0,
                                    "index": non_sequence_window_index,
                                    "segment": segment,
                                }
                            )
                            non_sequence_window_index += 1
                        latest_result = result
                        last_detected_capture_to_infer_end_ms = (
                            prediction_returned_at - segment[-1]["captured_at"]
                        ) * 1000
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
                draw_keypoints(frame, frame_data)
                draw_status(frame, segmenter.state, fps, latest_result)
                cv2.imshow("FairyTaleSL CNN1D realtime", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if prediction_requests is not None and prediction_thread is not None:
            prediction_requests.put(None)
            prediction_thread.join()
            handle_prediction_results()
        extractor.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
