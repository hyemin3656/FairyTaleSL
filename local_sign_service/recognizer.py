"""
recognizer.py — hyemin 브랜치의 인식 클래스를 import-friendly하게 재패키징.

원본: hyemin/webcam_cnn1d_realtime.py (930 LOC, 메인 루프 포함)
여기선 라이브러리로 쓸 수 있게 메인 루프(cv2.imshow 등)를 제거하고
3개 클래스 + 핵심 helper만 노출.

  - MediaPipeWebcamExtractor : 프레임 → MediaPipe Pose + Hands 키포인트
  - RealtimeSegmenter        : 손 검출 비율 기반 segment 시작/종료 판정
  - CNN1DRealtimeRecognizer  : MMAction2 CNN1D 모델 추론 wrapper
  - build_mmaction_sample()  : segment → 모델 입력 dict

원본 코드 변경 최소화 — 박혜민 의도/하이퍼파라미터 그대로 유지.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# mmaction2/mmengine은 사이드카 venv에 설치 필요 (README 참조)
# 별도 워크스페이스에 mmaction2 폴더가 있으면 추가
_PROJECT_ROOT = Path(__file__).resolve().parent
_MMACTION_ROOT = _PROJECT_ROOT.parent / "mmaction2"
if _MMACTION_ROOT.exists() and str(_MMACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMACTION_ROOT))

# ── 원본 상수 (그대로 복사) ─────────────────────────────────────────────
NUM_POSE_FULL = 33
NUM_POSE_USED = 23
NUM_HAND = 21
NUM_NODE = NUM_POSE_USED + NUM_HAND + NUM_HAND   # 65
COORD_DIM = 2
MP_COORD_DIM = 4
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
HAND_WRIST = 0
POSE_WRIST_MIN_VISIBILITY = 0.2
DEFAULT_HAND_POSE_MAX_DISTANCE = 0.25
TOP1_SCORE_THRESHOLD = 0.2
POSE_DRAW_COLOR = (255, 0, 0)
LEFT_HAND_DRAW_COLOR = (0, 255, 0)
RIGHT_HAND_DRAW_COLOR = (0, 0, 255)


# ── helper 함수들 (원본 그대로) ────────────────────────────────────────
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


def _landmark_to_pixel(landmark, width, height, min_score=0.0):
    x, y = float(landmark[0]), float(landmark[1])
    score = float(landmark[3]) if landmark.shape[0] > 3 else 1.0
    if min_score is not None and score <= min_score:
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


def draw_landmark_group(frame, landmarks, connections, color, min_score=0.0):
    import cv2

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
    import mediapipe as mp

    if frame_data["pose_detected"]:
        draw_landmark_group(
            frame,
            frame_data["pose"][:NUM_POSE_USED],
            mp.solutions.pose.POSE_CONNECTIONS,
            POSE_DRAW_COLOR,
            min_score=0.0,
        )
    if frame_data["left_hand_detected"]:
        draw_landmark_group(
            frame,
            frame_data["left_hand"],
            mp.solutions.hands.HAND_CONNECTIONS,
            LEFT_HAND_DRAW_COLOR,
            min_score=None,
        )
    if frame_data["right_hand_detected"]:
        draw_landmark_group(
            frame,
            frame_data["right_hand"],
            mp.solutions.hands.HAND_CONNECTIONS,
            RIGHT_HAND_DRAW_COLOR,
            min_score=None,
        )
    return frame


def interpolate_short_gaps(arr, frame_level_detection, max_gap=10):
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
        detected_for_val = out[:, j, 3] > 0
        if not np.array_equal(detected_for_val, detected):
            detected_for_val = detected
        if detected_for_val.all() or not detected_for_val.any():
            continue
        for c in range(3):
            series = pd.Series(out[:, j, c])
            series[~detected_for_val] = np.nan
            interp = series.interpolate(
                method="linear", limit=max_gap,
                limit_direction="both", limit_area="inside",
            )
            out[:, j, c] = interp.fillna(0).values
        interpolated = (~detected_for_val) & ((out[:, j, :3] != 0).any(axis=1))
        frame_detected_mask[interpolated] = True
        out[interpolated, j, 3] = 0.5
    return out, frame_detected_mask


def stack_segment_arrays(segment, key):
    return np.stack([frame_data[key] for frame_data in segment]).astype(np.float32)


def build_mmaction_sample(segment, max_gap=10):
    """segment list → MMAction2 모델 입력 dict."""
    pose = stack_segment_arrays(segment, "pose")[:, :NUM_POSE_USED]
    left = stack_segment_arrays(segment, "left_hand")
    right = stack_segment_arrays(segment, "right_hand")

    pose_detected = np.asarray([item["pose_detected"] for item in segment], dtype=bool)
    left_detected = np.asarray([item["left_hand_detected"] for item in segment], dtype=bool)
    right_detected = np.asarray([item["right_hand_detected"] for item in segment], dtype=bool)

    pose, _ = interpolate_short_gaps(pose, pose_detected, max_gap=max_gap)
    left, _ = interpolate_short_gaps(left, left_detected, max_gap=max_gap)
    right, _ = interpolate_short_gaps(right, right_detected, max_gap=max_gap)

    keypoint = np.concatenate(
        [pose[..., :COORD_DIM], left[..., :COORD_DIM], right[..., :COORD_DIM]], axis=1
    )
    keypoint_score = np.concatenate(
        [
            np.ones((pose.shape[0], NUM_POSE_USED), dtype=np.float32),
            left[..., 3],
            right[..., 3],
        ],
        axis=1,
    )
    if np.isnan(keypoint).any() or np.isnan(keypoint_score).any():
        raise ValueError("NaN in keypoint/score")
    total_frames = keypoint.shape[0]
    return {
        "frame_dir": "webcam",
        "total_frames": total_frames,
        "label": -1,
        "keypoint": keypoint[None, ...].astype(np.float32),
        "keypoint_score": keypoint_score[None, ...].astype(np.float32),
    }


# ── MediaPipe 추출기 ───────────────────────────────────────────────────
class MediaPipeWebcamExtractor:
    """웹캠 프레임 → pose 33 + lhand 21 + rhand 21 키포인트."""

    def __init__(
        self,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        hand_pose_max_distance: float = DEFAULT_HAND_POSE_MAX_DISTANCE,
    ) -> None:
        import mediapipe as mp
        self._mp = mp
        self.hand_pose_max_distance = hand_pose_max_distance
        self.hand_pose_max_distance_sq = hand_pose_max_distance * hand_pose_max_distance
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self.pose.close()
        self.hands.close()

    @staticmethod
    def _split_hands(hand_results):
        left_hand = None
        right_hand = None
        left_score = -1.0
        right_score = -1.0

        landmarks = hand_results.multi_hand_landmarks or []
        handedness = hand_results.multi_handedness or []
        for hand_landmarks, hand_info in zip(landmarks, handedness):
            if not hand_info.classification:
                continue

            classification = hand_info.classification[0]
            label = classification.label.lower()
            score = classification.score
            # MediaPipe Hands handedness is opposite to the Holistic labels used for training.
            if label == "left" and score > right_score:
                right_hand = hand_landmarks
                right_score = score
            elif label == "right" and score > left_score:
                left_hand = hand_landmarks
                left_score = score

        return left_hand, right_hand

    @staticmethod
    def _landmark_xy(landmark):
        return float(landmark.x), float(landmark.y)

    @staticmethod
    def _squared_distance_xy(point_a, point_b):
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        return dx * dx + dy * dy

    def _pose_wrist_points(self, pose_landmarks):
        if pose_landmarks is None:
            return None, None

        landmarks = pose_landmarks.landmark
        left = landmarks[POSE_LEFT_WRIST]
        right = landmarks[POSE_RIGHT_WRIST]
        left_point = (
            self._landmark_xy(left)
            if left.visibility >= POSE_WRIST_MIN_VISIBILITY
            else None
        )
        right_point = (
            self._landmark_xy(right)
            if right.visibility >= POSE_WRIST_MIN_VISIBILITY
            else None
        )
        return left_point, right_point

    def _remap_hands_by_pose_wrists(self, left_hand, right_hand, pose_landmarks):
        left_wrist, right_wrist = self._pose_wrist_points(pose_landmarks)
        if left_wrist is None and right_wrist is None:
            return left_hand, right_hand

        remapped = {"left": (None, float("inf")), "right": (None, float("inf"))}
        for hand_landmarks in (left_hand, right_hand):
            if hand_landmarks is None:
                continue

            hand_wrist = self._landmark_xy(hand_landmarks.landmark[HAND_WRIST])
            distances = []
            if left_wrist is not None:
                distances.append(
                    ("left", self._squared_distance_xy(hand_wrist, left_wrist))
                )
            if right_wrist is not None:
                distances.append(
                    ("right", self._squared_distance_xy(hand_wrist, right_wrist))
                )
            if not distances:
                continue

            side, distance_sq = min(distances, key=lambda item: item[1])
            if distance_sq > self.hand_pose_max_distance_sq:
                continue
            if distance_sq < remapped[side][1]:
                remapped[side] = (hand_landmarks, distance_sq)

        return remapped["left"][0], remapped["right"][0]

    def process(self, frame_bgr):
        """BGR 프레임 → 키포인트 dict (pose / left_hand / right_hand)."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_results = self.pose.process(rgb)
        hand_results = self.hands.process(rgb)

        left_hand_landmarks, right_hand_landmarks = self._split_hands(hand_results)
        left_hand_landmarks, right_hand_landmarks = self._remap_hands_by_pose_wrists(
            left_hand_landmarks,
            right_hand_landmarks,
            pose_results.pose_landmarks,
        )
        pose_detected = pose_results.pose_landmarks is not None
        left_detected = left_hand_landmarks is not None
        right_detected = right_hand_landmarks is not None

        pose = landmarks_to_array(pose_results.pose_landmarks, NUM_POSE_FULL)
        left = landmarks_to_array(left_hand_landmarks, NUM_HAND)
        right = landmarks_to_array(right_hand_landmarks, NUM_HAND)
        return {
            "pose": pose,
            "left_hand": left,
            "right_hand": right,
            "pose_detected": pose_detected,
            "left_hand_detected": left_detected,
            "right_hand_detected": right_detected,
        }


# ── 실시간 segmenter ───────────────────────────────────────────────────
class RealtimeSegmenter:
    """손 검출 비율 기반 시작/종료 판정 + 옵션 sliding window."""

    def __init__(
        self,
        fps: int = 30,
        window_sec: float = 0.5,
        start_ratio: float = 0.8,
        end_ratio: float = 0.8,
        min_frames: int = 8,
        max_record_sec: float = 10.0,
        sequence_level: bool = False,
        seq_window_frames: int = 35,
        seq_stride_frames: int = 35,
    ) -> None:
        self.fps = fps
        self.window_frames = max(1, int(window_sec * fps))
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.min_frames = min_frames
        self.max_record_frames = int(max_record_sec * fps)
        self.sequence_level = sequence_level
        self.seq_window_frames = seq_window_frames
        self.seq_stride_frames = seq_stride_frames

        self.detect_history: list[bool] = []  # 최근 window_frames 만큼만 유지
        self.segment_buffer: list[dict] = []
        self.recording = False
        self.last_emitted_seq_end = -1   # sequence-level: 마지막 추론한 window end index

    def reset(self) -> None:
        self.detect_history.clear()
        self.segment_buffer.clear()
        self.recording = False
        self.last_emitted_seq_end = -1

    def _detection_ratio(self) -> float:
        if not self.detect_history:
            return 0.0
        return sum(self.detect_history) / len(self.detect_history)

    def update(self, frame_data: dict):
        """프레임 1개 추가 → 종료된 segment 또는 sequence window 리스트 반환.

        반환: list of dict { "segment": [...], "type": "final"|"window" }
              (없으면 [])
        """
        hands_visible = bool(frame_data["left_hand_detected"] or frame_data["right_hand_detected"])
        self.detect_history.append(hands_visible)
        if len(self.detect_history) > self.window_frames:
            self.detect_history.pop(0)

        ratio = self._detection_ratio()
        produced: list[dict] = []

        if not self.recording:
            if hands_visible and ratio >= self.start_ratio:
                self.recording = True
                self.segment_buffer = [frame_data]
                self.last_emitted_seq_end = -1
            return produced

        # 녹화 중
        self.segment_buffer.append(frame_data)

        # 너무 길어지면 강제 종료
        if len(self.segment_buffer) >= self.max_record_frames:
            seg = self._finish()
            if seg is not None:
                produced.append({"segment": seg, "type": "final"})
            return produced

        # sequence-level: window 모이면 부분 추론
        if self.sequence_level and len(self.segment_buffer) >= self.seq_window_frames:
            end_idx = len(self.segment_buffer)
            start_idx = end_idx - self.seq_window_frames
            # stride 체크
            if (end_idx - self.last_emitted_seq_end) >= self.seq_stride_frames:
                produced.append({
                    "segment": self.segment_buffer[start_idx:end_idx],
                    "type": "window",
                })
                self.last_emitted_seq_end = end_idx

        # 종료 판정 (손이 사라짐)
        if not hands_visible and (1.0 - ratio) >= self.end_ratio:
            seg = self._finish()
            if seg is not None:
                produced.append({"segment": seg, "type": "final"})

        return produced

    def _finish(self):
        seg = self.segment_buffer
        self.recording = False
        self.segment_buffer = []
        if len(seg) < self.min_frames:
            return None
        # 마지막 detected frame까지만 잘라냄
        last_detected = -1
        for i in range(len(seg) - 1, -1, -1):
            if seg[i]["left_hand_detected"] or seg[i]["right_hand_detected"]:
                last_detected = i
                break
        if last_detected < self.min_frames - 1:
            return None
        return seg[: last_detected + 1]


# ── CNN1D 추론기 ───────────────────────────────────────────────────────
class CNN1DRealtimeRecognizer:
    """hyemin 브랜치 커스텀 CNN1D 모델 wrapper (mmaction2 대체)."""

    def __init__(
        self,
        config: Path,
        checkpoint: Path,
        label_map: Path,
        device: str = "cpu",
        topk: int = 5,
        max_gap: int = 10,
    ) -> None:
        import sys, torch
        # hyemin 모델 모듈 경로 추가
        _src = Path(__file__).resolve().parent / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from builder import build_model
        from config_utils import load_config
        from model import load_checkpoint

        self._torch = torch
        self.device = device
        self.topk = topk
        self.max_gap = max_gap
        self.label_map = self._load_label_map(label_map)

        cfg = load_config(str(config))
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model = self.model.to(device)
        load_checkpoint(self.model, str(checkpoint), map_location=device)
        self.model.eval()

    @staticmethod
    def _load_label_map(path: Path) -> dict:
        path = Path(path).expanduser()
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict_segment(self, segment: list[dict]) -> dict:
        """segment → top-k 예측 (hyemin preprocess 사용)."""
        import sys
        _src = Path(__file__).resolve().parent / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from data import preprocess_keypoint_sample

        cfg = self.cfg
        sample = build_mmaction_sample(segment, max_gap=self.max_gap)

        x = preprocess_keypoint_sample(
            sample,
            clip_len=getattr(cfg, "CLIP_LEN", 40),
            num_clips=getattr(cfg, "TEST_NUM_CLIPS", 1),
            test_mode=True,
            input_mode=getattr(cfg, "INPUT_MODE", "xyhandrel_bone"),
            keypoint_normalize=getattr(cfg, "KEYPOINT_NORMALIZE", None),
            random_horizontal_flip=getattr(cfg, "RANDOM_HORIZONTAL_FLIP", None),
            short_sample_interpolation=getattr(cfg, "SHORT_SAMPLE_INTERPOLATION", None),
            zero_pad_short=getattr(cfg, "ZERO_PAD_SHORT", False),
        )
        inputs = self._torch.from_numpy(x[None]).to(self.device)
        with self._torch.no_grad():
            scores = self.model.predict(inputs)[0]
        # with self._torch.no_grad():
        #     logits = self.model(inputs)
        # scores = self._torch.softmax(logits[0], dim=-1)
        return self._format(scores)

    def _format(self, scores) -> dict:
        topk = min(self.topk, scores.numel())
        vals, indices = scores.topk(topk)
        preds = []
        for cls, score in zip(indices.tolist(), vals.tolist()):
            preds.append({
                "class_id": int(cls),
                "label": self.label_map.get(str(int(cls)), str(int(cls))),
                "score": float(score),
            })
        return {"predictions": preds, "top1": preds[0] if preds else None}
