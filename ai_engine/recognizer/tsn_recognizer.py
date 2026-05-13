"""
TSNRecognizer — MMAction2 TSN 백본 어댑터.

기존 `tsn.predictor.TSNPredictor`를 위임만 한다. TSN 코드(predictor.py / data
pipeline)는 절대 수정하지 않는다.

스트리밍 모드(`process_frame`)는 시간 윈도우 + 균등 샘플링을 자체 보유한 deque로
관리. 프론트엔드가 이미 윈도우를 관리하는 현재 흐름에서는 `predict_clip`만 호출된다.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from tsn.predictor import TSNPredictor

from .base import Prediction, SignRecognizer


class TSNRecognizer(SignRecognizer):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        label_map_path: str,
        device: str = "cpu",
        *,
        num_clips: int = 8,
        window_ms: int = 3000,
    ):
        self._predictor = TSNPredictor(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            label_map_path=label_map_path,
            device=device,
        )
        self._num_clips = num_clips
        self._window_ms = window_ms
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._last_predict_ms: float = 0.0

    # ── 공개 API ──────────────────────────────────────────────────────────
    @property
    def num_classes(self) -> int:
        return len(self._predictor.label_map)

    def predict_clip(
        self, frames: list[np.ndarray], topk: int = 5
    ) -> list[Prediction]:
        if not frames:
            return []
        result = self._predictor.predict_frames(frames, topk=topk)
        return [Prediction(**p) for p in result["predictions"]]

    def process_frame(
        self, frame: np.ndarray, topk: int = 5, *, predict_every_ms: int = 1000
    ) -> list[Prediction] | None:
        now_ms = time.monotonic() * 1000
        self._buffer.append((now_ms, frame))
        # 시간 윈도우 밖 프레임 제거
        cutoff = now_ms - self._window_ms
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        if len(self._buffer) < self._num_clips:
            return None
        if now_ms - self._last_predict_ms < predict_every_ms:
            return None

        sampled = self._uniform_sample([f for _, f in self._buffer], self._num_clips)
        self._last_predict_ms = now_ms
        return self.predict_clip(sampled, topk=topk)

    def reset(self) -> None:
        self._buffer.clear()
        self._last_predict_ms = 0.0

    # ── 내부 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _uniform_sample(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
        total = len(frames)
        if total <= n:
            return frames
        return [frames[(i * total) // n] for i in range(n)]
