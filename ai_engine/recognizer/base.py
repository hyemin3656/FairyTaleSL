"""
SignRecognizer — 수어 인식 모델 추상 인터페이스.

목적: TSN 외 다른 백본(ST-GCN, MMPose+CTC, MediaPipe 등)으로 교체 가능하게
모델 호출부를 분리. 시나리오 핸들러(따라하기/질문)는 이 인터페이스만 의존한다.

구현체:
  - TSNRecognizer (ai_engine.recognizer.tsn_recognizer)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    class_id: int
    label: str
    score: float

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "label": self.label, "score": self.score}


class SignRecognizer(ABC):
    """
    수어 인식기 공통 인터페이스.

    상태 모드:
      - 일괄(stateless): predict_clip(frames) 직접 호출 — 프론트엔드가 sliding
        window를 외부 관리하는 현재 흐름.
      - 스트리밍(stateful): process_frame(frame) 반복 호출 — 서버 측 버퍼링.
        WebSocket 도입 시 사용 가능.
    """

    @abstractmethod
    def predict_clip(
        self, frames: list[np.ndarray], topk: int = 5
    ) -> list[Prediction]:
        """일괄 예측. frames: H×W×3 uint8 BGR/RGB list."""

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, topk: int = 5
    ) -> list[Prediction] | None:
        """단일 프레임 입력. 윈도우가 차면 Prediction 리스트, 아니면 None."""

    @abstractmethod
    def reset(self) -> None:
        """내부 버퍼/타이머 초기화."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """유효 라벨 개수 (예: KSL 67)."""
