"""
POST /predict — ST-GCN + CTC 수어 인식

Request:
  {
    "hands": [
      {
        "landmarks": [{"x":0.1,"y":0.2,"z":0.0}, ...×21],
        "handedness": "Right"
      }, ...
    ],
    "frame_buffer": [  // 선택: 다중 프레임 시퀀스
      { "hands": [...] }
    ]
  }

Response:
  {
    "gloss": "토끼",
    "confidence": 0.83,
    "sequence": ["토끼"],  // CTC decoded 전체 시퀀스
    "is_random_init": true  // 학습 가중치 없을 경우 true
  }
"""
import os
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.stgcn import get_model, load_weights, NUM_NODES
from models.ctc import beam_decode

router = APIRouter(tags=["predict"])

# 글로스 어휘 (Motion DB 기준 — 실제 운영 시 DB에서 로드)
VOCAB = [
    "토끼", "거북이", "경주", "달나라", "별", "도깨비",
    "할아버지", "소녀", "우주", "출발", "낮잠", "결승",
    "혹", "노래", "블랙홀", "용기",
]
BLANK_IDX = 0
WEIGHTS_PATH = os.environ.get("STGCN_WEIGHTS", "weights/stgcn.pt")
IS_RANDOM_INIT = not os.path.exists(WEIGHTS_PATH)

# 모델 초기화
_model = get_model(vocab=VOCAB, device="cpu")
load_weights(WEIGHTS_PATH)


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class HandFrame(BaseModel):
    landmarks: list[LandmarkPoint]  # 21개
    handedness: str = "Unknown"


class FrameEntry(BaseModel):
    hands: list[HandFrame]


class PredictRequest(BaseModel):
    hands: list[HandFrame]               # 현재 프레임 손 데이터
    frame_buffer: Optional[list[FrameEntry]] = None  # 다중 프레임 버퍼


def _build_tensor(frame_buffer: list[FrameEntry]) -> torch.Tensor:
    """
    frame_buffer → (1, C, T, V, M) 텐서 변환
    C=3, T=len(buffer), V=21, M=2
    """
    T = len(frame_buffer)
    C, V, M = 3, NUM_NODES, 2
    arr = np.zeros((1, C, T, V, M), dtype=np.float32)

    for t, frame in enumerate(frame_buffer):
        for m, hand in enumerate(frame.hands[:M]):
            for v, lm in enumerate(hand.landmarks[:V]):
                arr[0, 0, t, v, m] = lm.x
                arr[0, 1, t, v, m] = lm.y
                arr[0, 2, t, v, m] = lm.z

    return torch.from_numpy(arr)


@router.post("/predict")
async def predict(req: PredictRequest):
    # frame_buffer 없으면 현재 프레임을 30회 반복해 시퀀스 구성
    if req.frame_buffer and len(req.frame_buffer) >= 5:
        buffer = req.frame_buffer
    else:
        dummy_frame = FrameEntry(hands=req.hands)
        buffer = [dummy_frame] * 30

    x = _build_tensor(buffer)  # (1, 3, T, 21, 2)

    with torch.no_grad():
        log_probs = _model(x)  # (1, T', num_classes)

    log_probs_seq = log_probs[0]  # (T', num_classes)
    result = beam_decode(log_probs_seq, vocab=VOCAB, blank_idx=BLANK_IDX, beam_width=5)

    # 학습 가중치 없으면 confidence 보정 (무작위 출력임을 나타냄)
    confidence = result.confidence if not IS_RANDOM_INIT else round(result.confidence * 0.6, 3)

    return {
        "gloss": result.top_gloss or VOCAB[0],
        "confidence": confidence,
        "sequence": result.sequence,
        "is_random_init": IS_RANDOM_INIT,
    }
