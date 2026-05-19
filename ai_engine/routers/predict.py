"""
POST /predict — ST-GCN 수어 인식 (classify 모드, 67 한국어 클래스)

Request:
  {
    "hands": [{"landmarks": [{"x","y","z"}, ...×21], "handedness": "Left"|"Right"}, ...],
    "frame_buffer": [{"hands": [...]}, ...]   // 백엔드가 누적한 다중 프레임
  }

Response:
  {
    "gloss": "안녕",
    "confidence": 0.83,
    "is_random_init": false
  }
"""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter
from pydantic import BaseModel

from models.stgcn import get_model, load_weights, NUM_NODES

router = APIRouter(tags=["predict"])

# ── 레이블 로드 (tsn/class_labels.json, 0~66 한국어) ─────────────────────────
_LABEL_PATH = Path(__file__).resolve().parents[1] / "tsn" / "class_labels.json"
with open(_LABEL_PATH, encoding="utf-8") as _f:
    _label_map: dict[str, str] = json.load(_f)
VOCAB: list[str] = [_label_map[str(i)] for i in range(len(_label_map))]  # 67개

WEIGHTS_PATH = os.environ.get("STGCN_WEIGHTS", "weights/stgcn_best.pt")
IS_RANDOM_INIT = not os.path.exists(WEIGHTS_PATH)

T_FIXED = 105   # 학습과 동일
_model = get_model(vocab=VOCAB, device="cpu", mode="classify")
load_weights(WEIGHTS_PATH)


# ── Pydantic 스키마 ───────────────────────────────────────────────────────────
class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class HandFrame(BaseModel):
    landmarks: list[LandmarkPoint]
    handedness: str = "Unknown"


class FrameEntry(BaseModel):
    hands: list[HandFrame]


class PredictRequest(BaseModel):
    hands: list[HandFrame]
    frame_buffer: Optional[list[FrameEntry]] = None


# ── 정규화 (dataset.py의 _normalize_hands와 동일) ────────────────────────────
def _normalize_hands(arr: np.ndarray) -> np.ndarray:
    """arr: (T, 2, 21, 3) → 손목 기준 평행이동 + 손 크기 스케일"""
    out = arr.copy()
    wrist = out[:, :, 0:1, :]                                   # (T,2,1,3)
    out = out - wrist
    span = np.linalg.norm(out[:, :, 9, :], axis=-1, keepdims=True)  # (T,2,1)
    span = np.where(span < 1e-6, 1.0, span)[..., None]          # (T,2,1,1)
    out = out / span
    zero_mask = (arr.sum(axis=-1, keepdims=True) == 0)
    out = np.where(zero_mask, 0.0, out)
    return out.astype(np.float32)


def _fit_length(arr: np.ndarray, t_fixed: int) -> np.ndarray:
    """T 축을 t_fixed로 맞춤: center-crop 또는 zero-pad"""
    T = arr.shape[0]
    if T == t_fixed:
        return arr
    if T > t_fixed:
        start = (T - t_fixed) // 2
        return arr[start:start + t_fixed]
    pad = np.zeros((t_fixed - T,) + arr.shape[1:], dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _build_tensor(frame_buffer: list[FrameEntry]) -> torch.Tensor:
    """
    frame_buffer → (1, 3, T_fixed, 21, 2) 텐서
    dataset.py의 KSLKeypointDataset.__getitem__과 동일한 전처리.
    """
    T = len(frame_buffer)
    V, M = NUM_NODES, 2
    hands_arr = np.zeros((T, M, V, 3), dtype=np.float32)

    for t, frame in enumerate(frame_buffer):
        for hand in frame.hands[:M]:
            # "Left" → m=0, 그 외("Right"/Unknown) → m=1
            m = 0 if hand.handedness.lower() == "left" else 1
            for v, lm in enumerate(hand.landmarks[:V]):
                hands_arr[t, m, v, 0] = lm.x
                hands_arr[t, m, v, 1] = lm.y
                hands_arr[t, m, v, 2] = lm.z

    hands_arr = _normalize_hands(hands_arr)
    hands_arr = _fit_length(hands_arr, T_FIXED)  # (T_fixed, 2, 21, 3)

    # (3, T_fixed, 21, 2) → (1, 3, T_fixed, 21, 2)
    x = np.transpose(hands_arr, (3, 0, 2, 1))
    return torch.from_numpy(x).unsqueeze(0)


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@router.post("/predict")
async def predict(req: PredictRequest):
    buffer = req.frame_buffer if (req.frame_buffer and len(req.frame_buffer) >= 5) \
        else [FrameEntry(hands=req.hands)] * 30

    x = _build_tensor(buffer)  # (1, 3, 105, 21, 2)

    _model.eval()
    with torch.no_grad():
        logits = _model(x)          # (1, 67)
        probs = torch.softmax(logits, dim=-1)[0]  # (67,)

    top_idx = int(probs.argmax())
    confidence = float(probs[top_idx])

    return {
        "gloss": VOCAB[top_idx],
        "confidence": round(confidence, 4),
        "is_random_init": IS_RANDOM_INIT,
    }
