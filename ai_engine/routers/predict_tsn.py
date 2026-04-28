"""
POST /predict_frames — TSN(MMAction2) 기반 수어 인식

Request:
  { "frames": ["data:image/jpeg;base64,...", ...] }   # 보통 25프레임

Response:
  { "predictions": [{ "class_id": int, "label": str, "score": float }, ...] }
"""
import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ai_engine 루트를 import path에 추가 (tsn 패키지 탐색용)
_AI_ROOT = Path(__file__).resolve().parents[1]
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

router = APIRouter(tags=["predict_tsn"])

_TSN_DIR        = _AI_ROOT / "tsn"
_CONFIG_PATH    = os.environ.get("TSN_CONFIG", str(_TSN_DIR / "configs" / "tsn_kinetics_pretrained_ksl_finetuned.py"))
_CHECKPOINT     = os.environ.get("TSN_CHECKPOINT", str(_TSN_DIR / "checkpoints" / "model.pth"))
_LABEL_MAP_PATH = os.environ.get("TSN_LABELS", str(_TSN_DIR / "class_labels.json"))
_DEVICE         = os.environ.get("TSN_DEVICE", "cpu")

_predictor = None


def get_predictor():
    """지연 초기화: 첫 호출 또는 lifespan에서 명시 호출."""
    global _predictor
    if _predictor is not None:
        return _predictor
    if not os.path.exists(_CHECKPOINT):
        raise RuntimeError(
            f"TSN checkpoint not found: {_CHECKPOINT}\n"
            "Google Drive에서 model.pth 다운로드 후 ai_engine/tsn/checkpoints/ 에 두세요."
        )
    from tsn.predictor import TSNPredictor
    _predictor = TSNPredictor(
        config_path=_CONFIG_PATH,
        checkpoint_path=_CHECKPOINT,
        label_map_path=_LABEL_MAP_PATH,
        device=_DEVICE,
    )
    print(f"[AI Engine] TSN ready. device={_DEVICE}")
    return _predictor


class PredictFramesReq(BaseModel):
    frames: list[str]
    topk: int = 3


@router.post("/predict_frames")
async def predict_frames(req: PredictFramesReq):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames is empty")
    try:
        predictor = get_predictor()
        return predictor.predict_frames(req.frames, topk=req.topk)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e}")
