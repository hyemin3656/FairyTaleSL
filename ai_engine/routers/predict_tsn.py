"""
POST /predict_frames — 수어 인식 (SignRecognizer 추상화 경유)

Request:
  { "frames": ["data:image/jpeg;base64,...", ...], "topk": int }

Response:
  { "predictions": [{ "class_id": int, "label": str, "score": float }, ...] }

내부적으로 TSNRecognizer를 사용한다. 향후 다른 백본(ST-GCN 등)으로 교체 시
이 파일에서 구현체만 바꾸면 된다.
"""
import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_AI_ROOT = Path(__file__).resolve().parents[1]
if str(_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_ROOT))

from recognizer import SignRecognizer  # noqa: E402
from tsn.data.loading import load_frames_from_base64  # noqa: E402

router = APIRouter(tags=["predict_tsn"])

_TSN_DIR        = _AI_ROOT / "tsn"
_CONFIG_PATH    = os.environ.get("TSN_CONFIG", str(_TSN_DIR / "configs" / "tsn_kinetics_pretrained_ksl_finetuned.py"))
_CHECKPOINT     = os.environ.get("TSN_CHECKPOINT", str(_TSN_DIR / "checkpoints" / "model.pth"))
_LABEL_MAP_PATH = os.environ.get("TSN_LABELS", str(_TSN_DIR / "class_labels.json"))
_DEVICE         = os.environ.get("TSN_DEVICE", "cpu")

_recognizer: SignRecognizer | None = None


def get_recognizer() -> SignRecognizer:
    """지연 초기화. lifespan에서 명시 호출하여 warmup 수행."""
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    if not os.path.exists(_CHECKPOINT):
        raise RuntimeError(
            f"TSN checkpoint not found: {_CHECKPOINT}\n"
            "Google Drive에서 model.pth 다운로드 후 ai_engine/tsn/checkpoints/ 에 두세요."
        )
    from recognizer import TSNRecognizer
    _recognizer = TSNRecognizer(
        config_path=_CONFIG_PATH,
        checkpoint_path=_CHECKPOINT,
        label_map_path=_LABEL_MAP_PATH,
        device=_DEVICE,
    )
    print(f"[AI Engine] Recognizer ready (TSN, device={_DEVICE})")
    return _recognizer


# 하위 호환 alias — main.py lifespan이 get_predictor()를 호출하던 흔적 보존
def get_predictor() -> SignRecognizer:
    return get_recognizer()


class PredictFramesReq(BaseModel):
    frames: list[str]
    topk: int = 3


@router.post("/predict_frames")
async def predict_frames(req: PredictFramesReq):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames is empty")
    try:
        recognizer = get_recognizer()
        frames = load_frames_from_base64(req.frames)
        preds = recognizer.predict_clip(frames, topk=req.topk)
        return {"predictions": [p.to_dict() for p in preds]}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e}")
