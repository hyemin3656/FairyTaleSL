import os
import json
import torch
import numpy as np
import base64
import cv2

from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.utils import register_all_modules

from tsn.data.loading import load_frames_from_dir, load_frames_from_base64
from tsn.data.pipeline import build_frame_pipeline, build_frame_data

# PyTorch 2.6+ weights_only=True 기본값 회피 — 신뢰 가능한 자체 학습 체크포인트만 로드
try:
    from mmengine.logging.history_buffer import HistoryBuffer
    torch.serialization.add_safe_globals([HistoryBuffer])
except Exception:
    pass
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

class TSNPredictor:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        label_map_path,
        device="cuda:0"
    ):
        register_all_modules(init_default_scope=True)

        self.device = device
        self.model = init_recognizer(
            config_path,
            checkpoint_path,
            device=device
        )
        self.frame_pipeline = build_frame_pipeline()

        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

    def _format_predictions(self, pred_score, topk=1):
        # 학습 config 버그(num_classes=400)로 인해 출력은 400차원이지만
        # 실제 KSL 라벨은 0~(N-1)만 유효. label_map 크기로 슬라이싱.
        n_valid = len(self.label_map)
        valid_scores = pred_score[:n_valid]
        # softmax 재정규화 (잘려나간 logit 영향 보정)
        valid_scores = torch.softmax(torch.log(valid_scores.clamp(min=1e-12)), dim=0)

        predictions = []
        for pred in valid_scores.argsort(descending=True)[:topk]:
            class_id = int(pred)
            score = float(valid_scores[class_id])
            predictions.append({
                "class_id": class_id,
                "label": self.label_map.get(str(class_id), str(class_id)),
                "score": score
            })
        return predictions

    #prediction for video
    def predict(self, video_path, topk=1):
        result = inference_recognizer(self.model, video_path)

        return {
            "video_path": video_path,
            "predictions": self._format_predictions(result.pred_score, topk)
        }

    #prediction for selected frames
    def predict_frames(self, frames_input, topk=1):
        if isinstance(frames_input, str):
            frames = load_frames_from_dir(frames_input)

        elif isinstance(frames_input, list):
            if len(frames_input) == 0:
                raise ValueError("frames_input is empty")

            first = frames_input[0]

            if isinstance(first, str):
                frames = load_frames_from_base64(frames_input)

            if isinstance(first, np.ndarray):
                frames = frames_input

        else:
            raise TypeError(
            "frames_input must be a directory path, list of base64 strings, "
            "or list of numpy arrays"
            )

        data = build_frame_data(frames)
        data = self.frame_pipeline(data)

        with torch.no_grad():
            result = self.model.test_step({
                "inputs": [data["inputs"].to(self.device)],
                "data_samples": [data["data_samples"]]
            })[0]

        return {
            "predictions": self._format_predictions(result.pred_score, topk)
        }