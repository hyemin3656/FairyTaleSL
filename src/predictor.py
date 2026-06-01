import os
import json
import torch
import numpy as np
import base64
import cv2

from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.utils import register_all_modules

from src.data.loading import load_frames_from_dir, load_frames_from_base64
from src.data.pipeline import build_frame_pipeline, build_frame_data

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
        predictions = []

        for pred in pred_score.argsort(descending=True)[:topk]:
            class_id = int(pred)
            score = float(pred_score[class_id])

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