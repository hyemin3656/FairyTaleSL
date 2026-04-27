from mmaction.apis import init_recognizer, inference_recognizer
import json


class TSNPredictor:
    def __init__(self, config_path, checkpoint_path, label_map_path, device="cuda:0"):
        self.model = init_recognizer(config_path, checkpoint_path, device=device)

        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

    def predict(self, video_path, topk=1):
        result = inference_recognizer(self.model, video_path)

        predictions = []
        for pred in result.pred_score.argsort(descending=True)[:topk]:
            class_id = int(pred)
            score = float(result.pred_score[class_id])

            predictions.append({
                "class_id": class_id,
                "label": self.label_map.get(str(class_id), str(class_id)),
                "score": score
            })

        return {
            "video_path": video_path,
            "predictions": predictions
        }