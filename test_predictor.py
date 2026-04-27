from src.predictor import TSNPredictor

predictor = TSNPredictor(
    config_path="configs/tsn_kinetics_pretrained_ksl_finetuned.py",
    checkpoint_path="checkpoints/model.pth",
    label_map_path="src/class_labels.json",
    device="cuda:0"
)

result = predictor.predict("examples/04_56.mp4")
print(result)