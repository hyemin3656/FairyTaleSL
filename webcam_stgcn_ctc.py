import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose

from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules

from tool.webcam_keypoints import (
    FPS,
    ROI_CROP_SIZE,
    USE_ROI,
    MediaPipeKeypointExtractor,
    OnlineHandCropper,
    draw_overlay,
    require_mediapipe,
    save_segment,
)

from test_stgcn_ctc import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_LABEL_MAP,
    load_label_map,
    predict_from_each_keypoint,
    predict_from_total_keypoint_npy,
    resolve_device,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "realtime_outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture webcam keypoints with the same MediaPipe rules used for "
            "training, crop by hand-detection rule, save npy files, and run "
            "ST-GCN-BiLSTM-CTC inference."))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on. Use 'auto' to prefer CUDA when available.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--start-ratio", type=float, default=0.8)
    parser.add_argument("--end-ratio", type=float, default=0.8)
    parser.add_argument("--use-roi", action="store_true", default=USE_ROI)
    parser.add_argument("--roi-crop-size", type=int, default=ROI_CROP_SIZE)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-cropped-frames", type=int, default=1)

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.cls_head.dropout=0.0.")
    return parser.parse_args()




def load_model(config_path, checkpoint_path, device, cfg_options=None):
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    model = init_recognizer(
        cfg,
        checkpoint=str(Path(checkpoint_path).expanduser().resolve()),
        device=device,
    )
    return model, Compose(cfg.test_pipeline)


def main():
    args = parse_args()
    require_mediapipe()
    device = resolve_device(args.device)
    window = args.window if args.window is not None else int(args.fps * 0.5)
    sample_base = args.sample_name or datetime.now().strftime("webcam_%Y%m%d_%H%M%S")

    model, pipeline = load_model(
        args.config,
        args.checkpoint,
        device,
        cfg_options=args.cfg_options,
    )
    label_map = load_label_map(args.label_map)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    extractor = MediaPipeKeypointExtractor(
        use_roi=args.use_roi,
        roi_crop_size=args.roi_crop_size,
    )
    cropper = OnlineHandCropper(
        window=window,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
    )

    print("Press q to quit. Waiting for hand-detection start window...")
    print(
        f"Crop rule: window={window}, start_ratio={args.start_ratio}, "
        f"end_ratio={args.end_ratio}, use_roi={args.use_roi}")

    frame_idx = 0
    prediction_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            extracted, pose_result, hands_result, left_roi_box, right_roi_box = (
                extractor.process_frame(frame, frame_idx)
            )
            segment = cropper.update(extracted)

            status = "recording" if cropper.started else "waiting"
            if args.display:
                overlay = draw_overlay(
                    frame,
                    pose_result,
                    hands_result,
                    extracted,
                    left_roi_box=left_roi_box,
                    right_roi_box=right_roi_box,
                    status=status,
                )
                cv2.imshow("FairyTaleSL realtime ST-GCN-CTC", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if segment is not None:
                if len(segment) < args.min_cropped_frames:
                    print(f"Skipped short segment: {len(segment)} frames")
                    cropper.reset()
                    continue

                sample_name = sample_base
                if args.continuous:
                    sample_name = f"{sample_base}_{prediction_count:03d}"

                save_start = time.perf_counter()
                sample_dir, video_path, pose_np, left_np, right_np = save_segment(
                    segment,
                    args.output_dir,
                    sample_name,
                    args.fps,
                    save_video=True,
                    save_frames=True,
                )
                save_ms = (time.perf_counter() - save_start) * 1000

                # pred = predict_from_each_keypoint(
                #     model=model,
                #     pipeline=pipeline,
                #     keypoint_dir=sample_dir,
                #     label_map=label_map,
                #     device=device,
                # )
                pred = predict_from_each_keypoint(
                    model=model,
                    pipeline=pipeline,
                    arrs=[pose_np, left_np, right_np],
                    label_map=label_map,
                    device=device,
                )
                last_detection_to_prediction_ms = (
                    (time.perf_counter() - cropper.last_detected_time) * 1000
                    if cropper.last_detected_time is not None else 0.0)
                mediapipe_ms_per_frame = (
                    sum(item.elapsed_ms for item in segment) / len(segment)
                    if segment else 0.0)

                print("\nSegment saved:", sample_dir)
                print("Frame jpg dir:", sample_dir / "frames")
                print("Segment mp4:", video_path)
                print("Device:", device)
                print("Frames:", len(segment))
                print("Input shape:", pred["input_shape"])
                print("MediaPipe per frame: {:.3f} ms".format(mediapipe_ms_per_frame))
                print("Save npy time: {:.3f} ms".format(save_ms))
                print("preprocessing time: {:.3f} ms".format(pred["keypoint_build_latency"] * 1000))
                print("pipeline time: {:.3f} ms".format(pred["pipeline_latency"] * 1000))
                print("model latency time: {:.3f} ms".format(pred["model_latency"] * 1000))
                print("last detection to prediction time: {:.3f} ms".format(last_detection_to_prediction_ms))
                print("Pred gloss ids:", pred["gloss_ids"])
                print("Pred gloss labels:", pred["gloss_labels"])

                prediction_count += 1
                if not args.continuous:
                    break
                cropper.reset()

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        extractor.close()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
