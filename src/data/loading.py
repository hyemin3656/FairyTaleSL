import os
import numpy as np
import base64
import cv2
    
    
def load_frames_from_dir(frames_dir):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")

    frame_paths = [
        os.path.join(frames_dir, fname)
        for fname in sorted(os.listdir(frames_dir))
        if fname.lower().endswith(image_exts)
    ]

    if len(frame_paths) == 0:
        raise ValueError(f"No frame images found in {frames_dir}")

    frames = []

    for path in frame_paths:
        frame = cv2.imread(path)

        if frame is None:
            raise ValueError(f"Failed to read frame: {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return frames

def load_frames_from_base64(frames_base64):
    frames = []

    for data in frames_base64:
        if "," in data:
            _, data = data.split(",", 1)

        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode base64 image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return frames