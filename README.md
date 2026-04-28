# TSN Action Recognition Inference (MMAction2)

This repository provides **inference code for a Temporal Segment Network (TSN)** trained using MMAction2.  
It is designed to be easily integrated into backend or application systems.

---

## ⚙️ Requirements

- Python ≥ 3.8
- CUDA ≥ 10.2 (recommended: 11.8)
- cuDNN ≥ 8
- PyTorch ≥ 1.8 (tested on 2.0.1)

---

## 🔧 Installation

### 1. Install PyTorch (CUDA 11.8 example)

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install MMAction2

Clone and install MMAction2:

```bash
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

---

### 4. Clone this repository

```bash
cd ../
git clone https://github.com/hyemin3656/FairyTaleSL.git
```
---

## 📦 Checkpoint

Download your trained model and place it here:

https://drive.google.com/file/d/1EG3CH4RL7f8yjyEi-wnRlO2zzgA6ioc7/view?usp=drive_link

---

## Input Format

The test code supports two input formats for inference:

### 1. Base64-encoded Frames (Recommended)
- Input: `List[str]` (e.g., `data:image/jpeg;base64,...`)
- Suitable for real-time applications (e.g., webcam streaming)
- Frames are decoded into RGB numpy arrays before inference
- result = predictor.predict_frames("examples/frames")

### 2. Image Directory
- Input: `str` (path to a directory containing images)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Useful for offline testing and debugging
- Frames are loaded from disk in sorted order

### 3. Pre-loaded Frames (NumPy Arrays)
- Input: `List[np.ndarray]`
- Each frame should have shape `(H, W, 3)` in RGB format
- Suitable for advanced usage where frames are already processed in memory

> ⚠️ Note  
> In `test_predictor.py`, update the argument of `predictor.predict_frames()` depending on the input type:
> - Directory path (`str`)
> - Base64 image list (`List[str]`)
> - NumPy frame list (`List[np.ndarray]`)

## 🚀 Usage

Run inference on a single video:

```bash
python test_predictor.py
```

## 📁 Project Structure

```bash
workspace/
├── mmaction2/
└── FairyTaleSL/
    ├── configs/
    ├── checkpoints/
    ├── examples/
    ├── src/
    └── test_predictor.py
```
