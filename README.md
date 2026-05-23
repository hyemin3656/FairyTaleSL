# MMAction2 Action Recognition Training & Inference

This repository provides **training and inference code for action recognition models using MMAction2**.  
It currently supports model training based on MMAction2 configurations and includes test/inference code for both **TSN** and **ST-GCN** models.

The project is designed to be easily extended and integrated into backend or application systems.

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
git clone -b experiment/fairytalesl https://github.com/hyemin3656/mmaction2.git #my forked repository
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

[https://drive.google.com/drive/folders/1z9X_AbwOjSLq7-6pROKsCJxGwEZjj0k_?usp=sharing
](https://drive.google.com/drive/folders/1h0-ojkDY26MUHfenemiv3XG6t_NkgIEZ?usp=drive_link)

stgcn_bilstm_ctc/sequence_level_baseline/best_wer_epoch_85.pth is the sequence level prediction model.

---

## Input Format

### key-point ndarry

you can put any ndarry sample(ex.'00_004) into 'examples/sequenced_key_points'.

[https://drive.google.com/drive/folders/1UvOW9TJA62yQBDEF2LGvyLRnajNHOGnm?usp=sharing
](https://drive.google.com/file/d/1OiuxHetw91XUBeFvp3DhwN7Y4ChOr2vn/view?usp=drive_link)

## 🚀 Usage

Run st-gcn inference on a squence-level video:

```bash
python test_stgcn_ctc.py --checkpoint [pth path] --keypoint-dir /home/ubuntu/FairyTaleSL/examples/sequenced_key_points/00_004
```
Run st-gcn inference on a single video:

```bash
python test_stgcn.py
```

Run tsn inference on a single video:

```bash
python test_tsn.py
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
