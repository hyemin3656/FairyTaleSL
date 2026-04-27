# ST-GCN 베이스라인 학습 가이드

KSL 키포인트 데이터셋을 ST-GCN으로 67-클래스 고립단어 분류하는 베이스라인.

## 1. 입출력 정합성 (검증 완료)

| 자산 | 키 / 값 |
|---|---|
| `ai_engine/data/keypoints/` | 67개 클래스 폴더 (000~066) |
| `label_map.txt` | 67줄, `label_idx ↔ gloss_id`(예: 0↔'01', 30↔'36') |
| `split_result.csv` | 1,229행 — `class_id` 는 **원본 gloss(1~77)** |
| `class_label.p` | `{0: ' ', 1:'hi', ...}` — **`cl[label_idx + 1]`** 가 영어 단어 |

크로스체크 결과 split CSV의 모든 1,229행이 실제 `.npy` 파일과 매칭됨 (missing 0건).

> **주의**: split CSV는 sample-level 분할(모든 subject가 train/val/test에 동시 등장).
> subject-independent 평가가 필요하면 별도 split 재생성 필요.

## 2. DataLoader (`training/dataset.py`)

- 입력: `data/keypoints/{label_idx:03d}/{subject_id:02d}.npy` `(T, 225)`
- 손만 사용: `[:126]` → `(T, 2, 21, 3)` reshape
- 정규화: 손목(노드 0) 기준 평행이동 + (손목→중지 시작점) 거리로 스케일
- 길이 통일: `T_fixed=105` (75th percentile) — center-crop 또는 zero-pad
- **양손 0인 시퀀스 6개 자동 제외** (train 5, test 1)
- 출력 텐서: **`(3, 105, 21, 2)`** = `(C, T, V, M)` ← ST-GCN 요구 형식

확정 데이터셋 크기:
- train **973** / val **133** / test **117** = 1,223 (총 1,229 - 6)

## 3. 모델 (`models/stgcn.py`)

`STGCN(num_classes=67, mode='classify')`

- 손 그래프 21노드 × 2손 = 42노드 인접행렬
- 6개 ST-GCN 블록, 채널 3 → 64 → 64 → 128 → 128 → 256 → 256
- 시간축 stride=2 두 번 → T=105 → T'=27
- **`mode='classify'`**: 노드 평균 → 시간 평균 → Linear → `(B, 67)` logits
- **`mode='ctc'`**: 기존 동작 유지(`(B, T', num_classes)` log-softmax)
- 파라미터 **1.74M** — 매우 가볍

## 4. 학습 스크립트 (`training/train_stgcn.py`)

기본값:
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Scheduler: StepLR(step=20, gamma=0.5)
- Loss: CrossEntropy(label_smoothing=0.1)
- Epochs: 80, batch=32
- best val_acc 기준 체크포인트 자동 저장

산출물 (`out_dir/`):
- `stgcn_best.pt` — best val 가중치
- `config.json`, `history.json`, `test_result.json`
- `vocab.json` — `label_idx → 영어 단어` 리스트 (추론용)

## 5. Colab T4 실행 가이드

### 셀 1 — 마운트 & 의존성
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/PSYcho   # 본인 경로
!pip install -q torch numpy
```

### 셀 2 — 데이터 위치 확인
필요한 파일이 모두 동일 폴더 안에 있어야 합니다:
- `ai_engine/data/keypoints/` (66개 폴더 + label_map.txt) — **로컬에서 zip해서 업로드**
- `split_result.csv`, `class_label.p`

### 셀 3 — 학습 실행
```bash
!python ai_engine/training/train_stgcn.py \
    --keypoints_dir ai_engine/data/keypoints \
    --split_csv     split_result.csv \
    --label_map     ai_engine/data/keypoints/label_map.txt \
    --class_pickle  class_label.p \
    --out_dir       weights/stgcn_baseline \
    --epochs 80 --batch_size 64 --num_workers 2 --device cuda
```

### 예상 자원 사용량
- VRAM: **400~600 MB** (T4 16GB 중 4% 미만)
- 1 epoch (batch=64): **~5초** (T4) / ~55초 (CPU 로컬 측정값)
- 80 epoch 총 **~7분** (T4 기준)
- 디스크: 키포인트 데이터 ~수십 MB + 가중치 7 MB

## 6. 베이스라인 → 개선 로드맵

| 단계 | 내용 | 예상 효과 |
|---|---|---|
| **0. 베이스라인** | 위 그대로 | 67-클래스, 랜덤=1.5%. 첫 목표 **val_acc ≥ 0.60** |
| 1. 데이터 증강 | 시간축 random crop/스케일링, 좌표 가우시안 노이즈, 좌우 손 swap | +5~15%p |
| 2. 포즈 추가 | 손 21×2 + 어깨/팔꿈치 4노드 그래프로 확장, 인접행렬·`HAND_EDGES` 갱신 | 위치 기반 단어 개선 |
| 3. CTC 복원 | `mode='ctc'`로 multi-gloss 시퀀스 학습 (라벨 시퀀스 데이터 필요) | 연속수어 인식 |
| 4. 시간 모델링 강화 | TCN 블록 깊이↑ 또는 Transformer head | +3~5%p |
| 5. Subject-independent split | 새 split CSV 생성 후 재학습 → 일반화 평가 | 실전 일반화 측정 |

## 7. 알려진 한계

- **MediaPipe 좌표 z 정규화 없음** — 손/포즈 z 원점이 달라 z 가중치 자동학습에 의존
- **시퀀스마다 landmarker 재생성** (추출 단계) — 추가 데이터 확보 시 시간 비용
- 데이터셋이 **sample-level split** 이라 metric이 다소 낙관적(같은 subject가 train/val 양쪽 등장)
