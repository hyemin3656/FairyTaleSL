# FairyTaleSL

MMAction2 기반 수어 동작 인식 프로젝트입니다. 현재 실시간 웹캠 추론 스크립트는 MediaPipe Holistic으로 pose, left hand, right hand keypoint를 추출한 뒤 CNN1D 모델로 gloss를 예측합니다.

## Requirements

- Python 3.8 이상
- PyTorch
- OpenCV
- MediaPipe
- MMAction2
- mmengine
- pandas
- numpy

CUDA GPU가 있으면 `--device auto` 또는 `--device cuda:0`로 GPU 추론을 사용할 수 있습니다.

## Installation

### 1. PyTorch 설치 예시

CUDA 11.8 환경 예시입니다.

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Python dependency 설치

```bash
pip install -r requirements.txt
```

### 3. MMAction2 설치

workspace 루트에 `mmaction2`가 있어야 합니다.

```bash
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

## Checkpoint

학습된 모델 checkpoint를 다운로드한 뒤 `checkpoints/` 아래에 둡니다.

기본 checkpoint 경로:

```text
checkpoints/best_acc_top1_epoch_36.pth
```


## 실시간 웹캠 추론

실시간 웹캠 추론 스크립트:

```text
FairyTaleSL/webcam_cnn1d_realtime.py
```

기본 실행:

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py
```

웹캠 화면을 보면서 실행:

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py --show
```

GPU 지정:

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py --device cuda:0
```

CPU 강제 실행:

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py --device cpu
```

## 웹캠 표시

`--show` 옵션을 켜면 웹캠 화면에 keypoint가 표시됩니다.

- pose: 파랑
- left hand: 초록
- right hand: 빨강

화면에는 현재 상태, FPS, 최근 top-k 예측 결과도 함께 표시됩니다.

## Gloss Level Detection

기본 모드에서는 손이 감지되기 시작하면 recording 상태로 들어가고, 손이 일정 시간 감지되지 않으면 segment를 종료합니다. 종료된 segment는 마지막 detected frame까지만 잘라 모델에 한 번 입력됩니다.

예시:

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py --show
```

주요 옵션:

```text
--window-sec       시작/종료 판단에 사용할 시간 window. 기본값 0.5
--start-ratio      window 안에서 손 감지가 시작으로 판단될 비율. 기본값 0.8
--end-ratio        window 안에서 손 미감지가 종료로 판단될 비율. 기본값 0.8
--min-frames       너무 짧은 segment를 무시할 최소 프레임 수. 기본값 8
--max-record-sec   한 segment의 최대 녹화 시간. 기본값 10.0
```

## Keypoint 전처리

모델에 입력되기 전 MediaPipe로 추출한 pose, left hand, right hand keypoint는 각각 같은 방식으로 전처리됩니다.

짧은 미검출 구간은 `--max-gap` 값 이하일 때 선형보간으로 채웁니다. 보간된 프레임의 keypoint score는 0.5로 설정되고, 보간이 불가능한 구간은 0으로 유지됩니다. 이후 pose, left hand, right hand keypoint를 하나의 입력 배열로 합쳐 CNN1D 모델에 넣습니다.

## Sequence Level Detection

`--sequence-level-detection`을 켜면 segment가 끝날 때까지 기다리지 않고, start frame이 확정된 뒤 프레임이 쌓이는 동안 sliding window 방식으로 실시간 추론합니다.

기본값:

```text
window size = 90 frames
stride = 20 frames
```

실행 예시:

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py --sequence-level-detection --show 
```

window와 stride 변경:

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py ^
  --sequence-level-detection ^
  --sequence-window-frames 90 ^
  --sequence-stride-frames 10
```


Sequence mode 동작:

1. 첫 detected frame이 포함된 start frame을 확정합니다.
2. 이후 프레임을 계속 buffer에 추가합니다.
3. `--sequence-window-frames`만큼 프레임이 모이면 해당 window를 모델에 입력합니다.
4. 다음 window 시작점은 `--sequence-stride-frames`만큼 이동합니다.
5. 종료 조건을 만나면 마지막 detected frame을 포함하는 final window를 추가로 추론합니다.
6. 각 window의 top1 결과 중 score가 0.5 이하인 결과는 무시합니다.
7. 남은 top1 결과에서 연속 중복을 제거해 최종 gloss sequence를 출력합니다.

출력 예시:

```text
recording: window=0 type=regular frames=90 frame_range=12-101 ... top1=토끼(3) score=0.9231
recording: window=1 type=regular frames=90 frame_range=32-121 ... top1=토끼(3) score=0.8842
finished: window=2 type=final frames=90 frame_range=55-144 ... top1=달리다(8) score=0.8120
finished: sequence ended. gloss_sequence=토끼 달리다
```

## 비동기 추론

Sequence mode에서는 추론을 별도 worker thread에서 비동기로 실행합니다.

메인 루프는 웹캠 캡처, MediaPipe keypoint 추출, sliding window 생성만 수행하고, 완성된 window를 inference queue에 넣습니다. 모델 추론은 worker가 처리하므로 추론 중에도 카메라 캡처가 멈추는 시간을 줄일 수 있습니다.

## 추론 Window 이미지 저장

모델에 실제로 들어간 frame들을 jpg로 저장하여 확인하려면 `--save-images`를 사용합니다.

```bash
python FairyTaleSL/webcam_cnn1d_realtime.py --sequence-level-detection --save-images
```

top1 score가 0.5 이하인 추론 결과는 무시되며, 이 경우 해당 window 이미지는 저장하지 않습니다.

## 주요 옵션 정리

```text
--config                  MMAction2 config 경로
--checkpoint              모델 checkpoint 경로
--label-map               class id와 gloss label 매핑 json 경로
--device                  auto, cpu, cuda:0 등
--camera                  webcam index. 기본값 0
--width                   webcam width. 기본값 640
--height                  webcam height. 기본값 480
--fps                     webcam FPS. 기본값 30
--topk                    출력할 top-k 개수. 기본값 5
--window-sec              start/end 감지 판단 window 길이
--start-ratio             recording 시작 판단 비율
--end-ratio               recording 종료 판단 비율
--max-gap                 짧은 keypoint 미검출 구간 보간 길이
--min-frames              최소 segment frame 수
--max-record-sec          최대 recording 시간
--model-complexity        MediaPipe Holistic model complexity
--sequence-level-detection sliding window 기반 sequence 추론 사용
--sequence-window-frames  sequence window frame 수
--sequence-stride-frames  sequence sliding stride
--save-images             모델 입력 frame jpg 저장
--save-images-dir         저장 경로
--show                    webcam 화면 표시
--mirror                  webcam 좌우 반전
```

## Project Structure

```text
workspace/
  mmaction2/
  checkpoints/
    best_acc_top1_epoch_36.pth
  FairyTaleSL/
    README.md
    webcam_cnn1d_realtime.py
    src/
      class_labels.json
    saved_inference_windows/
```
