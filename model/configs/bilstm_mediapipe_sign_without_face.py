ANN_FILE = "../../../dataset/cropped_holistic_results_split/mediapipe_sign_3d_without_face_pose_score_1.pkl"
MODEL_TYPE = "bilstm"

NUM_CLASSES = 67
NUM_JOINTS = 65
NUM_PERSON = 1
IN_CHANNELS = 2
INPUT_MODE = "xy"
CLIP_LEN = 100

HIDDEN_SIZE = 64 #128
NUM_LAYERS = 1 #2
LSTM_DROPOUT = 0.3
HEAD_DROPOUT = 0.7 #0.5
DATA_BN = True
POOLING = "mean"

EPOCHS = 200
BATCH_SIZE = 16
TRAIN_REPEAT = 5
LR = 0.001
WEIGHT_DECAY = 0.1 #0.05
VAL_BEGIN = 5
VAL_INTERVAL = 1
TEST_NUM_CLIPS = 5
ZERO_PAD_SHORT = False

RANDOM_HORIZONTAL_FLIP = {
    "enabled": True,
    "prob": 0.5,
    "x_min": 0.0,
    "x_max": 1.0,
    "swap_left_right": True,
    "apply_in_test": False,
}
