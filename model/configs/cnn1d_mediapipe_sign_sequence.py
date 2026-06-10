# Sequence evaluation config for a CNN1D gloss classifier.
# NUM_CLASSES must match the gloss checkpoint output classes, not the number of full sequences.

ANN_FILE = "../../../dataset/gloss_sequences/mediapipe_sign_3d_without_face_pose_score_1.pkl"
MODEL_TYPE = "cnn1d"
NUM_CLASSES = 67
NUM_JOINTS = 65
IN_CHANNELS = 2
INPUT_MODE = "xy"  # xy, xyz, xyscore, or xyzscore
CLIP_LEN = 50
HIDDEN_CHANNELS = (64, 128, 64)
BACKBONE_DROPOUT = 0.1
HEAD_DROPOUT = 0.5

# Regular eval/training defaults kept compatible with the CNN1D gloss baseline.
EPOCHS = 200
BATCH_SIZE = 16
TRAIN_REPEAT = 5
LR = 0.001
WEIGHT_DECAY = 0.05
VAL_BEGIN = 10
VAL_INTERVAL = 1
TEST_NUM_CLIPS = 5
ZERO_PAD_SHORT = False
SHORT_SAMPLE_INTERPOLATION = {
    "enabled": True,
    "target": "clip_len",  # "clip_len", "sampled_frames", or an integer frame count
}

# Used by eval.py --sequence when CLI args are omitted.
SEQUENCE_WINDOW = 35 # fallback for single-scale sequence eval
SEQUENCE_STRIDE = 35 # fallback for single-scale sequence eval
SEQUENCE_WINDOWS = (35, 45)
SEQUENCE_STRIDE_RATIO = 0.7
SEQUENCE_SCORE_THRESHOLD = 0.8
SEQUENCE_COLLAPSE_REPEATS = True
SEQUENCE_INCLUDE_TAIL = True
