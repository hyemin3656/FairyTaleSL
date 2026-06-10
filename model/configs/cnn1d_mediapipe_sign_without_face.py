ANN_FILE = "../../../dataset/final_merged_dataset_combined/mediapipe_sign_3d_without_face_pose_score_1_70.pkl"
MODEL_TYPE = "cnn1d"
NUM_CLASSES = 67+70
NUM_JOINTS = 65
IN_CHANNELS = 2
INPUT_MODE = "xy" #xy or xyz or xyscore or xyzscore
CLIP_LEN = 50 #100
HIDDEN_CHANNELS = (64, 128, 64)
BACKBONE_DROPOUT = 0.1
HEAD_DROPOUT = 0.5

EPOCHS = 200
BATCH_SIZE = 16
TRAIN_REPEAT = 5
LR = 0.001
WEIGHT_DECAY = 0.05 #0.01
VAL_BEGIN = 10
VAL_INTERVAL = 1
TEST_NUM_CLIPS = 5
ZERO_PAD_SHORT = False

RANDOM_HORIZONTAL_FLIP = {
    "enabled": False, #True,
    "prob": 0.5,
    "x_min": 0.0,
    "x_max": 1.0,
    "swap_left_right": True,
    "apply_in_test": False,
}
SHORT_SAMPLE_INTERPOLATION = {
    "enabled": False, #True
    "target": "clip_len",  # "clip_len", "sampled_frames", or an integer frame count
}

# Keypoint normalization is applied before frame sampling.
# 65-joint layout: pose[0:23] + left_hand[23:44] + right_hand[44:65].
KEYPOINT_NORMALIZE = {
    "enabled": False,
    "center": False, #"shoulder", torso
    "scale": False, #"shoulder",
    "rotate": False,
    "left_shoulder_index": 11,
    "right_shoulder_index": 12,
    "scale_dims": "xy",
    "eps": 1e-6,
}

