ANN_FILE = "../../../dataset/holistic_result_comp2_aug_final/mediapipe_sign_3d_without_face_pose_score_1.pkl"
MODEL_TYPE = "cnn1d"
NUM_CLASSES = 67 + 134
NUM_JOINTS = 65
IN_CHANNELS = 6
INPUT_MODE = "xyhandrel_bone" #xy, xyz, xyscore, xyzscore, xyhandrel, xyhandrel_norm
CLIP_LEN = 40 #50
HIDDEN_CHANNELS = (64, 128, 64)
BACKBONE_DROPOUT = 0.1 #0.1
HEAD_DROPOUT = 0.5 #0.5

EPOCHS = 200
BATCH_SIZE = 16
TRAIN_REPEAT = 1
LR = 0.001
WEIGHT_DECAY = 0.05 #0.01
VAL_BEGIN = 10
VAL_INTERVAL = 1
TEST_NUM_CLIPS = 5
ZERO_PAD_SHORT = False

RANDOM_HORIZONTAL_FLIP = {
    "enabled": True, #True,
    "prob": 0.5,
    "x_min": 0.0,
    "x_max": 1.0,
    "swap_left_right": True,
    "apply_in_test": False,
}
SHORT_SAMPLE_INTERPOLATION = {
    "enabled": True, #True
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

