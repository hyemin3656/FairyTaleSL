from __future__ import annotations

from types import ModuleType

from model import BiLSTMRecognizer, CNN1DRecognizer, CNNLSTMRecognizer, LSTMRecognizer


def get_input_channels(cfg: ModuleType) -> int:
    input_mode = getattr(cfg, "INPUT_MODE", None)
    if input_mode == "xy":
        return 2
    if input_mode == "xyz":
        return 3
    if input_mode == "xyscore":
        return 3
    if input_mode == "xyzscore":
        return 4
    if input_mode == "xyhandrel":
        return 4
    if input_mode == "xyhandrel_norm":
        return 6
    if input_mode == "xyhandrel_bone":
        return 6
    if input_mode is not None:
        raise ValueError("INPUT_MODE must be one of: xy, xyz, xyscore, xyzscore, xyhandrel, xyhandrel_norm, xyhandrel_bone")
    return cfg.IN_CHANNELS


def build_model(cfg: ModuleType):
    model_type = getattr(cfg, "MODEL_TYPE", "cnn1d")
    in_channels = get_input_channels(cfg)
    if model_type == "cnn1d":
        return CNN1DRecognizer(
            num_classes=cfg.NUM_CLASSES,
            in_channels=in_channels,
            num_joints=cfg.NUM_JOINTS,
            hidden_channels=cfg.HIDDEN_CHANNELS,
            backbone_dropout=cfg.BACKBONE_DROPOUT,
            head_dropout=cfg.HEAD_DROPOUT,
        )
    if model_type == "bilstm":
        return BiLSTMRecognizer(
            num_classes=cfg.NUM_CLASSES,
            in_channels=in_channels,
            num_joints=cfg.NUM_JOINTS,
            num_person=getattr(cfg, "NUM_PERSON", 1),
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            lstm_dropout=cfg.LSTM_DROPOUT,
            head_dropout=cfg.HEAD_DROPOUT,
            data_bn=getattr(cfg, "DATA_BN", True),
            pooling=getattr(cfg, "POOLING", "mean"),
        )
    if model_type == "lstm":
        return LSTMRecognizer(
            num_classes=cfg.NUM_CLASSES,
            in_channels=in_channels,
            num_joints=cfg.NUM_JOINTS,
            num_person=getattr(cfg, "NUM_PERSON", 1),
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            lstm_dropout=cfg.LSTM_DROPOUT,
            head_dropout=cfg.HEAD_DROPOUT,
            data_bn=getattr(cfg, "DATA_BN", True),
            pooling=getattr(cfg, "POOLING", "mean"),
        )
    if model_type == "cnn_lstm":
        return CNNLSTMRecognizer(
            num_classes=cfg.NUM_CLASSES,
            in_channels=in_channels,
            num_joints=cfg.NUM_JOINTS,
            num_person=getattr(cfg, "NUM_PERSON", 1),
            cnn_channels=cfg.CNN_CHANNELS,
            cnn_kernel_size=getattr(cfg, "CNN_KERNEL_SIZE", 3),
            cnn_pool_kernel_size=getattr(cfg, "CNN_POOL_KERNEL_SIZE", 2),
            cnn_dropout=getattr(cfg, "CNN_DROPOUT", 0.1),
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            lstm_dropout=cfg.LSTM_DROPOUT,
            head_dropout=cfg.HEAD_DROPOUT,
            data_bn=getattr(cfg, "DATA_BN", True),
            pooling=getattr(cfg, "POOLING", "mean"),
            bidirectional=getattr(cfg, "BIDIRECTIONAL", False),
        )
    raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")
