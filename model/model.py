from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


class SkeletonCNN1D(nn.Module):
    """1D-CNN backbone matching the MMAction2 SkeletonCNN1D implementation."""

    def __init__(
        self,
        in_channels: int,
        num_joints: int,
        num_person: int = 1,
        hidden_channels: Sequence[int] = (64, 128, 64),
        kernel_size: int = 3,
        pool_kernel_size: int = 2,
        dropout: float = 0.0,
        data_bn: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_person = num_person
        self.out_channels = hidden_channels[-1]

        input_channels = num_joints * in_channels
        self.data_bn = nn.BatchNorm1d(input_channels) if data_bn else nn.Identity()

        layers = []
        current_channels = input_channels
        padding = kernel_size // 2
        for out_channels in hidden_channels:
            layers.extend(
                [
                    nn.Conv1d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=pool_kernel_size),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_channels = out_channels

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M, T, V, C)
        n, m, t, v, c = x.shape
        if v != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, but got {v}.")
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, but got {c}.")

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, v * c, t) #관절 flatten
        x = self.data_bn(x)
        x = self.conv(x)
        return x.view(n, m, self.out_channels, x.size(-1), 1)


class GCNHead(nn.Module):
    """GCN classification head matching MMAction2 GCNHead forward behavior."""

    def __init__(self, num_classes: int, in_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M, C, T, V)
        n, m, c, t, v = x.shape
        x = x.view(n * m, c, t, v)
        x = self.pool(x) #T, V 축에 대한 global average pooling
        x = x.view(n, m, c).mean(dim=1)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, but got {x.shape[1]}.")
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)


class CNN1DRecognizer(nn.Module):
    """RecognizerGCN-style wrapper with MMAction2-compatible state_dict names."""

    def __init__(
        self,
        num_classes: int = 67,
        in_channels: int = 2,
        num_joints: int = 65,
        num_person: int = 1,
        hidden_channels: Sequence[int] = (64, 128, 64),
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = SkeletonCNN1D(
            in_channels=in_channels,
            num_joints=num_joints,
            num_person=num_person,
            hidden_channels=hidden_channels,
            kernel_size=3,
            pool_kernel_size=2,
            dropout=backbone_dropout,
            data_bn=True,
        )
        self.cls_head = GCNHead(
            num_classes=num_classes,
            in_channels=hidden_channels[-1],
            dropout=head_dropout,
        )

    def forward_clip_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"Expected a 6D tensor, got shape {tuple(x.shape)}.")
        n, clips, m, t, v, c = x.shape
        x = x.view(n * clips, m, t, v, c)
        logits = self.cls_head(self.backbone(x))
        return logits.view(n, clips, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Supports (N, M, T, V, C) and test-time (N, clips, M, T, V, C).
        if x.ndim == 6:
            return self.forward_clip_logits(x).mean(dim=1)
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D or 6D tensor, got shape {tuple(x.shape)}.")
        return self.cls_head(self.backbone(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 6:
            return self.forward_clip_logits(x).softmax(dim=-1).mean(dim=1)
        return self.forward(x).softmax(dim=-1)


class BiLSTMRecognizer(nn.Module):
    """Bidirectional LSTM recognizer for skeleton sequences.

    Input shape follows the same convention as CNN1DRecognizer:
    (N, M, T, V, C), with optional test-time clips as
    (N, clips, M, T, V, C).
    """

    def __init__(
        self,
        num_classes: int = 67,
        in_channels: int = 2,
        num_joints: int = 65,
        num_person: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        lstm_dropout: float = 0.3,
        head_dropout: float = 0.5,
        data_bn: bool = True,
        pooling: str = "mean",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'.")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_person = num_person
        self.input_size = num_person * num_joints * in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)

        self.data_bn = nn.BatchNorm1d(self.input_size) if data_bn else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.output_size, num_classes)

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        n, m, t, v, c = x.shape
        if m != self.num_person:
            raise ValueError(f"Expected {self.num_person} persons, but got {m}.")
        if v != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, but got {v}.")
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, but got {c}.")

        x = x.permute(0, 2, 1, 3, 4).contiguous().view(n, t, self.input_size)
        x = self.data_bn(x.reshape(n * t, self.input_size)).view(n, t, self.input_size)
        return x

    def forward_clip_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"Expected a 6D tensor, got shape {tuple(x.shape)}.")
        n, clips, m, t, v, c = x.shape
        x = x.view(n * clips, m, t, v, c)
        logits = self.forward(x)
        return logits.view(n, clips, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 6:
            return self.forward_clip_logits(x).mean(dim=1)
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D or 6D tensor, got shape {tuple(x.shape)}.")

        x = self._flatten_input(x)
        out, _ = self.lstm(x)
        if self.pooling == "mean":
            feat = out.mean(dim=1)
        else:
            feat = out[:, -1]
        feat = self.dropout(feat)
        return self.fc(feat)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 6:
            return self.forward_clip_logits(x).softmax(dim=-1).mean(dim=1)
        return self.forward(x).softmax(dim=-1)


class LSTMRecognizer(BiLSTMRecognizer):
    """Unidirectional LSTM recognizer for skeleton sequences."""

    def __init__(
        self,
        num_classes: int = 67,
        in_channels: int = 2,
        num_joints: int = 65,
        num_person: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        lstm_dropout: float = 0.3,
        head_dropout: float = 0.5,
        data_bn: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            num_joints=num_joints,
            num_person=num_person,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lstm_dropout=lstm_dropout,
            head_dropout=head_dropout,
            data_bn=data_bn,
            pooling=pooling,
            bidirectional=False,
        )


class CNNLSTMRecognizer(nn.Module):
    """CNN + LSTM baseline for skeleton sequences.

    Conv1d first extracts local temporal features from flattened skeleton
    coordinates, then an LSTM models the resulting feature sequence.
    """

    def __init__(
        self,
        num_classes: int = 67,
        in_channels: int = 2,
        num_joints: int = 65,
        num_person: int = 1,
        cnn_channels: Sequence[int] = (64, 128),
        cnn_kernel_size: int = 3,
        cnn_pool_kernel_size: int = 2,
        cnn_dropout: float = 0.1,
        hidden_size: int = 128,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,
        head_dropout: float = 0.5,
        data_bn: bool = True,
        pooling: str = "mean",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'.")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_person = num_person
        self.input_size = num_person * num_joints * in_channels
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)

        self.data_bn = nn.BatchNorm1d(self.input_size) if data_bn else nn.Identity()
        padding = cnn_kernel_size // 2
        current_channels = self.input_size
        conv_layers = []
        for out_channels in cnn_channels:
            conv_layers.extend(
                [
                    nn.Conv1d(
                        current_channels,
                        out_channels,
                        kernel_size=cnn_kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if cnn_pool_kernel_size > 1:
                conv_layers.append(nn.MaxPool1d(kernel_size=cnn_pool_kernel_size))
            if cnn_dropout > 0:
                conv_layers.append(nn.Dropout(cnn_dropout))
            current_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(
            input_size=current_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.output_size, num_classes)

    def _to_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        n, m, t, v, c = x.shape
        if m != self.num_person:
            raise ValueError(f"Expected {self.num_person} persons, but got {m}.")
        if v != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, but got {v}.")
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, but got {c}.")

        x = x.permute(0, 2, 1, 3, 4).contiguous().view(n, t, self.input_size)
        x = self.data_bn(x.reshape(n * t, self.input_size)).view(n, t, self.input_size)
        x = x.transpose(1, 2).contiguous()
        x = self.cnn(x)
        return x.transpose(1, 2).contiguous()

    def forward_clip_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"Expected a 6D tensor, got shape {tuple(x.shape)}.")
        n, clips, m, t, v, c = x.shape
        x = x.view(n * clips, m, t, v, c)
        logits = self.forward(x)
        return logits.view(n, clips, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 6:
            return self.forward_clip_logits(x).mean(dim=1)
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D or 6D tensor, got shape {tuple(x.shape)}.")

        x = self._to_temporal_features(x)
        out, _ = self.lstm(x)
        if self.pooling == "mean":
            feat = out.mean(dim=1)
        else:
            feat = out[:, -1]
        feat = self.dropout(feat)
        return self.fc(feat)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 6:
            return self.forward_clip_logits(x).softmax(dim=-1).mean(dim=1)
        return self.forward(x).softmax(dim=-1)


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return OrderedDict((key[len(prefix) :], value) for key, value in state_dict.items())


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, bytes, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint": checkpoint,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }

