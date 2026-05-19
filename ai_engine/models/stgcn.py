"""
ST-GCN (Spatial-Temporal Graph Convolutional Network) for sign language recognition.

Reference: Yan et al., "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition", AAAI 2018.

Input tensor shape: (B, C, T, V, M)
  B = batch size
  C = 3  (x, y, z coordinates)
  T = temporal frames (variable; default buffer = 30)
  V = 21 (MediaPipe hand landmarks per hand)
  M = 2  (max hands)

Output: (B, T', num_classes) — frame-level logits for CTC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Graph adjacency (MediaPipe hand skeleton) ─────────────────────────────────
# 21 nodes: wrist(0), thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # index
    (5, 9), (9, 10), (10, 11), (11, 12), # middle
    (9, 13), (13, 14), (14, 15), (15, 16),# ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                              # palm
]
NUM_NODES = 21


def build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """정규화된 인접행렬 A (num_nodes × num_nodes)"""
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    A += torch.eye(num_nodes)  # self-loop
    D = A.sum(dim=1).pow(-0.5)
    D[D == float('inf')] = 0
    A = D.unsqueeze(1) * A * D.unsqueeze(0)
    return A


def build_block_adjacency(num_nodes: int, edges: list[tuple[int, int]],
                          num_hands: int) -> torch.Tensor:
    """두 손을 합친 블록 대각 인접행렬 (num_nodes*num_hands × num_nodes*num_hands)"""
    A_single = build_adjacency(num_nodes, edges)
    total = num_nodes * num_hands
    A_block = torch.zeros(total, total)
    for m in range(num_hands):
        s = m * num_nodes
        A_block[s:s+num_nodes, s:s+num_nodes] = A_single
    return A_block


# ── Graph Conv Block ──────────────────────────────────────────────────────────
class GraphConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C)
        x = torch.einsum("btvC,vu->btuC", x, self.A)
        return self.fc(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 temporal_kernel: int = 9, stride: int = 1, dropout: float = 0.5):
        super().__init__()
        pad = (temporal_kernel - 1) // 2
        self.gcn = GraphConv(in_ch, out_ch, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Conv over (T, V): kernel (t_k, 1) for temporal, no spatial mixing
            nn.Conv2d(out_ch, out_ch, (temporal_kernel, 1), stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        # GCN: operate per frame
        xg = x.permute(0, 2, 3, 1)          # (B, T, V, C)
        xg = self.gcn(xg)                    # (B, T, V, out_ch)
        xg = xg.permute(0, 3, 1, 2)         # (B, out_ch, T, V)
        xg = self.tcn(xg) + self.residual(x)
        return self.relu(xg)


# ── ST-GCN Main ───────────────────────────────────────────────────────────────
class STGCN(nn.Module):
    """
    ST-GCN for hand-gesture / sign-language recognition.

    mode='ctc'      : (B, T', num_classes) log-softmax  — 연속수어/CTC 학습용
    mode='classify' : (B, num_classes) logits           — 고립단어 분류용 (CE 손실)
    """

    def __init__(self, num_classes: int, num_nodes: int = NUM_NODES,
                 in_channels: int = 3, num_hands: int = 2,
                 mode: str = "ctc"):
        super().__init__()
        assert mode in {"ctc", "classify"}
        self.mode = mode
        self.num_hands = num_hands
        A = build_block_adjacency(num_nodes, HAND_EDGES, num_hands)  # (V*M, V*M)

        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes * num_hands)

        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64,  A, dropout=0.3),
            STGCNBlock(64,          64,  A, dropout=0.3),
            STGCNBlock(64,          128, A, stride=2, dropout=0.4),
            STGCNBlock(128,         128, A, dropout=0.4),
            STGCNBlock(128,         256, A, stride=2, dropout=0.5),
            STGCNBlock(256,         256, A, dropout=0.5),
        ])

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V, M = x.shape
        x = x.reshape(B, C, T, V * M)

        # 입력 정규화: (B, C*V*M, T) 축으로 BatchNorm1d
        bn_in = x.permute(0, 1, 3, 2).reshape(B, C * V * M, T)
        bn_in = self.input_bn(bn_in)
        x = bn_in.reshape(B, C, V * M, T).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)  # (B, ch, T', V*M)

        x = x.mean(dim=-1)        # (B, ch, T')
        if self.mode == "classify":
            x = x.mean(dim=-1)    # (B, ch)  ← 시간축 mean pool
            return self.fc(x)     # (B, num_classes) logits

        # ctc
        x = x.permute(0, 2, 1)    # (B, T', ch)
        x = self.fc(x)             # (B, T', num_classes)
        return F.log_softmax(x, dim=-1)


# ── Singleton with lazy init ──────────────────────────────────────────────────
_model: STGCN | None = None
_vocab: list[str] = []


def get_model(vocab: list[str], device: str = "cpu",
              mode: str = "classify") -> STGCN:
    global _model, _vocab
    if _model is None:
        # classify: 클래스 수 그대로, ctc: blank 토큰 +1
        num_classes = len(vocab) if mode == "classify" else len(vocab) + 1
        _model = STGCN(num_classes=num_classes, mode=mode).to(device)
        _model.eval()
        _vocab = vocab
    return _model


def load_weights(path: str):
    """실제 학습된 가중치 로드. 없으면 random init 유지."""
    import os
    global _model
    if _model is not None and os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        _model.load_state_dict(state, strict=False)
        print(f"[ST-GCN] weights loaded from {path}")
    else:
        print(f"[ST-GCN] no weights at '{path}' — using random init")
