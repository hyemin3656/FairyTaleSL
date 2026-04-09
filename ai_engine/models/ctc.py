"""
CTC Greedy / Beam-Search Decoder.

ST-GCN 출력 (T, num_classes) log-prob 텐서를 받아
글로스 시퀀스 + 신뢰도를 반환한다.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DecodeResult:
    sequence: list[str]          # 예측 글로스 시퀀스
    confidence: float            # 평균 log-prob → exp 변환 [0,1]
    top_gloss: Optional[str]     # 가장 최근 / 단일 글로스 (현재 프레임 결과)


# ── Greedy Decode ─────────────────────────────────────────────────────────────
def greedy_decode(log_probs: torch.Tensor, vocab: list[str],
                  blank_idx: int = 0) -> DecodeResult:
    """
    log_probs: (T, C) — log-softmax 출력
    vocab: 글로스 목록 (blank 제외, 인덱스 0은 blank)
    """
    T, C = log_probs.shape
    argmax = log_probs.argmax(dim=-1).tolist()  # (T,)
    probs = log_probs.exp()

    # Greedy CTC collapse
    prev = blank_idx
    decoded: list[str] = []
    conf_sum = 0.0
    count = 0

    for t, idx in enumerate(argmax):
        prob = probs[t, idx].item()
        if idx != blank_idx and idx != prev:
            # idx → vocab: blank=0, vocab[0]→1 ...
            gloss_idx = idx - 1
            if 0 <= gloss_idx < len(vocab):
                decoded.append(vocab[gloss_idx])
        if idx != blank_idx:
            conf_sum += prob
            count += 1
        prev = idx

    confidence = conf_sum / max(count, 1)
    top_gloss = decoded[-1] if decoded else (vocab[argmax[-1] - 1] if argmax[-1] != blank_idx else None)

    return DecodeResult(
        sequence=decoded,
        confidence=round(confidence, 3),
        top_gloss=top_gloss,
    )


# ── Beam-Search Decode ────────────────────────────────────────────────────────
@dataclass
class _Beam:
    seq: tuple = ()
    log_p_blank: float = 0.0
    log_p_nblank: float = -math.inf


def beam_decode(log_probs: torch.Tensor, vocab: list[str],
                blank_idx: int = 0, beam_width: int = 5) -> DecodeResult:
    """
    log_probs: (T, C)
    Returns best beam result.
    """
    T, C = log_probs.shape
    beams: dict[tuple, _Beam] = {(): _Beam(seq=(), log_p_blank=0.0, log_p_nblank=-math.inf)}

    for t in range(T):
        lp = log_probs[t]  # (C,)
        new_beams: dict[tuple, _Beam] = {}

        for seq, beam in beams.items():
            p_total = math.log(math.exp(beam.log_p_blank) + math.exp(beam.log_p_nblank) + 1e-30)

            # blank 확장
            log_pb_new = p_total + lp[blank_idx].item()
            key = seq
            _merge(new_beams, key, log_p_blank=log_pb_new, log_p_nblank=-math.inf)

            # 비-blank 확장
            for c in range(1, C):
                lpc = lp[c].item()
                last = seq[-1] if seq else None
                if last == c:
                    # repeat: extend only from blank path
                    log_pnb_new = beam.log_p_blank + lpc
                else:
                    log_pnb_new = p_total + lpc
                key2 = seq + (c,)
                _merge(new_beams, key2, log_p_blank=-math.inf, log_p_nblank=log_pnb_new)

        # prune to beam_width
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: math.log(math.exp(kv[1].log_p_blank) + math.exp(kv[1].log_p_nblank) + 1e-30),
                reverse=True,
            )[:beam_width]
        )

    best_seq, best_beam = max(
        beams.items(),
        key=lambda kv: math.log(math.exp(kv[1].log_p_blank) + math.exp(kv[1].log_p_nblank) + 1e-30),
    )

    decoded = [vocab[c - 1] for c in best_seq if 1 <= c <= len(vocab)]
    best_p = math.exp(
        math.log(math.exp(best_beam.log_p_blank) + math.exp(best_beam.log_p_nblank) + 1e-30)
    )
    confidence = round(min(best_p / max(T, 1), 1.0), 3)
    top_gloss = decoded[-1] if decoded else None

    return DecodeResult(sequence=decoded, confidence=confidence, top_gloss=top_gloss)


def _merge(beams: dict, key: tuple, log_p_blank: float, log_p_nblank: float):
    if key not in beams:
        beams[key] = _Beam(seq=key, log_p_blank=log_p_blank, log_p_nblank=log_p_nblank)
    else:
        b = beams[key]
        b.log_p_blank  = _log_add(b.log_p_blank, log_p_blank)
        b.log_p_nblank = _log_add(b.log_p_nblank, log_p_nblank)


def _log_add(a: float, b: float) -> float:
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
