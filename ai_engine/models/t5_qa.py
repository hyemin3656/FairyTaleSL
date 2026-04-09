"""
pko-T5 기반 Q&A 모듈
모델: KETI-AIR/ke-t5-base (HuggingFace Hub — 공개 Korean T5)

기능:
  1. generate_question(text) → 섹션 텍스트로부터 이해도 질문 생성
  2. evaluate_answer(question, context, answer) → 아동 답변 평가 + 피드백
  3. generate_followup_question(context, prev_q, user_a, feedback) → 힌트 포함 후속 질문

T5 모델 다운로드 실패 시 rule-based 폴백으로 자동 전환.
모델 첫 실행 시 HuggingFace에서 자동 다운로드 (~250MB).
이후 ~/.cache/huggingface 에 캐시되어 재사용.
"""
from __future__ import annotations

import re
import threading

_lock = threading.Lock()
_tokenizer = None
_model = None
_t5_available: bool | None = None  # None = 로딩 중 / True = 준비 / False = 실패
_loading_thread: threading.Thread | None = None

MODEL_NAME = "KETI-AIR/ke-t5-base"
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 128


def _do_load():
    """백그라운드 스레드에서 T5 모델 로드."""
    global _tokenizer, _model, _t5_available
    try:
        print(f"[T5] Loading {MODEL_NAME} in background…")
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tok = T5Tokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        mdl = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        mdl.eval()
        with _lock:
            _tokenizer = tok
            _model = mdl
            _t5_available = True
        print("[T5] Model ready.")
    except Exception as e:
        print(f"[T5] Load failed ({e.__class__.__name__}) — using rule-based fallback.")
        with _lock:
            _t5_available = False


def _ensure_loading():
    """첫 호출 시 백그라운드 로드 시작. 준비 여부를 즉시 반환."""
    global _loading_thread, _t5_available
    with _lock:
        if _t5_available is not None:
            return _t5_available
        if _loading_thread is None or not _loading_thread.is_alive():
            _loading_thread = threading.Thread(target=_do_load, daemon=True)
            _loading_thread.start()
    return False  # 아직 로딩 중 → fallback 사용


def _try_load() -> bool:
    """T5 사용 가능 여부 반환 (로딩 중이면 False)."""
    return _ensure_loading() is True or (_t5_available is True)


def preload():
    """서버 시작 시 백그라운드 로드 예약."""
    _ensure_loading()


def _generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    import torch
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True,
    )
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ── Rule-based 폴백 ──────────────────────────────────────────────────────────
_QUESTION_TEMPLATES = [
    "이 이야기에서 주인공은 무엇을 했나요?",
    "이야기에서 가장 중요한 사건은 무엇인가요?",
    "이야기의 결말은 어떻게 되었나요?",
    "주인공이 겪은 어려움은 무엇이었나요?",
    "이 이야기에서 배울 수 있는 교훈은 무엇인가요?",
]

_FOLLOWUP_TEMPLATES = [
    "이야기를 다시 떠올려 보세요. 주인공에게 어떤 일이 일어났나요?",
    "이야기의 내용을 생각해보면, 결국 어떤 일이 벌어졌나요?",
    "힌트: 이야기에서 중요한 사건을 기억해보세요. 무슨 일이 있었나요?",
]

_q_counter = 0
_f_counter = 0


def _fallback_question(section_text: str) -> str:
    """텍스트에서 핵심 단어를 추출해 간단한 질문 생성."""
    global _q_counter
    # 고유명사 / 긴 단어 추출
    words = [w for w in section_text.split() if len(w) >= 2]
    nouns = words[:3] if words else ["주인공"]
    noun_str = ", ".join(nouns)

    templates = [
        f"이야기에서 '{nouns[0]}'(은)는 어떤 일을 했나요?",
        f"'{noun_str}'에 대해 이야기는 무엇이라고 말하나요?",
    ] + _QUESTION_TEMPLATES

    q = templates[_q_counter % len(templates)]
    _q_counter += 1
    return q


def _fallback_evaluate(question: str, context: str, user_answer: str) -> dict:
    """단순 단어 겹침으로 점수 산출."""
    ctx_words = set(re.sub(r"[^\w\s]", "", context).split())
    ans_words = set(re.sub(r"[^\w\s]", "", user_answer).split())
    overlap = len(ans_words & ctx_words)
    score = min(overlap / max(len(ans_words), 1), 1.0)
    score = round(0.3 + score * 0.6, 2)  # 0.3–0.9 범위

    if score >= 0.7:
        feedback = "잘 이해하고 있어요! 훌륭한 답변이에요."
    elif score >= 0.5:
        feedback = "비슷하게 이해하고 있어요. 조금 더 생각해 볼까요?"
    else:
        feedback = "이야기를 다시 읽어보고 한 번 더 답해볼까요?"

    # 힌트: 텍스트 첫 문장
    hint_sentences = re.split(r"[.!?]", context)
    hint = hint_sentences[0].strip() if hint_sentences else context[:60]

    return {"score": score, "feedback": feedback, "correct_hint": hint}


def _fallback_followup(context: str, prev_question: str,
                       user_answer: str, feedback: str) -> str:
    global _f_counter
    hint_sentences = re.split(r"[.!?]", context)
    hint = hint_sentences[0].strip() if hint_sentences else context[:40]

    q = f"힌트: '{hint}'. 이 내용을 바탕으로, 주인공에게 어떤 일이 일어났나요?"
    _f_counter += 1
    return q


# ── 공개 API ──────────────────────────────────────────────────────────────────
def generate_question(section_text: str) -> str:
    if _try_load():
        try:
            prompt = (
                f"다음 이야기를 읽고 이해도를 확인하는 질문을 한 문장으로 만드세요.\n"
                f"이야기: {section_text}\n질문:"
            )
            result = _generate(prompt, max_new_tokens=60)
            result = re.split(r"[.?!。？！]", result)[0].strip()
            if not result.endswith("?"):
                result += "?"
            return result
        except Exception as e:
            print(f"[T5] generate_question error: {e}")
    return _fallback_question(section_text)


def evaluate_answer(question: str, context: str, user_answer: str) -> dict:
    if _try_load():
        try:
            fb_prompt = (
                f"이야기: {context}\n"
                f"질문: {question}\n"
                f"아동 답변: {user_answer}\n"
                f"위 답변이 올바른지 판단하고 친절하게 피드백을 한 문장으로 작성하세요.\n피드백:"
            )
            feedback = _generate(fb_prompt, max_new_tokens=80)

            hint_prompt = (
                f"이야기: {context}\n"
                f"질문: {question}\n"
                f"정답을 짧게 알려주세요.\n정답:"
            )
            hint = _generate(hint_prompt, max_new_tokens=40)

            answer_words = set(user_answer.split())
            hint_words = set(hint.split())
            overlap = len(answer_words & hint_words)
            score = min(overlap / max(len(hint_words), 1), 1.0)
            score = round(0.3 + score * 0.7, 2)

            return {"score": score, "feedback": feedback, "correct_hint": hint}
        except Exception as e:
            print(f"[T5] evaluate_answer error: {e}")
    return _fallback_evaluate(question, context, user_answer)


def generate_followup_question(context: str, prev_question: str,
                               user_answer: str, feedback: str) -> str:
    if _try_load():
        try:
            prompt = (
                f"이야기: {context}\n"
                f"이전 질문: {prev_question}\n"
                f"아동 답변: {user_answer}\n"
                f"피드백: {feedback}\n"
                f"아동이 아직 이해하지 못했습니다. "
                f"이야기의 핵심 내용을 힌트로 포함하여 더 쉬운 질문 한 문장을 만드세요.\n후속 질문:"
            )
            result = _generate(prompt, max_new_tokens=60)
            result = re.split(r"[.?!。？！]", result)[0].strip()
            if not result.endswith("?"):
                result += "?"
            return result
        except Exception as e:
            print(f"[T5] followup error: {e}")
    return _fallback_followup(context, prev_question, user_answer, feedback)
