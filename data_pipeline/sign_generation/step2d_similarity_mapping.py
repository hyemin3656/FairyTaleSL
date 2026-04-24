"""
step2d_similarity_mapping.py
video/keypoint가 없는 fallback 글로스를 sentence-transformers 임베딩으로
등록된 글로스 중 가장 유사한 글로스의 keypoint와 매핑

실행: python step2d_similarity_mapping.py
결과: gloss_list.json의 fallback_type="decompose", fallback_glosses=[유사글로스] 로 업데이트
"""
import json
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("sentence-transformers 설치 필요: pip install sentence-transformers")

BASE_DIR = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"
SIMILARITY_THRESHOLD = 0.60  # 이 값 이상일 때만 매핑

with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
    gloss_list = json.load(f)

# 등록된 글로스 (keypoint 보유)
registered = [g for g in gloss_list if g.get("keypoint_path")]
registered_glosses = [g["gloss"] for g in registered]

# 매핑 대상: video_path, keypoint_path 모두 없는 글로스
unmapped = [
    g for g in gloss_list
    if not g.get("keypoint_path") and not g.get("video_path")
]
print(f"등록 글로스: {len(registered)}개")
print(f"매핑 대상:   {len(unmapped)}개\n")

if not unmapped:
    print("매핑 대상이 없습니다.")
    raise SystemExit(0)

# ── 임베딩 ───────────────────────────────────────────────────────
model = SentenceTransformer("jhgan/ko-sroberta-multitask")
print("임베딩 계산 중...")

ref_embs = model.encode(registered_glosses, normalize_embeddings=True)   # (N, D)
tgt_embs = model.encode(
    [g["gloss"] for g in unmapped], normalize_embeddings=True
)  # (M, D)

# cosine similarity (이미 normalize 완료 → dot product = cosine)
sims = tgt_embs @ ref_embs.T   # (M, N)

updated = 0
skipped = 0

for i, g in enumerate(unmapped):
    scores = sims[i]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_gloss = registered_glosses[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        print(f"  ✅ {g['gloss']} → {best_gloss} (cos={best_score:.3f})")
        g["fallback_type"] = "decompose"
        g["fallback_glosses"] = [best_gloss]
        updated += 1
    else:
        print(f"  ⚠️  {g['gloss']} → {best_gloss} (cos={best_score:.3f}, 임계값 미달 — text 유지)")
        skipped += 1

# 결과 저장
with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(gloss_list, f, ensure_ascii=False, indent=2)

print(f"\n{'='*40}")
print(f"✅ 매핑 성공: {updated}개")
print(f"⚠️  임계값 미달 (text fallback 유지): {skipped}개")
print("\n다음 단계: python step5_seed_motion_db.py (DB 재시딩)")
