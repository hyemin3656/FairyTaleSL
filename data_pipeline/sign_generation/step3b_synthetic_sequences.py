"""
step3b_synthetic_sequences.py
motion_db.sqlite의 1-프레임 keypoint → 30-프레임 합성 시퀀스로 확장.

생성 패턴 (15fps, 2초):
  0~5  프레임: 중립 → 수어자세 ease-in (smoothstep)
  6~20 프레임: 수어자세 유지
  21~29 프레임: 수어자세 → 중립 ease-out (smoothstep)

중립 자세:
  - 손 landmark(0..125): 모두 0 (감지 안됨 = 팔 내림 상태)
  - 포즈 landmark(126..224): 팔이 자연스럽게 내려간 값
    어깨(11,12): x=0.3/0.7, y=0.5
    팔꿈치(13,14): x=0.3/0.7, y=0.75
    손목(15,16): x=0.3/0.7, y=1.0
    나머지: 0
"""
import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / "data/motion_db.sqlite"

TARGET_FPS = 15
N_FRAMES   = 30  # 2초

# ── 중립 포즈 225-dim 벡터 ────────────────────────────────────
# 인덱스 구조: lhand(0..62) + rhand(63..125) + pose(126..224)
# pose 내 인덱스(0-based): 11=lSh, 12=rSh, 13=lEl, 14=rEl, 15=lWr, 16=rWr
def _neutral_pose() -> np.ndarray:
    v = np.zeros(225, dtype=np.float32)
    # pose offset = 126
    off = 126

    def set_lm(idx: int, x: float, y: float, z: float = 0.0):
        v[off + idx * 3]     = x
        v[off + idx * 3 + 1] = y
        v[off + idx * 3 + 2] = z

    # 어깨
    set_lm(11, 0.3, 0.5)   # 왼쪽 어깨
    set_lm(12, 0.7, 0.5)   # 오른쪽 어깨
    # 팔꿈치
    set_lm(13, 0.3, 0.75)  # 왼쪽 팔꿈치
    set_lm(14, 0.7, 0.75)  # 오른쪽 팔꿈치
    # 손목 (y=1.0 → 아래쪽 끝, 팔 완전히 내림)
    set_lm(15, 0.3, 1.0)   # 왼쪽 손목
    set_lm(16, 0.7, 1.0)   # 오른쪽 손목
    return v

NEUTRAL = _neutral_pose()


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def make_sequence(sign_frame: np.ndarray) -> np.ndarray:
    """1-프레임 수어자세 → (N_FRAMES, 225) 시퀀스."""
    frames = []
    ease_in_end  = 6    # 0~5: ease-in
    hold_end     = 21   # 6~20: 유지
    # 21~29: ease-out

    for i in range(N_FRAMES):
        if i < ease_in_end:
            t = smoothstep(i / ease_in_end)
            f = NEUTRAL * (1 - t) + sign_frame * t
        elif i < hold_end:
            f = sign_frame.copy()
        else:
            t = smoothstep((i - hold_end) / (N_FRAMES - hold_end))
            f = sign_frame * (1 - t) + NEUTRAL * t
        frames.append(f)

    return np.array(frames, dtype=np.float32)  # (N_FRAMES, 225)


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM motion_db WHERE keypoint_data IS NOT NULL")
    total = cur.fetchone()[0]
    print(f"keypoint 있는 글로스: {total}개 → {N_FRAMES}-프레임 시퀀스 생성")

    cur.execute("SELECT gloss, keypoint_data FROM motion_db WHERE keypoint_data IS NOT NULL")
    rows = cur.fetchall()

    updated = 0
    skipped = 0
    bad_dim = 0

    for gloss, blob in rows:
        arr = np.frombuffer(blob, dtype=np.float32)

        # 225차원이 아닌 keypoint는 건너뜀
        if len(arr) % 225 != 0:
            bad_dim += 1
            continue

        n_frames = max(1, len(arr) // 225)

        if n_frames >= N_FRAMES:
            skipped += 1
            continue

        # 대표 프레임: 기존 시퀀스의 중간 프레임
        mid = n_frames // 2
        sign_frame = arr[mid * 225 : (mid + 1) * 225].copy()

        seq = make_sequence(sign_frame)
        new_blob = seq.tobytes()

        cur.execute(
            "UPDATE motion_db SET keypoint_data=? WHERE gloss=?",
            (new_blob, gloss)
        )
        updated += 1
        if updated % 500 == 0:
            print(f"  {updated}/{len(rows)} 완료...")
            conn.commit()

    conn.commit()
    conn.close()

    print(f"\n완료: {updated}개 업데이트, {skipped}개 건너뜀 (이미 {N_FRAMES}프레임+), {bad_dim}개 차원 불일치")
    print(f"각 글로스: {N_FRAMES}프레임 @{TARGET_FPS}fps = {N_FRAMES/TARGET_FPS:.1f}초")


if __name__ == "__main__":
    main()
