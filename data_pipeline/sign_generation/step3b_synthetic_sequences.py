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


# pose 내 팔 관련 랜드마크 인덱스 (0-based, pose 블록 기준)
# lSh=11, rSh=12, lEl=13, rEl=14, lWr=15, rWr=16
_ARM_POSE_IDX = [11, 12, 13, 14, 15, 16]

def _amplify_sign_pose(sign: np.ndarray, amp: float = 2.0) -> np.ndarray:
    """
    수어자세에서 팔/손목 좌표를 중립 대비 amp배 증폭.

    중립(y=1.0) 기준 손목이 y=0.78이면 차이 0.22를 amp배로 늘려
    y = 1.0 - amp*0.22 = 0.56 으로 이동 → 팔이 더 올라가 보임.
    손 landmark는 변경 없음 (손가락 모양 유지).
    """
    out = sign.copy()
    off = 126  # pose 블록 시작
    neutral_arm = NEUTRAL[off:off + 33 * 3].copy()

    for idx in _ARM_POSE_IDX:
        for ch in range(3):  # x, y, z
            pos     = off + idx * 3 + ch
            n_val   = neutral_arm[idx * 3 + ch]
            s_val   = sign[pos]
            out[pos] = n_val + amp * (s_val - n_val)

    return out


def make_sequence(sign_frame: np.ndarray, amp: float = 2.0) -> np.ndarray:
    """1-프레임 수어자세 → (N_FRAMES, 225) 시퀀스 (포즈 amplification 포함)."""
    amplified = _amplify_sign_pose(sign_frame, amp)
    frames = []
    ease_in_end = 6    # 0~5: ease-in
    hold_end    = 21   # 6~20: 유지
    # 21~29: ease-out

    for i in range(N_FRAMES):
        if i < ease_in_end:
            t = smoothstep(i / ease_in_end)
            f = NEUTRAL * (1 - t) + amplified * t
        elif i < hold_end:
            f = amplified.copy()
        else:
            t = smoothstep((i - hold_end) / (N_FRAMES - hold_end))
            f = amplified * (1 - t) + NEUTRAL * t
        frames.append(f)

    return np.array(frames, dtype=np.float32)  # (N_FRAMES, 225)


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    # ── 1) 실제 영상에서 추출한 keypoint로 synthetic 시퀀스 생성 ─
    # is_registered=1 이지만 프레임 수가 적은 경우만 처리 (= 구 1-프레임 데이터)
    cur.execute("""
        SELECT gloss, keypoint_data FROM motion_db
        WHERE keypoint_data IS NOT NULL AND is_registered = 1
    """)
    rows_real = cur.fetchall()
    updated_real = 0
    for gloss, blob in rows_real:
        arr = np.frombuffer(blob, dtype=np.float32)
        if len(arr) % 225 != 0:
            continue
        n = len(arr) // 225
        if n >= N_FRAMES:
            continue  # 이미 충분한 프레임 (실제 영상 데이터) — 건드리지 않음
        mid = n // 2
        sign_frame = arr[mid * 225 : (mid + 1) * 225].copy()
        seq = make_sequence(sign_frame, amp=2.0)
        cur.execute("UPDATE motion_db SET keypoint_data=? WHERE gloss=?", (seq.tobytes(), gloss))
        updated_real += 1
    conn.commit()
    print(f"실제 keypoint synthetic 변환: {updated_real}개")

    # ── 2) keypoint가 없는 글로스 → 평균 포즈 기반 synthetic ─────
    avg_pose_path = Path("/tmp/avg_sign_pose.npy")
    if not avg_pose_path.exists():
        print("평균 포즈 파일 없음 (/tmp/avg_sign_pose.npy). 먼저 평균 포즈를 계산하세요.")
        conn.close()
        return

    avg_sign = np.load(str(avg_pose_path)).astype(np.float32)
    generic_seq = make_sequence(avg_sign, amp=1.0)  # 평균 포즈는 증폭 없이
    generic_blob = generic_seq.tobytes()

    cur.execute("SELECT COUNT(*) FROM motion_db WHERE keypoint_data IS NULL")
    null_count = cur.fetchone()[0]
    print(f"keypoint 없는 글로스: {null_count}개 → generic synthetic 시퀀스 생성")

    cur.execute("UPDATE motion_db SET keypoint_data=? WHERE keypoint_data IS NULL", (generic_blob,))
    updated_null = cur.rowcount
    conn.commit()
    conn.close()

    print(f"\n완료: 실제→synthetic {updated_real}개, generic {updated_null}개")
    print(f"각 글로스: {N_FRAMES}프레임 @{TARGET_FPS}fps = {N_FRAMES/TARGET_FPS:.1f}초")


if __name__ == "__main__":
    main()
