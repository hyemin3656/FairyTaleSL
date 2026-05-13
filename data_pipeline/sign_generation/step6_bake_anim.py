"""
step6_bake_anim.py
MediaPipe Holistic keypoint(225차원) → VRM 본 회전 사전 계산 및 SQLite 저장 스크립트

keypoint_data BLOB 형식: float32, shape (N, 225)
  - 0..62:   왼손 (21 landmarks × 3)
  - 63..125: 오른손 (21 landmarks × 3)
  - 126..224: 포즈 (33 landmarks × 3)
"""

import json
import math
import sqlite3
import struct
import sys

import numpy as np

# scipy가 없는 환경을 위한 graceful fallback
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
DB_PATH = "/Users/SJ/PSYcho/FairyTaleSL/data_pipeline/sign_generation/data/motion_db.sqlite"

# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    """값을 [lo, hi] 범위로 제한"""
    return max(lo, min(hi, v))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 벡터 사이의 각도(라디안) 반환. 영벡터 처리 포함."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_val = np.dot(v1, v2) / (n1 * n2)
    cos_val = clamp(float(cos_val), -1.0, 1.0)
    return math.acos(cos_val)


# ---------------------------------------------------------------------------
# 포즈 본 계산
# ---------------------------------------------------------------------------

def compute_pose_bones(pose: np.ndarray) -> dict:
    """
    pose: shape (33, 3) — MediaPipe pose landmarks (정규화 좌표)
    반환: {lArm, rArm, lFore, rFore} 각각 [rx, ry, rz]
    """
    def p(idx):
        return pose[idx]  # (3,) array

    lSh = p(11); lEl = p(13); lWr = p(15)
    rSh = p(12); rEl = p(14); rWr = p(16)

    # 유효성 검사: y > 0 이고 z != 0 (카메라에 감지된 경우)
    hasL = bool(lSh[1] > 0 and lEl[1] != 0)
    hasR = bool(rSh[1] > 0 and rEl[1] != 0)

    # 왼팔
    # MediaPipe d[1]: +0.25 ≈ 팔 자연하강, 0 ≈ 수평(T포즈), -0.2 ≈ 팔 올림
    # VRM lArm.rz: 1.5 = 팔 아래, 0 = T포즈, -1.5 = 팔 위
    if hasL:
        d = lEl - lSh
        lArm_rx = clamp(float(d[0]) * 3.0, -1.0, 0.8)
        lArm_ry = clamp(float(d[2]) * 5.0, -1.5, 0.8)
        lArm_rz = clamp(float(d[1]) * 6.0, -1.5, 1.5)
        lArm = [lArm_rx, lArm_ry, lArm_rz]

        bend = clamp(angle_between(lEl - lSh, lWr - lEl), 0.0, 1.9)
        lFore = [0.0, -bend, 0.0]
    else:
        lArm = [0.0, 0.0, 1.5]
        lFore = [0.0, -0.1, 0.0]

    # 오른팔 (부호 반전)
    if hasR:
        d = rEl - rSh
        rArm_rx = clamp(-float(d[0]) * 3.0, -0.8, 1.0)
        rArm_ry = clamp(-float(d[2]) * 5.0, -0.8, 1.5)
        rArm_rz = clamp(-float(d[1]) * 6.0, -1.5, 1.5)
        rArm = [rArm_rx, rArm_ry, rArm_rz]

        bend = clamp(angle_between(rEl - rSh, rWr - rEl), 0.0, 1.9)
        rFore = [0.0, bend, 0.0]
    else:
        rArm = [0.0, 0.0, -1.5]
        rFore = [0.0, 0.1, 0.0]

    return {
        "lArm": lArm,
        "rArm": rArm,
        "lFore": lFore,
        "rFore": rFore,
    }


# ---------------------------------------------------------------------------
# 손 본 계산
# ---------------------------------------------------------------------------

# 각 손가락의 시작 landmark 인덱스 (hand 21점 기준)
FINGER_STARTS = {
    "Thumb":  1,
    "Index":  5,
    "Middle": 9,
    "Ring":   13,
    "Little": 17,
}


def compute_hand_bones(hand: np.ndarray, side: str) -> dict:
    """
    hand: shape (21, 3) — MediaPipe hand landmarks (정규화 좌표)
    side: 'l' 또는 'r'
    반환: {lHand/rHand, lThumb0..lLittle2 / rThumb0..rLittle2}
    """
    prefix = side  # 'l' or 'r'
    sign = 1.0 if side == "l" else -1.0

    # 손 유효성 확인
    has_hand = any(
        abs(float(hand[i, 0])) + abs(float(hand[i, 1])) > 0.01
        for i in range(21)
    )

    result = {}
    hand_key = f"{prefix}Hand"

    if has_hand:
        w   = hand[0]   # 손목
        iM  = hand[5]   # 검지 MCP
        mM  = hand[9]   # 중지 MCP
        pM  = hand[17]  # 새끼 MCP

        # 손가락 방향 벡터 (2D xy)
        fd = mM - w
        fd_len = math.sqrt(float(fd[0])**2 + float(fd[1])**2)
        if fd_len < 1e-9:
            fd_len = 1e-9
        fdxN = float(fd[0]) / fd_len
        fdyN = float(fd[1]) / fd_len

        # 2D cross product (손 방향 판별)
        v1 = iM - w
        v2 = pM - w
        normZ = float(v1[0]) * float(v2[1]) - float(v1[1]) * float(v2[0])

        fingerUpAngle = math.atan2(fdxN, -fdyN)
        tiltAngle = math.atan2(
            float(iM[1]) - float(pM[1]),
            float(iM[0]) - float(pM[0])
        )

        result[hand_key] = [
            clamp(fingerUpAngle * 0.4 * sign, -0.8, 0.8),
            clamp(-normZ * 3.0 * sign, -1.2, 1.2),
            clamp(tiltAngle * 0.5 * sign, -1.0, 1.0),
        ]

        # 손가락 curl: 5손가락 × 3관절
        for fname, start in FINGER_STARTS.items():
            for j in range(3):
                a = hand[start + j]
                b = hand[start + j + 1]
                dx = float(b[0]) - float(a[0])
                dy = float(b[1]) - float(a[1])
                seg_len = math.sqrt(dx**2 + dy**2)
                if seg_len < 1e-9:
                    seg_len = 1e-9
                fwd_dot = (dx * fdxN + dy * fdyN) / seg_len
                curl = clamp((1.0 - fwd_dot) * 0.8, 0.0, 1.4) * sign
                key = f"{prefix}{fname}{j}"
                result[key] = curl
    else:
        result[hand_key] = [0.0, 0.0, 0.0]
        for fname in FINGER_STARTS:
            for j in range(3):
                result[f"{prefix}{fname}{j}"] = 0.0

    return result


# ---------------------------------------------------------------------------
# 스무딩
# ---------------------------------------------------------------------------

def smooth_channel(arr: np.ndarray) -> np.ndarray:
    """
    1D 시계열 배열에 Savitzky-Golay 필터 또는 3점 이동평균 적용
    arr: shape (N,)
    """
    n = len(arr)
    if n < 3:
        return arr

    if HAS_SCIPY:
        # 홀수 윈도우 보장, 최소 3
        window = min(7, n)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            window = 3
        poly = min(2, window - 1)
        return savgol_filter(arr, window_length=window, polyorder=poly)
    else:
        # 3점 이동평균 fallback
        out = arr.copy().astype(float)
        for i in range(1, n - 1):
            out[i] = (arr[i - 1] + arr[i] + arr[i + 1]) / 3.0
        return out


# ---------------------------------------------------------------------------
# 단일 글로스 변환
# ---------------------------------------------------------------------------

def bake_gloss(keypoint_data: bytes, frame_count: int) -> str:
    """
    keypoint_data: float32 BLOB, shape (N, 225)
    반환: JSON 문자열
    """
    # BLOB → numpy array
    total_floats = len(keypoint_data) // 4
    arr = np.frombuffer(keypoint_data, dtype=np.float32).reshape(-1, 225)
    n = len(arr)

    # 각 채널별 프레임 리스트 초기화
    bone_channels: dict = {
        "lArm":  [[] for _ in range(3)],
        "rArm":  [[] for _ in range(3)],
        "lFore": [[] for _ in range(3)],
        "rFore": [[] for _ in range(3)],
        "lHand": [[] for _ in range(3)],
        "rHand": [[] for _ in range(3)],
    }
    # 손가락 채널
    finger_channels: dict = {}
    for side in ("l", "r"):
        for fname in ("Thumb", "Index", "Middle", "Ring", "Little"):
            for j in range(3):
                finger_channels[f"{side}{fname}{j}"] = []

    for frame_idx in range(n):
        kp = arr[frame_idx]  # (225,)

        # 왼손: indices 0..62 → shape (21, 3)
        lhand = kp[0:63].reshape(21, 3)
        # 오른손: indices 63..125 → shape (21, 3)
        rhand = kp[63:126].reshape(21, 3)
        # 포즈: indices 126..224 → shape (33, 3)
        pose = kp[126:225].reshape(33, 3)

        # 포즈 본
        pb = compute_pose_bones(pose)
        for bone in ("lArm", "rArm", "lFore", "rFore"):
            for ch in range(3):
                bone_channels[bone][ch].append(pb[bone][ch])

        # 왼손 본
        lhb = compute_hand_bones(lhand, "l")
        for ch in range(3):
            bone_channels["lHand"][ch].append(lhb["lHand"][ch])
        for fname in ("Thumb", "Index", "Middle", "Ring", "Little"):
            for j in range(3):
                key = f"l{fname}{j}"
                finger_channels[key].append(lhb[key])

        # 오른손 본
        rhb = compute_hand_bones(rhand, "r")
        for ch in range(3):
            bone_channels["rHand"][ch].append(rhb["rHand"][ch])
        for fname in ("Thumb", "Index", "Middle", "Ring", "Little"):
            for j in range(3):
                key = f"r{fname}{j}"
                finger_channels[key].append(rhb[key])

    # 스무딩 + 라운딩 적용
    def smooth_and_round(lst):
        a = smooth_channel(np.array(lst, dtype=float))
        return [round(float(v), 4) for v in a]

    bones_out: dict = {}

    # 3채널 본: 각 채널 스무딩 후 [rx,ry,rz] 리스트로 병합
    for bone in ("lArm", "rArm", "lFore", "rFore", "lHand", "rHand"):
        smoothed = [smooth_and_round(bone_channels[bone][ch]) for ch in range(3)]
        # 프레임 방향으로 전치: [[rx,ry,rz], ...]
        bones_out[bone] = [
            [smoothed[0][i], smoothed[1][i], smoothed[2][i]]
            for i in range(n)
        ]

    # 손가락 채널: 스칼라 리스트
    for key, lst in finger_channels.items():
        bones_out[key] = smooth_and_round(lst)

    result = {
        "fps": 15,
        "n": n,
        "bones": bones_out,
    }
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 메인 처리
# ---------------------------------------------------------------------------

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # anim_data 컬럼 추가 (이미 있으면 무시)
    try:
        cur.execute("ALTER TABLE motion_db ADD COLUMN anim_data TEXT")
        conn.commit()
        print("anim_data 컬럼 추가 완료")
    except sqlite3.OperationalError:
        pass  # 이미 존재

    # 처리 대상 조회
    cur.execute(
        "SELECT id, gloss, keypoint_data, frame_count FROM motion_db "
        "WHERE keypoint_data IS NOT NULL"
    )
    rows = cur.fetchall()
    total = len(rows)
    print(f"처리 대상: {total}개 글로스")

    ok = 0
    fail = 0

    for i, (row_id, gloss, keypoint_data, frame_count) in enumerate(rows, start=1):
        # 진행 상황 출력 (50개마다)
        if (i - 1) % 50 == 0:
            print(f"변환 중: {gloss} ({i}/{total})")

        try:
            anim_json = bake_gloss(keypoint_data, frame_count or 0)
            cur.execute(
                "UPDATE motion_db SET anim_data = ? WHERE id = ?",
                (anim_json, row_id),
            )
            ok += 1
        except Exception as e:
            print(f"  [오류] id={row_id}, gloss={gloss}: {e}", file=sys.stderr)
            fail += 1

        # 100개마다 커밋
        if i % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print(f"완료: {ok}개 성공, {fail}개 실패")


if __name__ == "__main__":
    main()
