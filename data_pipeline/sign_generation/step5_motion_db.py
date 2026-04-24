"""
gloss_list.json → SQLite motion_db 적재.

keypoint_data를 BLOB으로 직접 저장 후 .npy / .mp4 파일 삭제.
원본 영상 URL은 ksl.mv.db(H2)에 보존됨.

테이블: motion_db
  gloss         : 글로스 (PK)
  keypoint_data : 225차원 float32 numpy 배열 (BLOB)
  emotion_label : 기쁨 | 슬픔 | 분노 | 놀람 | 중립
  fallback_type : NULL | "decompose" | "text"
  is_registered : 1=keypoint 등록 완료, 0=미등록
"""
import json
import sqlite3
import numpy as np
from pathlib import Path

BASE_DIR        = Path(__file__).parent
GLOSS_LIST_PATH = BASE_DIR / "data/gloss_list.json"
DB_PATH         = BASE_DIR / "data/motion_db.sqlite"
VIDEO_DIR       = BASE_DIR / "data/videos"
KEYPOINT_DIR    = BASE_DIR / "data/keypoints"

with open(GLOSS_LIST_PATH, encoding="utf-8") as f:
    gloss_list = json.load(f)

conn = sqlite3.connect(str(DB_PATH))
cur  = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS motion_db (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        gloss         TEXT UNIQUE NOT NULL,
        keypoint_data BLOB,
        emotion_label TEXT DEFAULT '중립',
        fallback_type TEXT,
        is_registered INTEGER DEFAULT 0,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")
conn.commit()

success, fail, deleted_npy, deleted_mp4 = 0, 0, 0, 0

for g in gloss_list:
    gloss         = g["gloss"]
    keypoint_path = g.get("keypoint_path")
    video_path    = g.get("video_path")
    emotion_label = g.get("emotion_label", "중립")
    fallback_type = g.get("fallback_type")

    keypoint_data = None
    is_registered = 0

    if keypoint_path:
        kp = Path(keypoint_path)
        if kp.exists():
            arr = np.load(str(kp))
            keypoint_data = arr.astype(np.float32).tobytes()
            is_registered = 1

    try:
        cur.execute("""
            INSERT INTO motion_db
                (gloss, keypoint_data, emotion_label, fallback_type, is_registered)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(gloss) DO UPDATE SET
                keypoint_data = excluded.keypoint_data,
                emotion_label = excluded.emotion_label,
                fallback_type = excluded.fallback_type,
                is_registered = excluded.is_registered;
        """, (gloss, keypoint_data, emotion_label, fallback_type, is_registered))
        success += 1

        # DB 적재 성공 후 파일 삭제
        if keypoint_path and Path(keypoint_path).exists():
            Path(keypoint_path).unlink()
            deleted_npy += 1
        if video_path and Path(video_path).exists():
            Path(video_path).unlink()
            deleted_mp4 += 1

        # gloss_list에서 파일 경로 제거
        g["keypoint_path"] = None
        g["video_path"] = None

    except Exception as e:
        print(f"  ❌ {gloss}: {e}")
        fail += 1

conn.commit()

with open(GLOSS_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(gloss_list, f, ensure_ascii=False, indent=2)

cur.execute("SELECT COUNT(*) FROM motion_db WHERE is_registered=1")
registered = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM motion_db WHERE fallback_type='text'")
text_fb = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM motion_db WHERE fallback_type='decompose'")
decomp_fb = cur.fetchone()[0]
cur.execute("SELECT emotion_label, COUNT(*) FROM motion_db GROUP BY emotion_label ORDER BY 2 DESC")
emotion_dist = cur.fetchall()

db_size_mb = DB_PATH.stat().st_size / 1024 / 1024

print(f"========== Motion DB 적재 결과 ==========")
print(f"성공: {success}개 | 실패: {fail}개")
print(f"삭제: .npy {deleted_npy}개 / .mp4 {deleted_mp4}개")
print(f"\n등록 현황:")
print(f"  keypoint 등록:      {registered}개")
print(f"  decompose fallback: {decomp_fb}개")
print(f"  text fallback:      {text_fb}개")
print(f"\n감정 분포:")
for emotion, cnt in emotion_dist:
    print(f"  {emotion}: {cnt}개")
print(f"\nDB 크기: {db_size_mb:.1f} MB")
print(f"DB 경로: {DB_PATH}")

cur.close()
conn.close()
