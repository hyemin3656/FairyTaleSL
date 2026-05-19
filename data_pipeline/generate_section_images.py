"""
섹션별 동화 삽화 생성 스크립트
Imagen 4.0 API로 귀여운 캐릭터 스타일 이미지 생성
"""
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types

API_KEY   = os.environ["GEMINI_API_KEY"]
MODEL     = "imagen-4.0-fast-generate-001"
OUT_DIR   = Path(__file__).parent.parent / "backend/static/images/sections"
DATA_PATH = Path(__file__).parent / "sign_generation/data/fairy_tales_structured.json"

client = genai.Client(api_key=API_KEY)

STYLE = (
    "cute cartoon illustration, chibi style, bright pastel colors, "
    "friendly and playful, Korean fairy tale, children's book illustration, "
    "simple background, warm lighting, no text"
)

# 섹션별 장면 요약 (한→영 프롬프트용)
SCENE_PROMPTS = {
    ("토끼와_거북이", 1): "a cheerful rabbit and a slow turtle meeting in a sunny forest",
    ("토끼와_거북이", 2): "a tired rabbit sleeping under a tree while a determined turtle walks past toward the finish line",
    ("흥부와_놀부", 1): "a kind poor man with many children living in a small cozy house",
    ("흥부와_놀부", 2): "a man gently healing an injured swallow bird with bandages",
    ("흥부와_놀부", 3): "a family celebrating with a magical gourd overflowing with treasure",
    ("콩쥐팥쥐", 1): "a gentle girl doing chores alone while her stepmother and stepsister watch unkindly",
    ("콩쥐팥쥐", 2): "magical animals helping a sad girl complete impossible tasks at home",
    ("콩쥐팥쥐", 3): "a girl in a beautiful dress meeting a kind prince at a royal gathering",
    ("금도끼_은도끼", 1): "a woodcutter accidentally dropping his axe into a sparkling pond in the forest",
    ("금도끼_은도끼", 2): "a mountain spirit rising from water holding a shining golden axe",
    ("금도끼_은도끼", 3): "an honest woodcutter receiving a reward from a magical spirit with a warm smile",
    ("심청전", 1): "a blind father and his devoted young daughter living humbly together",
    ("심청전", 2): "a brave girl jumping into the ocean to save her father from poverty",
    ("심청전", 3): "a girl emerging from a giant lotus flower as a queen, reuniting with her father who can now see",
    ("단군신화", 1): "a divine being descending from heaven to a mountain with thousands of followers",
    ("단군신화", 2): "a tiger and a bear in a cave eating garlic and mugwort for a hundred days",
    ("단군신화", 3): "a couple standing proudly founding the first Korean kingdom called Gojoseon",
    ("선녀와_나무꾼", 1): "a kind woodcutter saving a deer from hunters in a snowy forest",
    ("선녀와_나무꾼", 2): "a beautiful fairy bathing in a magical pond while her robes are hidden nearby",
    ("선녀와_나무꾼", 3): "a family riding a magical winged horse up into the starry sky",
    ("해님달님", 1): "a mother carrying rice cake through a dark forest while a tiger watches",
    ("해님달님", 2): "a scary tiger disguised as a mother knocking on a door at night",
    ("해님달님", 3): "two children climbing a rope up to the sky to escape from a tiger",
    ("해님달님", 4): "a brother becoming the sun and a sister becoming the moon shining in the sky",
    ("팥죽할머니와_호랑이", 1): "an old grandmother making red bean porridge while a tiger threatens her",
    ("팥죽할머니와_호랑이", 2): "magical helpers including a chestnut, a needle, and a rope waiting to help the grandmother",
    ("팥죽할머니와_호랑이", 3): "a defeated tiger running away after being tricked by clever helpers",
    ("견우와_직녀", 1): "a weaving girl and a cowherd happily falling in love across the Milky Way",
    ("견우와_직녀", 2): "a couple sadly separated by the sky emperor forced to live on opposite sides of the stars",
    ("견우와_직녀", 3): "thousands of magpies forming a bridge across the Milky Way for the couple to meet once a year",
    ("장화홍련", 1): "two sisters living sadly with a cruel stepmother in a traditional Korean house",
    ("장화홍련", 2): "two ghost sisters appearing before a kind official to reveal the truth",
    ("장화홍련", 3): "justice being served and two sisters reborn peacefully in a new family",
}


def generate_image(tale_id: str, segment_idx: int, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [skip] 이미 존재: {out_path.name}")
        return True

    scene = SCENE_PROMPTS.get((tale_id, segment_idx))
    if not scene:
        print(f"  [skip] 프롬프트 없음: {tale_id} seg{segment_idx}")
        return False

    prompt = f"{scene}, {STYLE}"

    try:
        resp = client.models.generate_images(
            model=MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="4:3",
                safety_filter_level="block_low_and_above",
            ),
        )
        if resp.generated_images:
            img_bytes = resp.generated_images[0].image.image_bytes
            out_path.write_bytes(img_bytes)
            print(f"  [ok] {out_path.name}")
            return True
        else:
            print(f"  [fail] 이미지 없음: {tale_id} seg{segment_idx}")
            return False
    except Exception as e:
        print(f"  [error] {tale_id} seg{segment_idx}: {e}")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    books = data if isinstance(data, list) else data.get("books", [])
    ok = fail = 0

    for book in books:
        tale_id = book.get("tale_id", book.get("title", ""))
        title   = book.get("title", tale_id)
        segs    = book.get("segments", book.get("sections", []))
        print(f"\n=== {title} ({len(segs)}섹션) ===")

        for idx, _ in enumerate(segs, start=1):
            fname    = f"{tale_id}_{idx}.png"
            out_path = OUT_DIR / fname
            success  = generate_image(tale_id, idx, out_path)
            if success:
                ok += 1
            else:
                fail += 1
            time.sleep(1.5)  # API rate limit 방지

    print(f"\n완료: {ok}개 성공, {fail}개 실패")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
