"""
generate_section_images.py
Gemini 2.0 Flash 이미지 생성으로 동화 13편 섹션 삽화 생성 → frontend/public/images/sections/ 저장

실행: python3 generate_section_images.py
      python3 generate_section_images.py --dry-run   (프롬프트만 출력)
      python3 generate_section_images.py --force      (기존 파일 덮어쓰기)
"""
import sys
import json
import base64
import urllib.request
import time
from pathlib import Path

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR.parent / "frontend/public/images/sections"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRY_RUN = "--dry-run" in sys.argv
FORCE   = "--force"   in sys.argv

GEMINI_API_KEY = "AIzaSyA7si88YaJYxO8mD89-sLOVxTknubSggQ4"
GEMINI_MODEL   = "gemini-2.0-flash-preview-image-generation"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

STYLE = (
    "Korean children's book illustration style, "
    "beautiful anime art, soft watercolor wash background, "
    "warm rich colors, expressive characters in traditional Korean Joseon-era clothing, "
    "cinematic composition, detailed scenery, storybook atmosphere, "
    "no text, no letters, no watermark, no signature, 1024x768"
)

# (tale_id, seg_order) → 영어 장면 묘사 프롬프트
PROMPTS: dict[tuple[str, int], str] = {

    # ── 토끼와 거북이 ───────────────────────────────────────────
    ("토끼와_거북이", 1): (
        "Arrogant white rabbit in a bright vest laughing smugly at a steady determined tortoise "
        "at the start of a mountain forest race, cheerful woodland animals watching from the sidelines, "
        "tall Korean pine trees, clear blue sky, dappled sunlight on a winding dirt path"
    ),
    ("토끼와_거북이", 2): (
        "Triumphant green tortoise slowly crossing the finish line decorated with flowers while "
        "an embarrassed sleeping rabbit wakes up under a shady oak tree far behind, "
        "golden sunset rays through the forest, animals cheering and celebrating joyfully"
    ),

    # ── 흥부와 놀부 ─────────────────────────────────────────────
    ("흥부와_놀부",  1): (
        "Warm-hearted poor brother Heungbu in worn simple hanbok gently cradling an injured swallow "
        "with a broken leg, carefully wrapping a tiny splint with a strip of cloth, "
        "modest thatched-roof Korean countryside home interior, warm lamplight, autumn evening"
    ),
    ("흥부와_놀부",  2): (
        "Heungbu's family gathered in amazement as giant gourd splits open spilling gold coins, "
        "silk fabrics, and treasures onto their dirt floor, joyful expressions, tears of happiness, "
        "magical golden light filling the humble room, autumn harvest season"
    ),
    ("흥부와_놀부",  3): (
        "Greedy Nolbu in rich silk hanbok shrieking in horror as his gourds burst open releasing "
        "demons, mud, and rotten things flooding his grand house, karmic punishment scene, "
        "dark stormy sky outside, chaotic and dramatic atmosphere"
    ),

    # ── 콩쥐팥쥐 ───────────────────────────────────────────────
    ("콩쥐팥쥐",    1): (
        "Gentle Cinderella-like girl Kongjwi in ragged clothes kneeling and weeding the garden "
        "under the hot sun, cruel stepmother in fine hanbok watching with a scowl, "
        "Korean traditional courtyard with stone wall, persimmon tree, summer heat"
    ),
    ("콩쥐팥쥐",    2): (
        "Magical scene of a large ox plowing the field by itself, birds weaving straw mats, "
        "a fairy godmother in luminous white robes handing Kongjwi a beautiful jade-green hanbok, "
        "glowing moonlit Korean courtyard, silver lotus flowers blooming, enchanted night"
    ),
    ("콩쥐팥쥐",    3): (
        "Kongjwi in exquisite jade hanbok walking gracefully through a grand festival, "
        "a lotus flower shoe left behind on stone steps, the magistrate watching her depart "
        "with admiration, colorful paper lanterns, joyful village festival atmosphere"
    ),

    # ── 금도끼 은도끼 ───────────────────────────────────────────
    ("금도끼_은도끼", 1): (
        "Distraught woodcutter kneeling at the edge of a crystal-clear forest pond, watching "
        "his only iron axe sinking into the deep green water, tall ancient Korean pines "
        "reflected in the still surface, golden sunbeams, peaceful but sorrowful mood"
    ),
    ("금도끼_은도끼", 2): (
        "Ethereal mountain spirit deity rising from misty water holding a gleaming golden axe "
        "aloft, the honest woodcutter bowing humbly shaking his head, ancient sacred mountain "
        "setting, lotus flowers on water, divine ethereal light emanating from the spirit"
    ),
    ("금도끼_은도끼", 3): (
        "Grateful woodcutter receiving all three axes — iron, silver, and gold — from a smiling "
        "radiant mountain deity, humble woodcutter with tears of joy, forest bathed in golden "
        "divine light, a greedy neighbor watching enviously from behind a tree in the distance"
    ),

    # ── 심청전 ──────────────────────────────────────────────────
    ("심청전",       1): (
        "Young devoted girl Simcheong leading her blind elderly father carefully along a village "
        "path at dusk, worn simple hanbok, modest poverty but filled with loving care, "
        "autumn leaves falling, traditional Korean village with rice fields in background"
    ),
    ("심청전",       2): (
        "Brave Simcheong standing at the bow of a wooden ship in a violent storm, arms outstretched, "
        "sacrificing herself jumping into the raging dark Indangsu sea, sailors watching in "
        "awe and horror, massive waves, dramatic dark sky, white lotus petals falling"
    ),
    ("심청전",       3): (
        "Radiant Simcheong emerging from a giant glowing golden lotus flower in the royal palace "
        "throne room, blind father's eyes wide open with miraculous tears of joy and light, "
        "court officials bowing, divine golden beams of light, breathtaking magical moment"
    ),

    # ── 단군신화 ────────────────────────────────────────────────
    ("단군신화",     1): (
        "Majestic Prince Hwanung descending from the heavens to sacred Mount Taebaek "
        "surrounded by 3000 divine followers, golden celestial light parting the clouds, "
        "ancient towering Korean mountain peaks covered in mist, sacred pine forest below"
    ),
    ("단군신화",     2): (
        "A fierce tiger and a patient bear inside a cold dark cave, sparse light from the entrance, "
        "a bundle of mugwort and garlic on a stone between them, mythological ancient Korea, "
        "the tiger restless while the bear meditates with determination, mystical atmosphere"
    ),
    ("단군신화",     3): (
        "Noble and dignified Dangun, the first king of Gojoseon, standing on a mountain summit "
        "overlooking a vast fertile land at golden sunrise, wearing ceremonial tribal regalia, "
        "ancient tribal people bowing below, eagles soaring in the sky, founding of a nation"
    ),

    # ── 선녀와 나무꾼 ───────────────────────────────────────────
    ("선녀와_나무꾼", 1): (
        "Young woodcutter hiding shimmering fairy robes behind a pine tree while celestial maidens "
        "in white feathered dresses bathe in a moonlit secret forest pool, magical moonbeams "
        "on still water, white cranes nearby, enchanted ancient mountain forest, silver mist"
    ),
    ("선녀와_나무꾼", 2): (
        "Beautiful fairy in white celestial robes discovering her robes gone, children clinging "
        "to her hands as she begins to float upward into the night sky, woodcutter husband "
        "desperately reaching up from below, emotional tearful farewell, rising toward stars"
    ),
    ("선녀와_나무꾼", 3): (
        "Woodcutter riding a heavenly winged horse up through pink sunrise clouds toward the sky, "
        "glimpsing his fairy wife and two children waving from a celestial palace above the clouds, "
        "hopeful and bittersweet, golden light, mountain valley far below"
    ),

    # ── 해님달님 ────────────────────────────────────────────────
    ("해님달님",     1): (
        "Kind mother in traditional Korean dress hurrying home through a dark mountain forest "
        "at night with rice cake on her head, large menacing tiger stalking silently behind her "
        "through the shadowy trees, full moon casting eerie silver light, tension and dread"
    ),
    ("해님달님",     2): (
        "Terrifying tiger wearing the mother's clothes and hat knocking on a wooden house door, "
        "two frightened children peeking through a crack in the door, flickering oil lamp inside, "
        "dark night outside, the tiger's claws visible, heart-pounding suspense"
    ),
    ("해님달님",     3): (
        "Two children desperately climbing a glowing celestial rope descending from colorful dawn "
        "clouds, the snarling tiger below falling from a frayed rope into a millet field, "
        "divine golden light from above, miraculous rescue, dramatic and hopeful moment"
    ),
    ("해님달님",     4): (
        "Radiant sun goddess and gentle moon god shining from the sky over a peaceful Korean "
        "village in spring, flowers blooming in fields, happy children playing, "
        "golden sun rays and silver moonbeams together, serene and joyful resolution"
    ),

    # ── 혹부리 영감 ─────────────────────────────────────────────
    ("혹부리_영감",  1): (
        "Lovable old Korean man with a large goiter on his cheek wandering lost in a dark "
        "enchanted forest at night, fireflies glowing around him, ancient gnarled trees, "
        "distant sound of mysterious music drawing him forward, lonely but curious expression"
    ),
    ("혹부리_영감",  2): (
        "Blue and green Korean goblins (dokkaebi) with clubs, feasting and dancing wildly "
        "around a blazing bonfire in the forest, the old man singing his heart out in the center "
        "while goblins clap and cheer in delight, magical chaotic energy, glowing night scene"
    ),
    ("혹부리_영감",  3): (
        "Delighted old man touching his now-smooth cheek with tears of joy, sunrise over a "
        "Korean countryside village, blooming cherry blossoms, a wide smile on his face, "
        "warm golden morning light, villagers watching in amazement, heartwarming scene"
    ),

    # ── 별을 찾아서 ─────────────────────────────────────────────
    ("별을_찾아서",  1): (
        "Wide-eyed young girl standing on a hilltop at night pointing at the starry sky where "
        "one star has gone dark and missing, deep blue night sky full of glittering stars, "
        "her small determined silhouette, wonder and determination in her expression"
    ),
    ("별을_찾아서",  2): (
        "The girl bouncing lightly on the soft grey surface of the moon, meeting an adorable "
        "white rabbit holding a glowing lantern, vast starry cosmos surrounding them, "
        "Earth visible in the distant sky, magical and dreamlike space fantasy"
    ),
    ("별을_찾아서",  3): (
        "Brave girl in a tiny spacecraft flying through swirling colorful nebulas toward a "
        "black hole, finding and releasing a lost star that blazes back to life, "
        "the entire universe lighting up with cascading rainbow starlight, triumphant moment"
    ),

    # ── 팥죽할머니와 호랑이 ──────────────────────────────────────
    ("팥죽할머니와_호랑이", 1): (
        "Elderly Korean grandmother in simple hanbok picking red beans in an autumn mountain "
        "clearing, a massive striped tiger blocking the path behind her, red maple leaves "
        "falling, tension in the quiet mountain air, the grandmother unaware of the danger"
    ),
    ("팥죽할머니와_호랑이", 2): (
        "Cozy Korean house interior, grandmother stirring a bubbling pot of red bean porridge "
        "over fire while a chestnut hides in the ashes, an awl sticks in the floor, "
        "a millstone waits by the door, magic helpers secretly gathering, warm lamplight"
    ),
    ("팥죽할머니와_호랑이", 3): (
        "Striped tiger shrieking and fleeing into the dark forest, hit by a popping chestnut, "
        "poked by an awl, crushed under a millstone, the grandmother and helpers celebrating "
        "joyfully with bowls of red bean porridge by the warm hearth, happy ending"
    ),

    # ── 견우와 직녀 ─────────────────────────────────────────────
    ("견우와_직녀",  1): (
        "Handsome cowherd Gyeonu and beautiful weaver girl Jiknyeo meeting for the first time "
        "in a heavenly garden among clouds, their eyes meeting with instant love, "
        "a gentle ox beside him, a loom behind her, celestial palace in golden twilight sky"
    ),
    ("견우와_직녀",  2): (
        "Gyeonu and Jiknyeo separated by the vast shimmering Milky Way, each alone on opposite "
        "banks of the silver river of stars, reaching toward each other with tears streaming, "
        "Jiknyeo weeping at her loom, Gyeonu with his faithful ox, night sky filled with stars"
    ),
    ("견우와_직녀",  3): (
        "Magpies and crows forming a living bridge across the Milky Way on Chilseok night, "
        "Gyeonu and Jiknyeo joyfully running toward each other to embrace on the bird bridge, "
        "colorful aurora lights, falling star petals, cosmic celebration scene"
    ),

    # ── 장화홍련 ────────────────────────────────────────────────
    ("장화홍련",    1): (
        "Two beautiful sisters Janghwa and Hongryeon in Joseon-era Korea huddling together "
        "in a cold courtyard while their cruel stepmother in fine clothes looms over them "
        "with a scowl, dark autumn atmosphere, withered persimmon tree, long cold shadows"
    ),
    ("장화홍련",    2): (
        "Two sisters' sorrowful ghosts appearing as pale luminous spirits by a dark moonlit pond, "
        "weeping before the upright magistrate who listens with grave concern, "
        "full moon reflection in still black water, ethereal mist, tragic and haunting scene"
    ),
    ("장화홍련",    3): (
        "Noble magistrate pronouncing justice with scroll in hand in a bright sunlit courtroom, "
        "two sisters reborn as happy children laughing and running through spring cherry blossoms, "
        "pink petals falling, warm sunlight, justice and joy restored, beautiful resolution"
    ),
}


def generate_image(prompt: str, save_path: Path) -> bool:
    if save_path.exists() and not FORCE:
        print(f"    [SKIP] {save_path.name}")
        return True

    full_prompt = f"{prompt}, {STYLE}"
    body = json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            GEMINI_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())

        # 응답에서 인라인 이미지 데이터 추출
        parts = resp["candidates"][0]["content"]["parts"]
        img_part = next((p for p in parts if "inlineData" in p), None)
        if not img_part:
            print(f"    [ERROR] 이미지 데이터 없음: {list(parts[0].keys())}")
            return False

        img_bytes = base64.b64decode(img_part["inlineData"]["data"])
        if len(img_bytes) < 5000:
            print(f"    [ERROR] 응답 너무 작음 ({len(img_bytes)}B)")
            return False

        with open(save_path, "wb") as f:
            f.write(img_bytes)
        print(f"    ✅ {save_path.name} ({len(img_bytes)//1024}KB)")
        return True

    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", errors="replace")[:200]
        print(f"    ❌ HTTP {e.code}: {body_err}")
        return False
    except Exception as e:
        print(f"    ❌ {e}")
        return False


def main():
    success, fail = 0, 0

    for (tale_id, seg_order), prompt in PROMPTS.items():
        fname = f"{tale_id}_{seg_order}.jpg"
        save_path = OUTPUT_DIR / fname

        if DRY_RUN:
            print(f"  [{tale_id} §{seg_order}] {prompt[:100]}…")
            continue

        print(f"\n  [{tale_id} 섹션{seg_order}]")
        ok = generate_image(prompt, save_path)
        if ok:
            success += 1
        else:
            fail += 1
        # 분당 10회 제한 → 7초 간격 (여유 있게)
        if not save_path.exists() or FORCE:
            time.sleep(7)

    if not DRY_RUN:
        print(f"\n{'='*50}")
        print(f"완료: {success}개 성공, {fail}개 실패")
        print(f"저장: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
