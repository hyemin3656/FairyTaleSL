import { useNavigate } from "react-router-dom";
import type { BookListItem } from "../../api/books";
import imgGeonwoo     from "../../imgs/견우와 직녀.png";
import imgGoldAxe     from "../../imgs/금도끼 은도끼.png";
import imgDangun      from "../../imgs/단군신화.png";
import imgFindStar    from "../../imgs/별을 찾아서.jpg";
import imgFairy       from "../../imgs/선녀와 나무꾼.png";
import imgSimcheong   from "../../imgs/심청전.png";
import imgJanghwa     from "../../imgs/장화홍련.png";
import imgKongjwi     from "../../imgs/콩쥐팥쥐.png";
import imgRabbit      from "../../imgs/토끼와 거북이.jpg";
import imgPatjuk      from "../../imgs/팥죽 할머니와 호랑이.png";
import imgSun         from "../../imgs/해님 달님.png";
import imgHumpback    from "../../imgs/혹부리 영감.jpg";
import imgHeungbu     from "../../imgs/흥부와 놀부.png";

interface Props {
  book: BookListItem;
}

const COVER_MAP: [string[], string][] = [
  [["견우", "직녀"],            imgGeonwoo],
  [["금도끼", "은도끼"],        imgGoldAxe],
  [["단군"],                    imgDangun],
  [["별이를 찾", "별을 찾"],    imgFindStar],
  [["선녀", "나무꾼"],          imgFairy],
  [["심청"],                    imgSimcheong],
  [["장화", "홍련"],            imgJanghwa],
  [["콩쥐", "팥쥐"],            imgKongjwi],
  [["토끼", "거북"],            imgRabbit],
  [["팥죽", "호랑이"],          imgPatjuk],
  [["해님", "달님"],            imgSun],
  [["혹부리"],                  imgHumpback],
  [["흥부", "놀부"],            imgHeungbu],
];

function pickLocalCover(title: string): string | null {
  for (const [keywords, img] of COVER_MAP) {
    if (keywords.some((kw) => title.includes(kw))) return img;
  }
  return null;
}

export default function BookCard({ book }: Props) {
  const navigate = useNavigate();
  const localCover = pickLocalCover(book.title);
  const coverSrc = localCover ?? book.cover_image_url;

  return (
    <article
      style={cardStyle}
      onClick={() => navigate(`/books/${book.id}`)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && navigate(`/books/${book.id}`)}
    >
      <div style={coverWrapStyle}>
        {coverSrc ? (
          <img src={coverSrc} alt={book.title} style={coverImgStyle} />
        ) : (
          <div style={{ ...coverPlaceholderStyle, background: getCoverTheme(book.title).gradient }}>
            <span style={{ fontSize: 52, filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.3))" }}>
              {getCoverTheme(book.title).emoji}
            </span>
            <span style={coverTitleStyle}>{book.title}</span>
          </div>
        )}
      </div>

      <div style={bodyStyle}>
        <h3 style={titleStyle}>{book.title}</h3>

        {book.author && (
          <p style={authorStyle}>{book.author}</p>
        )}

        {book.description && (
          <p style={descStyle}>{book.description}</p>
        )}

        <div style={metaRowStyle}>
          <span style={ageBadgeStyle}>
            {book.target_age_min}–{book.target_age_max}세
          </span>
          <div style={categoryRowStyle}>
            {book.categories.map((cat) => (
              <span key={cat} style={categoryBadgeStyle}>
                {cat}
              </span>
            ))}
          </div>
        </div>
      </div>
    </article>
  );
}

const COVER_THEMES: { keywords: string[]; gradient: string; emoji: string }[] = [
  { keywords: ["금도끼", "은도끼", "도끼"], gradient: "linear-gradient(135deg,#f6d365,#fda085)", emoji: "🪓" },
  { keywords: ["단군", "환웅", "하늘나라"], gradient: "linear-gradient(135deg,#a18cd1,#fbc2eb)", emoji: "🐻" },
  { keywords: ["선녀", "나무꾼"], gradient: "linear-gradient(135deg,#84fab0,#8fd3f4)", emoji: "🪄" },
  { keywords: ["심청"], gradient: "linear-gradient(135deg,#4facfe,#00f2fe)", emoji: "🌊" },
  { keywords: ["콩쥐", "팥쥐"], gradient: "linear-gradient(135deg,#f093fb,#f5576c)", emoji: "🌸" },
  { keywords: ["토끼", "거북"], gradient: "linear-gradient(135deg,#43e97b,#38f9d7)", emoji: "🐢" },
  { keywords: ["해님", "달님", "해와"], gradient: "linear-gradient(135deg,#f7971e,#ffd200)", emoji: "☀️" },
  { keywords: ["홍부", "놀부"], gradient: "linear-gradient(135deg,#ff6e7f,#bfe9ff)", emoji: "🎵" },
  { keywords: ["흥부"], gradient: "linear-gradient(135deg,#ff6e7f,#bfe9ff)", emoji: "🎵" },
  { keywords: ["호랑이", "곶감"], gradient: "linear-gradient(135deg,#f9844a,#fee140)", emoji: "🐯" },
  { keywords: ["견우", "직녀"], gradient: "linear-gradient(135deg,#667eea,#764ba2)", emoji: "⭐" },
  { keywords: ["신데렐라", "공주"], gradient: "linear-gradient(135deg,#e0c3fc,#8ec5fc)", emoji: "👸" },
];

const DEFAULT_GRADIENTS = [
  "linear-gradient(135deg,#667eea,#764ba2)",
  "linear-gradient(135deg,#f093fb,#f5576c)",
  "linear-gradient(135deg,#4facfe,#00f2fe)",
  "linear-gradient(135deg,#43e97b,#38f9d7)",
];

function getCoverTheme(title: string): { gradient: string; emoji: string } {
  for (const theme of COVER_THEMES) {
    if (theme.keywords.some((kw) => title.includes(kw))) return theme;
  }
  const idx = [...title].reduce((a, c) => a + c.charCodeAt(0), 0) % DEFAULT_GRADIENTS.length;
  return { gradient: DEFAULT_GRADIENTS[idx], emoji: "📖" };
}

const cardStyle: React.CSSProperties = {
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: "var(--radius)",
  boxShadow: "var(--shadow)",
  cursor: "pointer",
  display: "flex",
  flexDirection: "column",
  transition: "transform 0.15s, box-shadow 0.15s",
  overflow: "hidden",
};

const coverWrapStyle: React.CSSProperties = {
  width: "100%",
  aspectRatio: "4 / 3",
  background: "#f1f5f9",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  overflow: "hidden",
};

const coverImgStyle: React.CSSProperties = {
  width: "100%",
  height: "100%",
  objectFit: "cover",
};

const coverPlaceholderStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  gap: 10,
  width: "100%",
  height: "100%",
};

const coverTitleStyle: React.CSSProperties = {
  color: "#fff",
  fontWeight: 800,
  fontSize: 15,
  textShadow: "0 1px 4px rgba(0,0,0,0.4)",
  textAlign: "center",
  padding: "0 12px",
  lineHeight: 1.3,
};

const bodyStyle: React.CSSProperties = {
  padding: "16px",
  display: "flex",
  flexDirection: "column",
  gap: 6,
  flex: 1,
};

const titleStyle: React.CSSProperties = {
  fontSize: 17,
  fontWeight: 700,
  color: "var(--color-text)",
  lineHeight: 1.4,
};

const authorStyle: React.CSSProperties = {
  fontSize: 13,
  color: "var(--color-text-muted)",
};

const descStyle: React.CSSProperties = {
  fontSize: 13,
  color: "var(--color-text-muted)",
  display: "-webkit-box",
  WebkitLineClamp: 2,
  WebkitBoxOrient: "vertical",
  overflow: "hidden",
};

const metaRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  flexWrap: "wrap",
  marginTop: 8,
};

const ageBadgeStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  padding: "2px 8px",
  borderRadius: 20,
  background: "#ede9fe",
  color: "#6d28d9",
};

const categoryBadgeStyle: React.CSSProperties = {
  fontSize: 11,
  padding: "2px 8px",
  borderRadius: 20,
  background: "#f0fdf4",
  color: "#16a34a",
  border: "1px solid #bbf7d0",
};

const categoryRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 4,
  flexWrap: "wrap",
};
