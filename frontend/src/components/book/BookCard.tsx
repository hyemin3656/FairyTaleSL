import { useNavigate } from "react-router-dom";
import type { BookListItem } from "../../api/books";
import rabbitTurtleImg from "../../imgs/토끼와 거북이.jpg";
import humpGrandpaImg from "../../imgs/혹부리 영감.jpg";
import findStarImg from "../../imgs/별을 찾아서.jpg";

interface Props {
  book: BookListItem;
}

function pickLocalCover(title: string): string | null {
  if (title.includes("토끼") || title.includes("거북")) return rabbitTurtleImg;
  if (title.includes("혹부리")) return humpGrandpaImg;
  if (title.includes("별")) return findStarImg;
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
          <div style={coverPlaceholderStyle}>
            <span style={{ fontSize: 48 }}>📖</span>
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
  alignItems: "center",
  justifyContent: "center",
  width: "100%",
  height: "100%",
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
