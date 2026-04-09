import { useNavigate } from "react-router-dom";
import BookList from "../components/book/BookList";

export default function BookSelectPage() {
  const navigate = useNavigate();

  return (
    <div>
      <header style={headerStyle}>
        <button style={backBtnStyle} onClick={() => navigate("/")}>
          ← 홈
        </button>
        <h2 style={headerTitleStyle}>동화책 선택</h2>
      </header>
      <BookList />
    </div>
  );
}

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 16,
  padding: "16px 24px",
  background: "var(--color-surface)",
  borderBottom: "1px solid var(--color-border)",
  position: "sticky",
  top: 0,
  zIndex: 10,
};

const backBtnStyle: React.CSSProperties = {
  padding: "6px 14px",
  borderRadius: 8,
  background: "transparent",
  border: "1.5px solid var(--color-border)",
  fontSize: 14,
  color: "var(--color-text-muted)",
};

const headerTitleStyle: React.CSSProperties = {
  fontSize: 20,
  fontWeight: 700,
};
