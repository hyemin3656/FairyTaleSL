import { useNavigate } from "react-router-dom";
import { useAuthContext } from "../context/AuthContext";

export default function HomePage() {
  const navigate = useNavigate();
  const { user, logout } = useAuthContext();

  return (
    <div style={pageStyle}>
      {/* 우상단 네비 */}
      <div style={navStyle}>
        {user ? (
          <>
            <span style={nickStyle}>{user.nickname}</span>
            <button style={navBtnStyle} onClick={() => navigate("/mypage")}>마이페이지</button>
            <button style={{ ...navBtnStyle, background: "#e2e8f0", color: "#475569" }} onClick={logout}>
              로그아웃
            </button>
          </>
        ) : (
          <>
            <button style={navBtnStyle} onClick={() => navigate("/login")}>로그인</button>
            <button style={{ ...navBtnStyle, background: "#e2e8f0", color: "#475569" }} onClick={() => navigate("/register")}>
              회원가입
            </button>
          </>
        )}
      </div>

      <div style={heroStyle}>
        <h1 style={titleStyle}>PSYcho</h1>
        <p style={subtitleStyle}>수어로 함께 읽는 동화책 플랫폼</p>
        <button style={btnStyle} onClick={() => navigate("/books")}>
          동화책 고르기
        </button>
      </div>
    </div>
  );
}

const pageStyle: React.CSSProperties = {
  minHeight: "100vh",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "linear-gradient(135deg, #ede9fe 0%, #dbeafe 100%)",
  position: "relative",
};
const navStyle: React.CSSProperties = {
  position: "absolute",
  top: 20,
  right: 24,
  display: "flex",
  alignItems: "center",
  gap: 10,
};
const nickStyle: React.CSSProperties = {
  fontSize: 14,
  color: "#475569",
  fontWeight: 600,
};
const navBtnStyle: React.CSSProperties = {
  padding: "8px 16px",
  background: "#4f46e5",
  color: "#fff",
  fontSize: 14,
  fontWeight: 600,
  borderRadius: 8,
};
const heroStyle: React.CSSProperties = {
  textAlign: "center",
  display: "flex",
  flexDirection: "column",
  gap: 20,
  alignItems: "center",
};
const titleStyle: React.CSSProperties = {
  fontSize: 64,
  fontWeight: 800,
  color: "#4f46e5",
  letterSpacing: "-2px",
};
const subtitleStyle: React.CSSProperties = { fontSize: 20, color: "#475569" };
const btnStyle: React.CSSProperties = {
  marginTop: 12,
  padding: "14px 36px",
  background: "#4f46e5",
  color: "#fff",
  fontSize: 17,
  fontWeight: 700,
  borderRadius: 12,
  boxShadow: "0 4px 16px rgba(79,70,229,0.3)",
};
