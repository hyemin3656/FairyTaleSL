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

      <main style={mainStyle}>
        <section style={heroStyle}>
          <span style={badgeStyle}>수어 학습 · 전래동화</span>
          <h1 style={titleStyle}>FairyTaleSL</h1>
          {user ? (
            <>
              <p style={subtitleStyle}>
                안녕하세요, <strong style={{ color: "#4338ca" }}>{user.nickname}</strong>님 👋
              </p>
              <p style={leadStyle}>
                오늘은 어떤 이야기를 함께 읽어볼까요?
                <br />
                마지막에 읽던 동화부터 이어서 학습할 수 있어요.
              </p>
              <div style={ctaRowStyle}>
                <button style={primaryBtnStyle} onClick={() => navigate("/books")}>
                  동화책 보러 가기
                </button>
                <button style={secondaryBtnStyle} onClick={() => navigate("/mypage")}>
                  내 학습 기록
                </button>
              </div>
            </>
          ) : (
            <>
              <p style={subtitleStyle}>수어로 함께 읽는 동화책 플랫폼</p>
              <p style={leadStyle}>
                귀에 익은 전래동화를 수어와 함께 읽어 보세요.
                <br />
                아이와 가족이 손끝으로 이야기를 나누는 가장 따뜻한 방법입니다.
              </p>
              <div style={ctaRowStyle}>
                <button style={primaryBtnStyle} onClick={() => navigate("/books")}>
                  동화책 고르기
                </button>
              </div>
              <p style={loginHintStyle}>
                학습 기록을 저장하려면 <a style={linkStyle} onClick={() => navigate("/login")}>로그인</a>하거나{" "}
                <a style={linkStyle} onClick={() => navigate("/register")}>회원가입</a>하세요.
              </p>
            </>
          )}
        </section>

        <section style={featuresStyle}>
          {FEATURES.map((f) => (
            <div key={f.title} style={featureCardStyle}>
              <div style={featureIconStyle}>{f.icon}</div>
              <h3 style={featureTitleStyle}>{f.title}</h3>
              <p style={featureDescStyle}>{f.desc}</p>
            </div>
          ))}
        </section>

        <footer style={footerStyle}>
          © 2026 FairyTaleSL · 종합설계 프로젝트
        </footer>
      </main>
    </div>
  );
}

const FEATURES: { icon: string; title: string; desc: string }[] = [
  {
    icon: "📖",
    title: "전래동화 13권",
    desc: "토끼와 거북이부터 별을 찾아서까지, 친숙한 이야기들을 모았습니다.",
  },
  {
    icon: "🤟",
    title: "문장별 수어 영상",
    desc: "한 문장씩 따라 읽으며 수어 단어와 표현을 자연스럽게 익힙니다.",
  },
  {
    icon: "🌱",
    title: "어린이 눈높이",
    desc: "5–12세 연령대에 맞춘 짧은 단락과 부드러운 화면 구성을 제공합니다.",
  },
];

const pageStyle: React.CSSProperties = {
  minHeight: "100vh",
  background: "linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)",
  position: "relative",
  display: "flex",
  justifyContent: "center",
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
const mainStyle: React.CSSProperties = {
  width: "100%",
  maxWidth: 1040,
  padding: "120px 24px 48px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 72,
};
const heroStyle: React.CSSProperties = {
  textAlign: "center",
  display: "flex",
  flexDirection: "column",
  gap: 16,
  alignItems: "center",
};
const badgeStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "4px 12px",
  borderRadius: 999,
  background: "#eef2ff",
  border: "1px solid #c7d2fe",
  color: "#4338ca",
  fontSize: 12,
  fontWeight: 600,
  letterSpacing: 0.2,
};
const titleStyle: React.CSSProperties = {
  fontSize: 56,
  fontWeight: 800,
  color: "#4f46e5",
  letterSpacing: "-1.5px",
  margin: 0,
};
const subtitleStyle: React.CSSProperties = {
  fontSize: 18,
  color: "#475569",
  margin: 0,
};
const leadStyle: React.CSSProperties = {
  fontSize: 14,
  color: "#64748b",
  lineHeight: 1.7,
  maxWidth: 480,
  margin: "8px 0 0",
};
const ctaRowStyle: React.CSSProperties = {
  marginTop: 12,
  display: "flex",
  gap: 12,
  flexWrap: "wrap",
  justifyContent: "center",
};
const primaryBtnStyle: React.CSSProperties = {
  padding: "12px 28px",
  background: "#4f46e5",
  color: "#fff",
  fontSize: 15,
  fontWeight: 700,
  borderRadius: 10,
  boxShadow: "0 2px 8px rgba(79,70,229,0.18)",
};
const secondaryBtnStyle: React.CSSProperties = {
  padding: "12px 24px",
  background: "#fff",
  color: "#4338ca",
  fontSize: 15,
  fontWeight: 600,
  borderRadius: 10,
  border: "1px solid #c7d2fe",
};
const loginHintStyle: React.CSSProperties = {
  fontSize: 13,
  color: "#94a3b8",
  marginTop: 4,
};
const linkStyle: React.CSSProperties = {
  color: "#4f46e5",
  fontWeight: 600,
  cursor: "pointer",
  textDecoration: "underline",
};
const featuresStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
  gap: 16,
  width: "100%",
};
const featureCardStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 14,
  padding: "22px 20px",
  display: "flex",
  flexDirection: "column",
  gap: 8,
};
const featureIconStyle: React.CSSProperties = {
  fontSize: 26,
};
const featureTitleStyle: React.CSSProperties = {
  fontSize: 15,
  fontWeight: 700,
  color: "#1e293b",
  margin: 0,
};
const featureDescStyle: React.CSSProperties = {
  fontSize: 13,
  color: "#64748b",
  lineHeight: 1.6,
  margin: 0,
};
const footerStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#94a3b8",
  textAlign: "center",
};
