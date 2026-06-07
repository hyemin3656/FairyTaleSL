import { useNavigate } from "react-router-dom";
import { useAuthContext } from "../context/AuthContext";
import RecommendationCarousel from "../components/home/RecommendationCarousel";
import SafeBoundary from "../components/common/SafeBoundary";

export default function HomePage() {
  const navigate = useNavigate();
  const { user, logout } = useAuthContext();

  return (
    <div style={pageStyle}>
      {/* 좌상단 로고 */}
      <div style={logoStyle} onClick={() => navigate("/")}>
        <span style={logoTextStyle}>FairyTaleSL</span>
      </div>

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
        {/* 로그인 시: 좌(hero) + 우(추천 캐러셀) 2-column. 비로그인 시: hero만 중앙 */}
        <section style={user ? heroRowStyle : heroStyle}>
        <div style={user ? heroLeftStyle : heroStyle}>
          <span style={badgeStyle}>수어 학습 · 전래동화 · 세계명작 · AI 도우미</span>
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
                전래동화부터 세계명작까지, 20권의 이야기를 수어와 함께 읽어 보세요.
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
        </div>
        {user && (
          <SafeBoundary label="RecommendationCarousel">
            <RecommendationCarousel />
          </SafeBoundary>
        )}
        </section>

        {/* 학습 흐름 — 4 step */}
        <section style={stepsWrapStyle}>
          <div style={sectionHeaderStyle}>
            <span style={sectionEyebrowStyle}>HOW IT WORKS</span>
            <h2 style={sectionTitleStyle}>이렇게 학습해요</h2>
            <p style={sectionLeadStyle}>
              읽고 · 따라하고 · 물어보고 · 확인하는 네 단계로 자연스럽게 익혀요.
            </p>
          </div>
          <div style={stepsGridStyle}>
            {STEPS.map((s, i) => (
              <div key={s.title} style={stepCardStyle}>
                <div style={stepNumStyle}>{String(i + 1).padStart(2, "0")}</div>
                <div style={stepIconStyle}>{s.icon}</div>
                <h3 style={stepTitleStyle}>{s.title}</h3>
                <p style={stepDescStyle}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 주요 기능 — 6 cards */}
        <section style={{ width: "100%" }}>
          <div style={sectionHeaderStyle}>
            <span style={sectionEyebrowStyle}>FEATURES</span>
            <h2 style={sectionTitleStyle}>이런 걸 할 수 있어요</h2>
          </div>
          <div style={featuresStyle}>
            {FEATURES.map((f) => (
              <div key={f.title} style={featureCardStyle}>
                <div style={{ ...featureIconStyle, color: f.accent }}>{f.icon}</div>
                <h3 style={featureTitleStyle}>{f.title}</h3>
                <p style={featureDescStyle}>{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 통계 미니 배너 */}
        <section style={statsRowStyle}>
          {STATS.map((s) => (
            <div key={s.label} style={statCellStyle}>
              <div style={statValueStyle}>{s.value}</div>
              <div style={statLabelStyle}>{s.label}</div>
            </div>
          ))}
        </section>

        <footer style={footerStyle}>
          © 2026 FairyTaleSL · 수어로 만나는 따뜻한 동화 시간
        </footer>
      </main>
    </div>
  );
}

const STEPS: { icon: string; title: string; desc: string }[] = [
  {
    icon: "📖",
    title: "동화 읽기",
    desc: "아바타가 동화 본문을 한 섹션씩 수어로 보여줘요.",
  },
  {
    icon: "🤟",
    title: "따라해보기",
    desc: "방금 본 수어를 직접 따라하며 손에 익혀요.",
  },
  {
    icon: "❓",
    title: "질문하기",
    desc: "수어나 글로 자유롭게 물어보면 AI가 답해줘요.",
  },
  {
    icon: "🧠",
    title: "퀴즈로 마무리",
    desc: "짧은 단답형 퀴즈로 이야기를 한 번 더 정리해요.",
  },
];

const FEATURES: { icon: string; title: string; desc: string; accent: string }[] = [
  {
    icon: "📖",
    title: "동화책 20권",
    desc: "한국 전래동화부터 신데렐라 · 인어공주까지, 다양한 장르의 동화를 모았어요.",
    accent: "#7c3aed",
  },
  {
    icon: "🤟",
    title: "문장별 수어 아바타",
    desc: "한 문장씩 따라 읽으며 수어 단어와 표현을 자연스럽게 익혀요.",
    accent: "#4f46e5",
  },
  {
    icon: "🎥",
    title: "실시간 수어 인식",
    desc: "웹캠으로 보여준 수어를 즉시 인식해서 단어로 알려줘요.",
    accent: "#0891b2",
  },
  {
    icon: "🤖",
    title: "AI 도우미",
    desc: "동화 속 인물·사건이 궁금하면 수어·키보드로 자유롭게 물어볼 수 있어요.",
    accent: "#db2777",
  },
  {
    icon: "🧠",
    title: "섹션별 퀴즈",
    desc: "본문에서 곧장 찾을 수 있는 단답형 문제로 이해를 가볍게 점검해요.",
    accent: "#f59e0b",
  },
  {
    icon: "📊",
    title: "학습 이력 저장",
    desc: "어디까지 읽었는지, 어떤 답을 했는지 마이페이지에 기록돼요.",
    accent: "#10b981",
  },
];

const STATS: { value: string; label: string }[] = [
  { value: "20", label: "수록 동화" },
  { value: "12", label: "장르 카테고리" },
  { value: "6,000+", label: "수어 모션 단어" },
  { value: "4–12세", label: "권장 연령" },
];

const pageStyle: React.CSSProperties = {
  minHeight: "100vh",
  background: "linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)",
  position: "relative",
  display: "flex",
  justifyContent: "center",
};
const logoStyle: React.CSSProperties = {
  position: "absolute",
  top: 20,
  left: 24,
  display: "flex",
  alignItems: "center",
  gap: 8,
  cursor: "pointer",
  userSelect: "none",
};
const logoIconStyle: React.CSSProperties = {
  fontSize: 22,
  lineHeight: 1,
};
const logoTextStyle: React.CSSProperties = {
  fontSize: 17,
  fontWeight: 800,
  color: "#4f46e5",
  letterSpacing: "-0.3px",
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
const heroRowStyle: React.CSSProperties = {
  width: "100%",
  display: "flex",
  flexDirection: "row",
  gap: 32,
  alignItems: "center",
  justifyContent: "center",
  flexWrap: "wrap",
};
const heroLeftStyle: React.CSSProperties = {
  textAlign: "center",
  display: "flex",
  flexDirection: "column",
  gap: 16,
  alignItems: "center",
  flex: "1 1 480px",
  maxWidth: 600,
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
const sectionHeaderStyle: React.CSSProperties = {
  textAlign: "center",
  marginBottom: 28,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 8,
};
const sectionEyebrowStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 800,
  color: "#7c3aed",
  letterSpacing: "1.4px",
};
const sectionTitleStyle: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 800,
  color: "#1e293b",
  margin: 0,
  letterSpacing: "-0.4px",
};
const sectionLeadStyle: React.CSSProperties = {
  fontSize: 13,
  color: "#64748b",
  margin: 0,
};

const stepsWrapStyle: React.CSSProperties = {
  width: "100%",
};
const stepsGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
  gap: 12,
};
const stepCardStyle: React.CSSProperties = {
  position: "relative",
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 14,
  padding: "20px 18px 18px",
  display: "flex",
  flexDirection: "column",
  gap: 6,
  boxShadow: "0 1px 3px rgba(15,23,42,0.04)",
};
const stepNumStyle: React.CSSProperties = {
  position: "absolute",
  top: 12,
  right: 14,
  fontSize: 12,
  fontWeight: 800,
  color: "#c7d2fe",
  letterSpacing: 1,
};
const stepIconStyle: React.CSSProperties = {
  fontSize: 28,
  lineHeight: 1,
  marginBottom: 4,
};
const stepTitleStyle: React.CSSProperties = {
  fontSize: 15,
  fontWeight: 800,
  color: "#1e293b",
  margin: 0,
};
const stepDescStyle: React.CSSProperties = {
  fontSize: 12.5,
  color: "#64748b",
  lineHeight: 1.6,
  margin: 0,
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
  transition: "transform 0.15s, box-shadow 0.15s",
  boxShadow: "0 1px 3px rgba(15,23,42,0.04)",
};
const featureIconStyle: React.CSSProperties = {
  fontSize: 28,
  lineHeight: 1,
};
const featureTitleStyle: React.CSSProperties = {
  fontSize: 15,
  fontWeight: 800,
  color: "#1e293b",
  margin: 0,
};
const featureDescStyle: React.CSSProperties = {
  fontSize: 13,
  color: "#64748b",
  lineHeight: 1.6,
  margin: 0,
};

const statsRowStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
  gap: 12,
  width: "100%",
  padding: "22px 18px",
  background: "linear-gradient(135deg, #eef2ff 0%, #fdf4ff 100%)",
  border: "1px solid #e9d5ff",
  borderRadius: 16,
};
const statCellStyle: React.CSSProperties = {
  textAlign: "center",
  display: "flex",
  flexDirection: "column",
  gap: 4,
};
const statValueStyle: React.CSSProperties = {
  fontSize: 26,
  fontWeight: 800,
  color: "#4f46e5",
  letterSpacing: "-0.5px",
};
const statLabelStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#64748b",
  fontWeight: 600,
};

const footerStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#94a3b8",
  textAlign: "center",
};
