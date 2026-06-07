import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuthContext } from "../context/AuthContext";

export default function RegisterPage() {
  const { register } = useAuthContext();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    email: "",
    username: "",
    nickname: "",
    password: "",
    confirm: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((prev) => ({ ...prev, [k]: e.target.value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (form.password !== form.confirm) {
      setError("비밀번호가 일치하지 않습니다.");
      return;
    }
    setLoading(true);
    try {
      await register({
        email: form.email,
        username: form.username,
        nickname: form.nickname,
        password: form.password,
      });
      navigate("/");
    } catch (err: any) {
      setError(normalizeError(err) || "회원가입에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  // FastAPI 422는 detail이 객체 배열({type, loc, msg, ...})로 오므로 문자열로 정규화
  function normalizeError(err: any): string {
    const d = err?.response?.data?.detail;
    if (!d) return err?.message ?? "";
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((e: any) => {
          if (typeof e === "string") return e;
          const field = Array.isArray(e?.loc) ? e.loc.slice(1).join(".") : "";
          return field ? `${field}: ${e?.msg ?? "오류"}` : e?.msg ?? JSON.stringify(e);
        })
        .join(", ");
    }
    if (typeof d === "object") return d.msg ?? JSON.stringify(d);
    return String(d);
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">PSYcho</div>
        <h2 className="auth-title">회원가입</h2>

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label>이메일</label>
            <input type="email" value={form.email} onChange={set("email")} placeholder="이메일 입력" required />
          </div>
          <div className="form-group">
            <label>아이디</label>
            <input type="text" value={form.username} onChange={set("username")} placeholder="2~50자" minLength={2} required />
          </div>
          <div className="form-group">
            <label>닉네임</label>
            <input type="text" value={form.nickname} onChange={set("nickname")} placeholder="표시될 이름" required />
          </div>
          <div className="form-group">
            <label>비밀번호</label>
            <input type="password" value={form.password} onChange={set("password")} placeholder="6자 이상" minLength={6} required />
          </div>
          <div className="form-group">
            <label>비밀번호 확인</label>
            <input type="password" value={form.confirm} onChange={set("confirm")} placeholder="비밀번호 재입력" required />
          </div>

          {error && <p className="auth-error">{error}</p>}

          <button className="btn-auth" type="submit" disabled={loading}>
            {loading ? "가입 중…" : "가입하기"}
          </button>
        </form>

        <p className="auth-switch">
          이미 계정이 있으신가요? <Link to="/login">로그인</Link>
        </p>
      </div>
    </div>
  );
}
