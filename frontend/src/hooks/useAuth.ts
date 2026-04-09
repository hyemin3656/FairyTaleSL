import { useState, useEffect, useCallback } from "react";
import { fetchMe, login as apiLogin, register as apiRegister, type UserOut } from "../api/auth";

export function useAuth() {
  const [user, setUser] = useState<UserOut | null>(null);
  const [loading, setLoading] = useState(true);

  // 앱 시작 시 토큰이 있으면 me 정보 로드
  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) { setLoading(false); return; }
    fetchMe()
      .then(setUser)
      .catch(() => localStorage.removeItem("access_token"))
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const { access_token } = await apiLogin(username, password);
    localStorage.setItem("access_token", access_token);
    const me = await fetchMe();
    setUser(me);
  }, []);

  const register = useCallback(async (payload: {
    email: string; username: string; nickname: string; password: string;
  }) => {
    const { access_token } = await apiRegister(payload);
    localStorage.setItem("access_token", access_token);
    const me = await fetchMe();
    setUser(me);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("access_token");
    setUser(null);
  }, []);

  return { user, loading, login, register, logout };
}
