import axios from "axios";

const http = axios.create({
  baseURL: "/api/v1",
  timeout: 30_000,           // /qa/child(Gemini)은 응답까지 수 초 — 10s는 너무 짧음
});

// JWT 인터셉터
http.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 응답 에러: 백엔드의 detail 필드를 Error.message로 끌어올린다.
// (기본 axios 메시지 "Request failed with status code 502"는 사용자에게 무의미)
http.interceptors.response.use(
  (resp) => resp,
  (err) => {
    const detail = err?.response?.data?.detail;
    if (typeof detail === "string" && detail.length > 0) {
      err.message = detail;
    }
    return Promise.reject(err);
  }
);

export default http;
