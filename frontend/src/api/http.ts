import axios from "axios";

const http = axios.create({
  baseURL: "/api/v1",
  timeout: 10_000,
});

// JWT 인터셉터 (Step 2에서 auth 추가 시 활성화)
http.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default http;
