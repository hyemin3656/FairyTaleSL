import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// 프록시 대상:
//   - 로컬 개발: http://localhost:8000 (기본값)
//   - Docker Compose: 컨테이너에서 VITE_PROXY_TARGET=http://backend:8000 으로 override
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const httpTarget = process.env.VITE_PROXY_TARGET || env.VITE_PROXY_TARGET || "http://localhost:8000";
  const wsTarget = httpTarget.replace(/^http/, "ws");

  return {
    plugins: [react()],
    server: {
      port: 5173,
      allowedHosts: ["frontend", "localhost"],
      proxy: {
        "/api": {
          target: httpTarget,
          changeOrigin: true,
        },
        "/ws": {
          target: wsTarget,
          ws: true,
        },
      },
    },
  };
});
