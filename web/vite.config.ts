import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the interface runs on 5173 and forwards /api to the API on 8000 (uvicorn), so the
// browser sees one address and no CORS is needed. In production pacer.api.web serves both.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
