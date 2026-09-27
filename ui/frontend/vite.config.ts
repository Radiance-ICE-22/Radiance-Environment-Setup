import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to a backend on :8800 (GALLEY_API overrides).
const api = process.env.GALLEY_API ?? "http://127.0.0.1:8800";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: api, ws: true } } },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1200 },
});
