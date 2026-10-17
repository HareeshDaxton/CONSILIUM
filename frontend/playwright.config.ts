import { defineConfig, devices } from "@playwright/test";

// Phase 12.4 clinician walkthrough in a real browser against the offline dev
// stack (scripts/dev_ui_stack.py: real API + Postgres + worker, canned LLM,
// fake vision over HTTP). Ports differ from the compose stack (8000/5173) so a
// running stack is never touched; the API database is reset and re-seeded on
// every run. Needs `docker compose up -d postgres`. Not part of the offline CI
// gate (it needs Postgres and a browser).
const API_PORT = Number(process.env.E2E_API_PORT ?? 8765);
const UI_PORT = Number(process.env.E2E_UI_PORT ?? 5199);

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${UI_PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `uv run python scripts/dev_ui_stack.py --reset --no-google --port ${API_PORT} --out frontend/e2e/.stack.json`,
      cwd: "..",
      url: `http://127.0.0.1:${API_PORT}/readyz`,
      reuseExistingServer: false,
      timeout: 180_000,
      stdout: "pipe",
    },
    {
      command: `npm run dev -- --port ${UI_PORT}`,
      url: `http://localhost:${UI_PORT}`,
      env: { VITE_API_PROXY: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
