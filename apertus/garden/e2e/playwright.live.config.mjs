import {defineConfig} from "@playwright/test";

const baseURL = process.env.GARDEN_LIVE_BASE_URL || "http://127.0.0.1:8877";
const externalServer = Boolean(process.env.GARDEN_LIVE_BASE_URL);

export default defineConfig({
  testDir: "./tests",
  testMatch: ["live.spec.mjs", "eurollm-live.spec.mjs"],
  fullyParallel: false,
  workers: 1,
  timeout: 180_000,
  expect: {
    timeout: 30_000
  },
  forbidOnly: true,
  outputDir: "test-results/live",
  reporter: [
    ["line"],
    ["html", {open: "never", outputFolder: "playwright-report/live"}]
  ],
  use: {
    baseURL,
    headless: true,
    channel: process.env.GARDEN_BROWSER_CHANNEL || undefined,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off"
  },
  webServer: externalServer
    ? undefined
    : {
        command: "./scripts/start-live.sh",
        url: `${baseURL}/api/health`,
        timeout: 120_000,
        reuseExistingServer: false,
        stdout: "pipe",
        stderr: "pipe"
      }
});
