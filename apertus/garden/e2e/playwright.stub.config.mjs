import {defineConfig} from "@playwright/test";

const baseURL = process.env.GARDEN_STUB_BASE_URL || "http://127.0.0.1:8876";
const externalServer = Boolean(process.env.GARDEN_STUB_BASE_URL);

export default defineConfig({
  testDir: "./tests",
  testMatch: ["stub.spec.mjs", "api-client.spec.mjs", "model-comparison.spec.mjs", "detector-readers.spec.mjs"],
  fullyParallel: false,
  workers: 1,
  timeout: 45_000,
  expect: {
    timeout: 10_000
  },
  forbidOnly: true,
  outputDir: "test-results/stub",
  reporter: [
    ["line"],
    ["html", {open: "never", outputFolder: "playwright-report/stub"}]
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
        command: "./scripts/start-stub.sh",
        url: `${baseURL}/api/health`,
        timeout: 30_000,
        reuseExistingServer: false,
        stdout: "pipe",
        stderr: "pipe"
      }
});
