import {defineConfig} from "@playwright/test";
export default defineConfig({
  testDir:"./tests",testMatch:"experimental-addons.spec.mjs",workers:1,fullyParallel:false,timeout:45000,forbidOnly:true,
  outputDir:"test-results/addons",reporter:[["line"]],
  use:{baseURL:"http://127.0.0.1:8897",headless:true,channel:process.env.GARDEN_BROWSER_CHANNEL||undefined,trace:"retain-on-failure",screenshot:"only-on-failure"},
  webServer:{command:"./scripts/start-admin.sh --addons-only",url:"http://127.0.0.1:8897/api/health",reuseExistingServer:false,timeout:30000,stdout:"pipe",stderr:"pipe"}
});
