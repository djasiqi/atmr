const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './e2e',
  timeout: 120000,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm start',
    url: 'http://127.0.0.1:3000/login',
    timeout: 180000,
    reuseExistingServer: false,
    env: {
      BROWSER: 'none',
      REACT_APP_DEMO_MODE: 'true',
      REACT_APP_API_BASE_URL: 'http://127.0.0.1:5100/api/v1',
      REACT_APP_API_URL: 'http://127.0.0.1:5100/api/v1',
    },
  },
});
