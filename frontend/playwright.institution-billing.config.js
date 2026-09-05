const { defineConfig, devices } = require('@playwright/test');

/**
 * G3 HOLD — navigateur réel, hors demo.
 * toBeInTheDocument seul = FAIL. Voir docs/ops/institution-billing-hold-certification.md
 */
module.exports = defineConfig({
  testDir: './e2e',
  testMatch: '**/institution-billing-g3.spec.js',
  timeout: 120000,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:3001',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'desktop',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 800 },
      },
    },
    {
      name: 'mobile',
      use: {
        ...devices['Pixel 5'],
        viewport: { width: 375, height: 667 },
      },
    },
  ],
  webServer: {
    command: 'npm start',
    url: 'http://127.0.0.1:3001/login',
    timeout: 180000,
    reuseExistingServer: !process.env.CI,
    env: {
      PORT: '3001',
      BROWSER: 'none',
      REACT_APP_DEMO_MODE: 'false',
      REACT_APP_API_BASE_URL: 'http://127.0.0.1:5100/api/v1',
      REACT_APP_API_URL: 'http://127.0.0.1:5100/api/v1',
    },
  },
});
