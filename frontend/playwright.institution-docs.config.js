const { defineConfig, devices } = require('@playwright/test');

/**
 * Socle Playwright — documentation Portail Institution (Lot 1).
 * Indépendant du parcours démo commercial (DEMO_MODE forcé à false).
 * Aucun appel production : API locale uniquement.
 */

// Port 3000 : origine déjà autorisée par le login web local (SOCKETIO_CORS_ORIGINS).
// Un autre port (ex. 3002) est refusé : « Origine non autorisée ».
const FRONTEND_PORT = process.env.INSTITUTION_DOCS_FRONTEND_PORT || '3000';
const API_BASE =
  process.env.INSTITUTION_DOCS_API_URL
  || process.env.REACT_APP_API_BASE_URL
  || 'http://127.0.0.1:5000/api/v1';

module.exports = defineConfig({
  testDir: './e2e/institution-docs',
  testMatch: '**/*.spec.js',
  timeout: 120000,
  expect: {
    timeout: 15000,
  },
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: {
    ...devices['Desktop Chrome'],
    baseURL: `http://127.0.0.1:${FRONTEND_PORT}`,
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 2,
    locale: 'fr-CH',
    timezoneId: 'Europe/Zurich',
    reducedMotion: 'reduce',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'institution-docs',
      use: {
        viewport: { width: 1440, height: 1000 },
        deviceScaleFactor: 2,
        locale: 'fr-CH',
        timezoneId: 'Europe/Zurich',
        reducedMotion: 'reduce',
      },
    },
  ],
  webServer: {
    command: 'npm start',
    url: `http://127.0.0.1:${FRONTEND_PORT}/login`,
    timeout: 180000,
    reuseExistingServer: !process.env.CI,
    env: {
      PORT: FRONTEND_PORT,
      BROWSER: 'none',
      REACT_APP_DEMO_MODE: 'false',
      REACT_APP_API_BASE_URL: API_BASE,
      REACT_APP_API_URL: API_BASE,
    },
  },
});
