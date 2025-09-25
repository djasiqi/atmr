// src/setupTests.js
import '@testing-library/jest-dom';

// --- DÉBUT DE LA CORRECTION ---
// Polyfill pour window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // déprécié
    removeListener: jest.fn(), // déprécié
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});
// --- FIN DE LA CORRECTION ---