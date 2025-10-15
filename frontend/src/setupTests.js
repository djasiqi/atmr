// jest-dom adds custom jest matchers for asserting on DOM nodes.
import "@testing-library/jest-dom";

// ✅ Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// ✅ Mock window.location
delete window.location;
window.location = { href: "", reload: jest.fn() };

// ✅ Mock console pour tests propres
global.console = {
  ...console,
  error: jest.fn(),
  warn: jest.fn(),
};

// ✅ Mock Socket.IO pour tests composants
jest.mock("socket.io-client", () => {
  return {
    io: jest.fn(() => ({
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
      connect: jest.fn(),
      disconnect: jest.fn(),
      connected: false,
    })),
  };
});
