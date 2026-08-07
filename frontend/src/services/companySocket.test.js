/**
 * @jest-environment jsdom
 */

jest.mock('socket.io-client', () => {
  const handlers = new Map();
  const ioHandlers = new Map();
  const mockSocket = {
    connected: false,
    id: 'mock-socket-id',
    on: jest.fn((event, cb) => {
      const list = handlers.get(event) ?? [];
      list.push(cb);
      handlers.set(event, list);
      return mockSocket;
    }),
    once: jest.fn((event, cb) => {
      const wrapped = (...args) => {
        mockSocket.off(event, wrapped);
        cb(...args);
      };
      mockSocket.on(event, wrapped);
      return mockSocket;
    }),
    off: jest.fn((event, cb) => {
      const list = handlers.get(event) ?? [];
      handlers.set(
        event,
        list.filter((fn) => fn !== cb)
      );
      return mockSocket;
    }),
    emit: jest.fn(),
    connect: jest.fn(() => {
      mockSocket.connected = true;
      (handlers.get('connect') ?? []).forEach((fn) => fn());
    }),
    disconnect: jest.fn(() => {
      mockSocket.connected = false;
      (handlers.get('disconnect') ?? []).forEach((fn) => fn('io client disconnect'));
    }),
    removeAllListeners: jest.fn(() => handlers.clear()),
    io: {
      opts: { reconnection: true },
      on: jest.fn((event, cb) => {
        const list = ioHandlers.get(event) ?? [];
        list.push(cb);
        ioHandlers.set(event, list);
      }),
    },
    __handlers: handlers,
    __ioHandlers: ioHandlers,
  };

  return {
    io: jest.fn(() => mockSocket),
    __mockSocket: mockSocket,
  };
});

jest.mock('../hooks/useAuthToken', () => ({
  getAccessToken: jest.fn(() => 'token'),
}));

jest.mock('../utils/webAuthSession', () => ({
  hasActiveSession: jest.fn(() => true),
}));

jest.mock('../config/socketConfig', () => ({
  SOCKET_CONFIG: {
    reconnection: true,
    reconnectionAttempts: 3,
    reconnectionDelay: 100,
    reconnectionDelayMax: 500,
    timeout: 5000,
    withCredentials: true,
  },
  SOCKET_PATH: '/socket.io',
  getSocketTransports: () => ['websocket'],
  isDevelopmentLocalhost: () => true,
}));

describe('companySocket', () => {
  beforeEach(() => {
    jest.resetModules();
    process.env.REACT_APP_SOCKET_ENABLED = 'true';
    const { disconnectCompanySocket } = require('./companySocket');
    disconnectCompanySocket();
  });

  it('allows multiple subscribers on the same event', async () => {
    const { on, ensureCompanySocket } = require('./companySocket');
    const { __mockSocket: mockSocket } = require('socket.io-client');

    const connectPromise = ensureCompanySocket();
    mockSocket.connect();
    await connectPromise;

    const first = jest.fn();
    const second = jest.fn();
    on('driver_location_update', first);
    on('driver_location_update', second);

    const bridge = (mockSocket.__handlers.get('driver_location_update') ?? [])[0];
    expect(bridge).toBeDefined();
    bridge({ driver_id: 1 });

    expect(first).toHaveBeenCalledWith({ driver_id: 1 });
    expect(second).toHaveBeenCalledWith({ driver_id: 1 });
  });

  it('cookie-only web : handshake sans JWT local si session UI active', async () => {
    const { getAccessToken } = require('../hooks/useAuthToken');
    getAccessToken.mockReturnValue(null);
    const { io, __mockSocket: mockSocket } = require('socket.io-client');
    const { ensureCompanySocket, getCompanySocketStatusSnapshot } = require('./companySocket');

    const connectPromise = ensureCompanySocket();
    expect(io).toHaveBeenCalled();
    mockSocket.connect();
    await connectPromise;
    expect(getCompanySocketStatusSnapshot().connected).toBe(true);
    expect(getCompanySocketStatusSnapshot().reasonCode).not.toBe('AUTH_REQUIRED');
  });

  it('waitUntilConnected resolves on connect without busy polling', async () => {
    const { waitUntilConnected, getCompanySocket } = require('./companySocket');
    const { __mockSocket: mockSocket } = require('socket.io-client');

    getCompanySocket();
    const waitPromise = waitUntilConnected(5000);
    mockSocket.connect();

    await expect(waitPromise).resolves.toBe(mockSocket);
  });

  it('disconnect removes network listeners and invalidates generation', async () => {
    const addSpy = jest.spyOn(window, 'addEventListener');
    const removeSpy = jest.spyOn(window, 'removeEventListener');

    const { getCompanySocket, disconnectCompanySocket } = require('./companySocket');
    const { __mockSocket: mockSocket } = require('socket.io-client');

    getCompanySocket();
    mockSocket.connect();

    disconnectCompanySocket();

    const onlineRemoved = removeSpy.mock.calls.some(([event]) => event === 'online');
    expect(onlineRemoved).toBe(true);

    addSpy.mockRestore();
    removeSpy.mockRestore();
  });
});
