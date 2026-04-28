const mockOnMap = new Map<string, ((payload?: unknown) => void)[]>();
const mockEmit = jest.fn();
const mockOn = jest.fn((event: string, cb: (payload?: unknown) => void) => {
  const list = mockOnMap.get(event) ?? [];
  list.push(cb);
  mockOnMap.set(event, list);
});
const mockOff = jest.fn((event: string, cb: (payload?: unknown) => void) => {
  const list = mockOnMap.get(event) ?? [];
  mockOnMap.set(event, list.filter((fn) => fn !== cb));
});
const mockOnce = jest.fn((event: string, cb: () => void) => mockOn(event, cb));
const mockDisconnect = jest.fn();

jest.mock('socket.io-client', () => ({
  io: jest.fn(() => ({
    connected: true,
    emit: mockEmit,
    on: mockOn,
    off: mockOff,
    once: mockOnce,
    disconnect: mockDisconnect,
  })),
}));

describe('institutionRealtimeBridge', () => {
  beforeEach(() => {
    jest.resetModules();
    mockOnMap.clear();
    mockEmit.mockClear();
    mockOn.mockClear();
    mockOff.mockClear();
    mockOnce.mockClear();
    mockDisconnect.mockClear();
  });

  it('joins institution room and subscribes/unsubscribes events', () => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require('../institutionRealtimeBridge') as typeof import('../institutionRealtimeBridge');

    mod.joinInstitutionRealtime(77);
    expect(mockEmit).toHaveBeenCalledWith('join_institution', { institution_id: 77 });

    const cb = jest.fn();
    const unsubscribe = mod.subscribeInstitutionEvents(cb);

    const eventListener = (mockOnMap.get('request_sent') ?? [])[0];
    expect(eventListener).toBeDefined();
    eventListener?.({ id: 1 });
    expect(cb).toHaveBeenCalledWith('request_sent', { id: 1 });

    unsubscribe();
    const listenersAfter = mockOnMap.get('request_sent') ?? [];
    expect(listenersAfter).toHaveLength(0);
  });

  it('disconnects cleanly', () => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require('../institutionRealtimeBridge') as typeof import('../institutionRealtimeBridge');
    mod.ensureInstitutionSocket();
    mod.disconnectInstitutionRealtime();
    expect(mockDisconnect).toHaveBeenCalled();
  });
});
