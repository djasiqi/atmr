import type { Socket } from 'socket.io-client';
import { io } from 'socket.io-client';

import { apiBaseRoot, getMemoryAccessToken } from '@/services/api';

type InstitutionRealtimeEvent =
  | 'request_sent'
  | 'offer_accepted'
  | 'request_converted'
  | 'booking_status_updated'
  | 'request_cancelled'
  | 'booking_cancelled'
  | 'new_notification';

const EVENT_NAMES: InstitutionRealtimeEvent[] = [
  'request_sent',
  'offer_accepted',
  'request_converted',
  'booking_status_updated',
  'request_cancelled',
  'booking_cancelled',
  'new_notification',
];

let socket: Socket | null = null;
let currentInstitutionId: number | null = null;

function buildSocket(): Socket {
  const token = getMemoryAccessToken();
  return io(apiBaseRoot, {
    path: '/socket.io',
    transports: ['websocket', 'polling'],
    autoConnect: true,
    auth: token ? { token } : {},
    extraHeaders: token ? { Authorization: `Bearer ${token}` } : {},
    reconnection: true,
    reconnectionAttempts: 8,
    reconnectionDelay: 1000,
  });
}

export function ensureInstitutionSocket(): Socket {
  if (!socket) {
    socket = buildSocket();
  }
  return socket;
}

export function joinInstitutionRealtime(institutionId: number): void {
  currentInstitutionId = institutionId;
  const s = ensureInstitutionSocket();
  if (s.connected) {
    s.emit('join_institution', { institution_id: institutionId });
    return;
  }
  s.once('connect', () => {
    s.emit('join_institution', { institution_id: institutionId });
  });
}

export function subscribeInstitutionEvents(
  onEvent: (event: InstitutionRealtimeEvent, payload: unknown) => void,
): () => void {
  const s = ensureInstitutionSocket();
  const unsubs = EVENT_NAMES.map((eventName) => {
    const listener = (payload: unknown) => onEvent(eventName, payload);
    s.on(eventName, listener);
    return () => s.off(eventName, listener);
  });

  const reconnectListener = () => {
    if (currentInstitutionId) {
      s.emit('join_institution', { institution_id: currentInstitutionId });
    }
  };
  s.on('reconnect', reconnectListener);

  return () => {
    unsubs.forEach((off) => off());
    s.off('reconnect', reconnectListener);
  };
}

export function disconnectInstitutionRealtime(): void {
  if (!socket) return;
  socket.disconnect();
  socket = null;
  currentInstitutionId = null;
}
