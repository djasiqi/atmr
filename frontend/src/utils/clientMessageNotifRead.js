export const CLIENT_MESSAGE_READ_EVENT = 'lirie-client-message-read';
const READ_STORAGE_KEY = 'lirie.clientMessageNotifRead';

export function clientNotificationReadKey(id) {
  if (typeof id === 'string' && id.startsWith('invoice-')) return id;
  const numeric = Number(id);
  return Number.isFinite(numeric) ? numeric : String(id);
}

export function readClientMessageIds() {
  try {
    const raw = JSON.parse(window.localStorage.getItem(READ_STORAGE_KEY) || '[]');
    return new Set(Array.isArray(raw) ? raw.map(clientNotificationReadKey) : []);
  } catch {
    return new Set();
  }
}

export function rememberClientMessageRead(ids) {
  const next = readClientMessageIds();
  ids.forEach((id) => next.add(clientNotificationReadKey(id)));
  window.localStorage.setItem(READ_STORAGE_KEY, JSON.stringify([...next]));
  window.dispatchEvent(new Event(CLIENT_MESSAGE_READ_EVENT));
  return next;
}

export function bookingHasUnreadCarrierMessage(booking, notifications, readIds) {
  const ids = new Set(
    [booking?.id, ...(booking?.route_request_ids || [])]
      .map(Number)
      .filter((id) => Number.isFinite(id) && id > 0)
  );
  return (notifications || []).some(
    (item) => ids.has(Number(item?.booking_id)) && !readIds.has(Number(item?.id))
  );
}
