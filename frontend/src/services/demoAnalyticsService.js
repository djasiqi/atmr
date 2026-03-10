const DEMO_EVENTS = new Set([
  'demo_session_start',
  'demo_step_reached',
  'demo_completed',
]);

export const trackDemoEvent = (eventName, payload = {}) => {
  if (!DEMO_EVENTS.has(eventName)) return;
  const record = {
    event: eventName,
    payload,
    ts: new Date().toISOString(),
  };
  // Sprint 2: métriques minimales et anonymisées (client-side log).
  // Ce point d'extension permet d'ajouter un endpoint backend ensuite.
  // eslint-disable-next-line no-console
  console.info('[demo-analytics]', record);

  fetch('/api/v1/demo_access/analytics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event: eventName, payload }),
    keepalive: true,
  }).catch(() => {});
};

