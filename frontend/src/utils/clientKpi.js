export function trackClientKpiEvent(name, payload = {}) {
  const event = {
    name,
    at: Date.now(),
    ...payload,
  };

  if (typeof window !== 'undefined') {
    if (!window.__LIRIE_CLIENT_KPI__) {
      window.__LIRIE_CLIENT_KPI__ = [];
    }
    window.__LIRIE_CLIENT_KPI__.push(event);
    window.dispatchEvent(
      new CustomEvent('lirie-client-kpi', {
        detail: event,
      })
    );
  }
  return event;
}

