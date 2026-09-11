/**
 * Trace le pipeline clic → create → send (durée depuis le clic).
 */

function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export function createInstitutionSubmitTrace() {
  const startedAt = nowMs();
  return {
    mark(stage, extra = {}) {
      const payload = {
        stage,
        duration_ms: Math.round(nowMs() - startedAt),
        ...extra,
      };
      console.info('[institution_submit]', payload);
      return payload;
    },
  };
}
