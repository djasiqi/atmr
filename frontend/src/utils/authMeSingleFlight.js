/** Un seul GET /auth/me en vol — StrictMode / listeners ne doivent pas doubler. */

let inflight = null;

export function fetchAuthMeSingleFlight(requestFn) {
  if (typeof requestFn !== 'function') {
    return Promise.reject(new Error('fetchAuthMeSingleFlight: requestFn requis'));
  }
  if (inflight) return inflight;
  inflight = Promise.resolve(requestFn()).finally(() => {
    inflight = null;
  });
  return inflight;
}

export function resetAuthMeSingleFlightForTests() {
  inflight = null;
}
