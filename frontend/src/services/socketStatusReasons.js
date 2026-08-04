/**
 * Libellés FR pour les causes de déconnexion / échec Socket.IO (company).
 */

const AUTH_LABELS = {
  AUTH_REQUIRED: 'Session absente — reconnectez-vous',
  AUTH_INVALID: 'Session invalide — reconnectez-vous',
  TOKEN_EXPIRED: 'Session expirée — reconnectez-vous',
  AUTH_FORBIDDEN: 'Accès temps réel refusé pour ce compte',
  COMPANY_NOT_FOUND: 'Entreprise introuvable pour la connexion temps réel',
  DRIVER_OR_COMPANY_NOT_FOUND: 'Compte ou entreprise introuvable',
  RATE_LIMIT: 'Trop de tentatives — réessayez dans un instant',
};

const DISCONNECT_LABELS = {
  'io server disconnect': 'Déconnecté par le serveur',
  'io client disconnect': 'Connexion fermée localement',
  'ping timeout': 'Délai d’attente dépassé (réseau lent)',
  'transport close': 'Liaison réseau interrompue',
  'transport error': 'Erreur de transport réseau',
  'parse error': 'Erreur de protocole Socket.IO',
  unauthorized: 'Non autorisé par le serveur',
};

const TRANSPORT_HINTS = [
  { test: /xhr poll error|ECONNREFUSED|ECONNRESET|ENOTFOUND/i, label: 'Serveur temps réel injoignable' },
  { test: /websocket error|websocket/i, label: 'Échec de la connexion WebSocket' },
  { test: /timeout/i, label: 'Délai de connexion dépassé' },
  { test: /transport close/i, label: 'Liaison réseau interrompue' },
];

export function labelForAuthCode(code) {
  if (!code) return null;
  const upper = String(code);
  for (const key of Object.keys(AUTH_LABELS)) {
    if (upper.includes(key)) return { reasonCode: key, reasonLabel: AUTH_LABELS[key] };
  }
  return null;
}

export function labelForDisconnectReason(reason) {
  const key = String(reason || '').trim();
  if (!key) {
    return { reasonCode: 'DISCONNECTED', reasonLabel: 'Connexion temps réel coupée' };
  }
  if (DISCONNECT_LABELS[key]) {
    return { reasonCode: key, reasonLabel: DISCONNECT_LABELS[key] };
  }
  return { reasonCode: key, reasonLabel: `Déconnecté (${key})` };
}

export function labelForConnectError(message) {
  const msg = String(message || '').trim();
  if (!msg) {
    return { reasonCode: 'CONNECT_ERROR', reasonLabel: 'Échec de connexion au temps réel' };
  }
  const auth = labelForAuthCode(msg);
  if (auth) return auth;
  for (const hint of TRANSPORT_HINTS) {
    if (hint.test.test(msg)) {
      return { reasonCode: 'TRANSPORT_ERROR', reasonLabel: hint.label };
    }
  }
  // Message technique trop long → résumé court
  const short = msg.length > 80 ? `${msg.slice(0, 77)}…` : msg;
  return { reasonCode: 'CONNECT_ERROR', reasonLabel: short };
}

export function labelForOffline() {
  return { reasonCode: 'OFFLINE', reasonLabel: 'Hors ligne — pas de réseau' };
}

export function labelForDisabled() {
  return {
    reasonCode: 'SOCKET_DISABLED',
    reasonLabel: 'Temps réel désactivé (configuration)',
  };
}

export function labelForMissingToken() {
  return {
    reasonCode: 'AUTH_REQUIRED',
    reasonLabel: 'Session absente — reconnectez-vous pour activer le temps réel',
  };
}

export function labelForReconnectFailed() {
  return {
    reasonCode: 'RECONNECT_FAILED',
    reasonLabel: 'Reconnexion impossible après plusieurs essais',
  };
}
