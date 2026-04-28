const STATUS_CONFIG = {
  awaiting_client_payment: {
    tone: 'warning',
    icon: 'credit-card',
    labels: {
      client: 'Paiement requis',
      company: 'Paiement client requis',
      driver: 'Paiement en attente',
      institution: 'Paiement en attente',
    },
  },
  pending: {
    tone: 'warning',
    icon: 'hourglass',
    labels: {
      client: 'En attente de confirmation',
      company: 'Demande recue',
      driver: 'Mission proposee',
      institution: 'Demande en traitement',
    },
  },
  assigned: {
    tone: 'info',
    icon: 'user-check',
    labels: {
      client: 'Chauffeur trouve',
      company: 'Mission affectee',
      driver: 'Mission acceptee',
      institution: 'Transporteur assigne',
    },
  },
  in_progress: {
    tone: 'info',
    icon: 'car',
    labels: {
      client: 'Mission en cours',
      company: 'Course en cours',
      driver: 'Trajet en cours',
      institution: 'Transport en cours',
    },
  },
  completed: {
    tone: 'success',
    icon: 'check-circle',
    labels: {
      client: 'Mission terminee',
      company: 'Course terminee',
      driver: 'Mission terminee',
      institution: 'Transport termine',
    },
  },
  canceled: {
    tone: 'danger',
    icon: 'x-circle',
    labels: {
      client: 'Mission annulee',
      company: 'Reservation annulee',
      driver: 'Mission annulee',
      institution: 'Demande annulee',
    },
  },
  failed: {
    tone: 'danger',
    icon: 'alert',
    labels: {
      client: 'Echec',
      company: 'Echec',
      driver: 'Echec',
      institution: 'Echec',
    },
  },
};

export function getMissionStatusPresentation(rawStatus, role = 'client') {
  const status = String(rawStatus || '').toLowerCase();
  const roleKey = String(role || 'client').toLowerCase();
  const cfg = STATUS_CONFIG[status];
  if (!cfg) {
    return {
      status: status || 'unknown',
      tone: 'neutral',
      icon: 'dot',
      label: 'Statut inconnu',
    };
  }
  return {
    status,
    tone: cfg.tone,
    icon: cfg.icon,
    label: cfg.labels[roleKey] || cfg.labels.client,
  };
}

export function getStatusToneClass(tone, classes) {
  if (tone === 'success') return classes.statusCompleted;
  if (tone === 'info') return classes.statusInProgress;
  if (tone === 'danger') return classes.statusCanceled;
  return classes.statusDefault;
}
