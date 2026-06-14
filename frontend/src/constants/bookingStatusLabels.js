/** Libellés FR unifiés pour les statuts de course (espace entreprise). */
export const BOOKING_STATUS_LABELS = {
  pending: 'En attente',
  accepted: 'Acceptée',
  assigned: 'Assignée',
  en_route: 'En route',
  in_progress: 'En cours',
  completed: 'Terminée',
  return_completed: 'Retour terminé',
  canceled: 'Annulée',
  cancelled: 'Annulée',
  rejected: 'Refusée',
  no_show: 'Non présenté',
  awaiting_client_payment: 'Paiement client',
};

export function getBookingStatusLabel(status) {
  const key = String(status || '').toLowerCase();
  return BOOKING_STATUS_LABELS[key] || (status ? String(status).replace(/_/g, ' ') : '—');
}
