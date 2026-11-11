/** Libellé de statut du registre. Un solde nul sans paiement n'est pas « Payée ». */

export function registryStatusLabel(invoice) {
  const status = String(invoice?.status || '').toLowerCase();
  const amountPaid = Number(invoice?.amount_paid || 0);
  const balanceDue = Number(invoice?.balance_due || 0);
  if (status === 'paid' && amountPaid <= 0 && balanceDue <= 0) return 'Solde nul';
  const labels = {
    draft: 'Brouillon',
    sent: 'Envoyée',
    partially_paid: 'Partiellement payée',
    paid: 'Payée',
    overdue: 'En retard',
    cancelled: 'Annulée',
  };
  return labels[status] || status;
}
