/**
 * Affichage dossiers facturation — statuts/actions SSOT backend.
 */

export const OPERATIONAL_STATUS_LABELS = {
  A_CALCULER: 'À calculer',
  A_CONTROLER: 'À contrôler',
  PRETE_A_CLOTURER: 'Prête à clôturer',
  PRETE_A_EMETTRE: 'Prête à émettre',
  A_ENVOYER: 'À envoyer',
  A_ENCAISSER: 'À encaisser',
  PARTIALLY_PAID: 'Partiellement payée',
  OVERDUE: 'En retard',
  PAID: 'Payée',
  CANCELLED: 'Annulée',
  CREDITED: 'Créditée',
};

export const ACTION_LABELS = {
  VIEW: 'Consulter',
  RECALCULATE_DOSSIER: 'Calculer',
  REVIEW: 'Contrôler',
  ISSUE: 'Émettre',
  MARK_SENT: 'Marquer comme envoyée',
  RECORD_PAYMENT: 'Enregistrer un paiement',
  DOWNLOAD_PDF: 'Télécharger le PDF',
  CHANGE_DUE_DATE: 'Modifier l’échéance',
  CANCEL: 'Annuler avant envoi',
  CREDIT: 'Créer un avoir',
  VIEW_CREDIT_NOTE: 'Voir l’avoir',
  VIEW_PAYMENTS: 'Voir les paiements',
  SEND_REMINDER: 'Envoyer un rappel',
  REVERSE_PAYMENT: 'Contre-passer un paiement',
  EDIT_INVOICE: 'Éditer la facture',
  CORRECT_INVOICE: 'Corriger la facture',
};

/** Groupes menu ••• (ordre d’affichage). */
export const ACTION_GROUPS = [
  {
    id: 'FACTURE',
    label: 'Facture',
    actions: [
      'VIEW',
      'DOWNLOAD_PDF',
      'MARK_SENT',
      'EDIT_INVOICE',
      'VIEW_CREDIT_NOTE',
      'REVIEW',
      'ISSUE',
    ],
  },
  {
    id: 'PAIEMENT',
    label: 'Paiement',
    actions: ['RECORD_PAYMENT', 'VIEW_PAYMENTS', 'REVERSE_PAYMENT'],
  },
  {
    id: 'CORRECTION',
    label: 'Correction',
    actions: ['CHANGE_DUE_DATE', 'CORRECT_INVOICE', 'CREDIT', 'RECALCULATE_DOSSIER'],
  },
  {
    id: 'SUIVI',
    label: 'Suivi',
    actions: ['SEND_REMINDER'],
  },
  {
    id: 'EXCEPTION',
    label: 'Action exceptionnelle',
    actions: ['CANCEL'],
  },
];

export const operationalBadgeClass = (status, styleMap) => {
  switch (status) {
    case 'PAID':
      return styleMap.badgePaid;
    case 'A_ENVOYER':
    case 'PRETE_A_EMETTRE':
      return styleMap.badgeSent;
    case 'PARTIALLY_PAID':
      return styleMap.badgePartiallyPaid;
    case 'OVERDUE':
      return styleMap.badgeOverdue;
    case 'CANCELLED':
      return styleMap.badgeCancelled;
    case 'CREDITED':
      return styleMap.badgeCredited;
    case 'A_ENCAISSER':
      return styleMap.badgeIssued;
    case 'PRETE_A_CLOTURER':
    case 'A_CONTROLER':
      return styleMap.badgePartiallyPaid;
    default:
      return styleMap.badgeIssued;
  }
};

/** Actions exécutables depuis le menu ••• du registre (sans ouvrir le drawer). */
export const ROW_MENU_ACTIONS = new Set([
  'DOWNLOAD_PDF',
  'RECALCULATE_DOSSIER',
  'ISSUE',
  'MARK_SENT',
]);

export const groupAllowedActions = (
  allowedActions = [],
  primaryAction = null,
  { rowMenuOnly = false } = {}
) => {
  const set = new Set(allowedActions);
  return ACTION_GROUPS.map((g) => ({
    ...g,
    items: g.actions.filter(
      (a) =>
        set.has(a) &&
        a !== primaryAction &&
        (!rowMenuOnly || ROW_MENU_ACTIONS.has(a))
    ),
  })).filter((g) => g.items.length > 0);
};

export { fmtMoney, fmtDate } from './issuedInvoiceUi';
