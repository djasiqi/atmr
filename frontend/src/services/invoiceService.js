import apiClient from '../utils/apiClient';

const API_BASE = '';

// Service pour la gestion des factures
export const invoiceService = {
  // Récupérer la liste des factures avec filtres
  async fetchInvoices(companyId, filters = {}) {
    const params = new URLSearchParams();

    Object.entries(filters).forEach(([key, value]) => {
      if (value !== '' && value !== null && value !== undefined) {
        params.append(key, value);
      }
    });

    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/invoices?${params}`
    );
    return response.data;
  },

  // Générer une nouvelle facture
  async generateInvoice(companyId, data) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/generate`,
      data
    );
    return response.data;
  },

  // Récupérer les détails d'une facture
  async getInvoice(companyId, invoiceId) {
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}`
    );
    return response.data;
  },

  // Marquer une facture comme envoyée
  async sendInvoice(companyId, invoiceId) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/send`
    );
    return response.data;
  },

  // Enregistrer un paiement
  async postPayment(companyId, invoiceId, paymentData) {
    // Normaliser le mode de paiement côté frontend pour éviter les erreurs enum
    const methodMap = {
      'Virement bancaire': 'bank_transfer',
      BANK_TRANSFER: 'bank_transfer',
      'bank-transfer': 'bank_transfer',
      'bank transfer': 'bank_transfer',
      Espèces: 'cash',
      especes: 'cash',
      CASH: 'cash',
      Carte: 'card',
      CARD: 'card',
      adjustment: 'adjustment',
      ADJUSTMENT: 'adjustment',
    };
    const rawMethod = paymentData?.method;
    const normalizedMethod = rawMethod
      ? methodMap[rawMethod] ||
        methodMap[String(rawMethod).toLowerCase()] ||
        String(rawMethod).toLowerCase()
      : 'bank_transfer';
    const payload = { ...paymentData, method: normalizedMethod };
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/payments`,
      payload
    );
    return response.data;
  },

  // Générer un rappel
  async postReminder(companyId, invoiceId, reminderData) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/reminders`,
      reminderData
    );
    return response.data;
  },

  // Régénérer le PDF d'une facture
  async regenerateInvoicePdf(companyId, invoiceId) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/regenerate-pdf`
    );
    return response.data;
  },

  // Annuler une facture
  async cancelInvoice(companyId, invoiceId) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/cancel`
    );
    return response.data;
  },

  // Récupérer les paramètres de facturation
  async fetchBillingSettings(companyId) {
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/billing-settings`
    );
    return response.data;
  },

  // Mettre à jour les paramètres de facturation
  async updateBillingSettings(companyId, settingsData) {
    const response = await apiClient.put(
      `${API_BASE}/invoices/companies/${companyId}/billing-settings`,
      settingsData
    );
    return response.data;
  },

  // Exporter les factures en CSV
  async exportInvoicesCSV(companyId, filters = {}) {
    const params = new URLSearchParams();

    Object.entries(filters).forEach(([key, value]) => {
      if (value !== '' && value !== null && value !== undefined) {
        params.append(key, value);
      }
    });

    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/invoices/export?${params}`,
      {
        responseType: 'blob',
      }
    );

    // Créer un lien de téléchargement
    const blob = new Blob([response.data], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `factures_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  },

  // NOUVEAU: Récupérer la liste des institutions (cliniques)
  async fetchInstitutions(companyId) {
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clients/institutions`
    );
    return response.data;
  },

  // Clients éligibles (trajets non facturés)
  async fetchEligibleClients(companyId, { search, limit } = {}) {
    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (limit) params.append('limit', limit);
    const query = params.toString();
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clients/eligible${query ? `?${query}` : ''}`
    );
    return response.data;
  },

  // NOUVEAU: Marquer/démarquer un client comme institution
  async toggleInstitution(companyId, clientId, data) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/clients/${clientId}/toggle-institution`,
      data
    );
    return response.data;
  },

  async duplicateInvoice(companyId, invoiceId) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/duplicate`
    );
    return response.data;
  },

  // NOUVEAU: Générer des factures consolidées (plusieurs patients vers une clinique)
  async generateConsolidatedInvoice(companyId, data) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/generate`,
      data
    );
    return response.data;
  },

  // NOUVEAU: Récupérer les réservations non encore facturées d'un client
  async fetchUnbilledReservations(companyId, clientId, filters = {}) {
    const params = new URLSearchParams();

    if (filters.year) params.append('year', filters.year);
    if (filters.month) params.append('month', filters.month);
    if (filters.billed_to_type) params.append('billed_to_type', filters.billed_to_type);

    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clients/${clientId}/unbilled-reservations?${params}`
    );
    return response.data;
  },
};

// Fonctions utilitaires pour les composants
export const formatCurrency = (amount) => {
  return new Intl.NumberFormat('fr-CH', {
    style: 'currency',
    currency: 'CHF',
  }).format(amount);
};

export const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString('fr-FR');
};

export const formatDateTime = (dateString) => {
  return new Date(dateString).toLocaleString('fr-FR');
};

export const getStatusLabel = (status) => {
  const statusLabels = {
    draft: 'Brouillon',
    sent: 'Envoyée',
    partially_paid: 'Partiellement payée',
    paid: 'Payée',
    overdue: 'En retard',
    cancelled: 'Annulée',
  };
  return statusLabels[status] || status;
};

export const getStatusColor = (status) => {
  const statusColors = {
    draft: '#6c757d',
    sent: '#17a2b8',
    partially_paid: '#ffc107',
    paid: '#28a745',
    overdue: '#dc3545',
    cancelled: '#6c757d',
  };
  return statusColors[status] || '#6c757d';
};

export const getReminderLabel = (level) => {
  if (level === 0) return null;

  const reminderLabels = {
    1: '1er rappel',
    2: '2e rappel',
    3: 'Dernier rappel',
  };
  return reminderLabels[level] || `Rappel ${level}`;
};

export const getReminderColor = (level) => {
  const reminderColors = {
    1: '#ffc107',
    2: '#fd7e14',
    3: '#dc3545',
  };
  return reminderColors[level] || '#6c757d';
};

export const canSendInvoice = (invoice) => {
  return invoice.status === 'draft';
};

export const canAddPayment = (invoice) => {
  return !['paid', 'cancelled'].includes(invoice.status);
};

export const canGenerateReminder = (invoice) => {
  return !['paid', 'cancelled'].includes(invoice.status) && invoice.balance_due > 0;
};

export const canRegeneratePdf = (invoice) => {
  return invoice.status !== 'cancelled';
};

export const canCancelInvoice = (invoice) => {
  return invoice.status === 'draft' && invoice.amount_paid === 0;
};

export const canDuplicateInvoice = (invoice) => {
  return ['sent', 'partially_paid', 'paid', 'overdue'].includes(invoice.status) || invoice.status === 'cancelled';
};

export const duplicateInvoice = async (companyId, invoiceId) => {
  return invoiceService.duplicateInvoice(companyId, invoiceId);
};

export const getNextReminderLevel = (invoice) => {
  return Math.min(invoice.reminder_level + 1, 3);
};

export const isOverdue = (invoice) => {
  return invoice.balance_due > 0 && new Date(invoice.due_date) < new Date();
};

export const getDaysOverdue = (invoice) => {
  if (!isOverdue(invoice)) return 0;
  const dueDate = new Date(invoice.due_date);
  const now = new Date();
  return Math.floor((now - dueDate) / (1000 * 60 * 60 * 24));
};

// Export des fonctions principales pour compatibilité
export const {
  fetchInvoices,
  generateInvoice,
  sendInvoice,
  postPayment,
  postReminder,
  regenerateInvoicePdf,
  cancelInvoice,
  fetchBillingSettings,
  updateBillingSettings,
  exportInvoicesCSV,
} = invoiceService;
