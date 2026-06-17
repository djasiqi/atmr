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
  // options.cacheBust : évite un JSON / pdf_url obsolètes (cache navigateur ou intermédiaire) après régénération PDF
  async getInvoice(companyId, invoiceId, options = {}) {
    const params = new URLSearchParams();
    if (options.cacheBust) {
      params.set('_cb', String(Date.now()));
    }
    const qs = params.toString() ? `?${params.toString()}` : '';
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}${qs}`
    );
    return response.data;
  },

  // Marquer une facture comme envoyée (legacy - utiliser sendInvoiceByEmail ou markInvoiceAsSent)
  async sendInvoice(companyId, invoiceId) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/send`,
      { send_method: 'paper' } // Par défaut, marquer comme envoyée (papier)
    );
    return response.data;
  },

  // Envoyer une facture par email
  async sendInvoiceByEmail(companyId, invoiceId, options = {}) {
    const payload = {
      send_method: 'email',
      recipient_email: options.recipient_email || null,
      force_regenerate_pdf: options.force_regenerate_pdf || false,
    };
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/send`,
      payload
    );
    return response.data;
  },

  // Marquer une facture comme envoyée (papier) sans envoyer d'email
  async markInvoiceAsSent(companyId, invoiceId) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/send`,
      { send_method: 'paper' }
    );
    return response.data;
  },

  // Marquer plusieurs factures brouillon comme envoyées en une seule opération
  async bulkMarkAsSent(companyId, invoiceIds, sendMethod = 'paper') {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/bulk-send`,
      { invoice_ids: invoiceIds, send_method: sendMethod }
    );
    return response.data;
  },

  // Envoyer un rappel par email
  async sendReminderByEmail(companyId, invoiceId, reminderId, options = {}) {
    const payload = {
      recipient_email: options.recipient_email || null,
      force_regenerate_pdf: options.force_regenerate_pdf || false,
    };
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/reminders/${reminderId}/send`,
      payload
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

  // Exporter les paiements encaissés en CSV (comptabilité)
  async exportPaymentsCSV(companyId, options = {}) {
    const { year, month, decimal = 'comma', with_meta = 0, companyName } = options;

    if (!year || !month) {
      throw new Error('Année et mois sont requis');
    }

    const params = new URLSearchParams();
    params.append('year', year);
    params.append('month', month);
    params.append('decimal', decimal);
    if (with_meta) {
      params.append('with_meta', with_meta);
    }

    try {
      const response = await apiClient.get(
        `${API_BASE}/invoices/companies/${companyId}/exports/payments.csv?${params}`,
        {
          responseType: 'blob',
        }
      );

      // Vérifier si le blob est vide (pas de données)
      if (response.data.size === 0) {
        const error = new Error('Aucun paiement enregistré sur cette période');
        error.response = { status: 404, data: { size: 0 } };
        throw error;
      }

      // Créer un lien de téléchargement
      const blob = new Blob([response.data], { type: 'text/csv; charset=utf-8' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      // Nom de fichier : encaissements_YYYY-MM.csv ou encaissements_NOM_ENTREPRISE_YYYY-MM.csv
      const monthStr = String(month).padStart(2, '0');
      let filename = `encaissements_${year}-${monthStr}.csv`;
      if (companyName) {
        // Sanitiser le nom de l'entreprise (enlever caractères spéciaux)
        const sanitized = companyName
          .replace(/[^a-zA-Z0-9\s-_]/g, '')
          .replace(/\s+/g, '_')
          .substring(0, 30); // Limiter la longueur
        filename = `encaissements_${sanitized}_${year}-${monthStr}.csv`;
      }
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      // Propager les erreurs HTTP avec le statut pour une gestion spécifique
      if (err.response) {
        throw err;
      }
      // Erreur réseau ou autre
      throw new Error(err.message || "Erreur lors de l'export du CSV");
    }
  },

  // NOUVEAU: Récupérer la liste des institutions (cliniques)
  async fetchInstitutions(companyId) {
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clients/institutions`
    );
    return response.data;
  },

  // Clients éligibles (trajets non facturés)
  async fetchEligibleClients(
    companyId,
    { search, limit, year, month, bill_to_client_id, clinic_company_id, billed_to_type } = {}
  ) {
    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (limit) params.append('limit', limit);
    if (year) params.append('year', year);
    if (month) params.append('month', month);
    if (bill_to_client_id) params.append('bill_to_client_id', bill_to_client_id);
    if (clinic_company_id) params.append('clinic_company_id', clinic_company_id);
    if (billed_to_type) params.append('billed_to_type', billed_to_type);
    const query = params.toString();
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clients/eligible${query ? `?${query}` : ''}`
    );
    return response.data;
  },

  // Totaux pour facture clinique mensuelle (S2)
  async fetchClinicMonthlyTotals(
    companyId,
    { year, month, clinic_company_id, include_client_ids } = {}
  ) {
    const params = new URLSearchParams();
    if (year) params.append('year', year);
    if (month) params.append('month', month);
    if (clinic_company_id) params.append('clinic_company_id', clinic_company_id);
    if (include_client_ids && include_client_ids.length > 0) {
      params.append('include_client_ids', include_client_ids.join(','));
    }
    const query = params.toString();
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clinic-monthly-totals${query ? `?${query}` : ''}`
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

  // Récupérer les partenaires facturables (optionnellement filtrés par période)
  async fetchBillablePartners(companyId, filters = {}) {
    const params = new URLSearchParams();
    if (filters.year) params.append('year', filters.year);
    if (filters.month) params.append('month', filters.month);
    const query = params.toString();
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/partners/billable${query ? `?${query}` : ''}`
    );
    return response.data;
  },

  // Récupérer les transferts facturables pour un partenariat + période
  async fetchPartnerTransfers(companyId, partnershipId, { year, month }) {
    const params = new URLSearchParams();
    if (year) params.append('year', year);
    if (month) params.append('month', month);
    const query = params.toString();
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/partners/${partnershipId}/transfers?${query}`
    );
    return response.data;
  },

  // Générer une facture partenaire
  async generatePartnerInvoice(companyId, data) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/partners/invoices/generate`,
      data
    );
    return response.data;
  },

  // NOUVEAU: Récupérer les réservations non encore facturées d'un client
  // Si include_invoiced=1 et clinic_company_id fournis (contexte S2), inclut aussi les transports déjà facturés
  async fetchUnbilledReservations(companyId, clientId, filters = {}) {
    const params = new URLSearchParams();

    if (filters.year) params.append('year', filters.year);
    if (filters.month) params.append('month', filters.month);
    if (filters.billed_to_type) params.append('billed_to_type', filters.billed_to_type);
    if (filters.include_invoiced) params.append('include_invoiced', '1');
    if (filters.clinic_company_id) params.append('clinic_company_id', filters.clinic_company_id);

    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clients/${clientId}/unbilled-reservations?${params}`
    );
    return response.data;
  },

  async fetchUnbilledReservationIds(companyId, clientId, filters = {}) {
    const params = new URLSearchParams();

    if (filters.year) params.append('year', filters.year);
    if (filters.month) params.append('month', filters.month);
    if (filters.billed_to_type) params.append('billed_to_type', filters.billed_to_type);

    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/clients/${clientId}/unbilled-reservations/ids?${params}`
    );
    return response.data;
  },

  // Récupérer une réservation spécifique par ID (pour hydratation d'objets minimaux)
  async fetchReservationById(companyId, reservationId) {
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/reservations/${reservationId}`
    );
    return response.data;
  },

  /** Prévisualisation V1 (payeur + mois) — GET period-preview */
  async fetchPeriodPreview(companyId, { year, month, clientId, clinicCompanyId }) {
    const params = new URLSearchParams();
    params.append('year', year);
    params.append('month', month);
    if (clientId) params.append('client_id', clientId);
    if (clinicCompanyId) params.append('clinic_company_id', clinicCompanyId);
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/invoices/period-preview?${params}`
    );
    return response.data;
  },

  /** V2 : agrégat des payeurs avec courses à facturer sur la période (dashboard) */
  async fetchBillingOpportunities(companyId, year, month) {
    const params = new URLSearchParams();
    params.append('year', year);
    params.append('month', month);
    const response = await apiClient.get(
      `${API_BASE}/invoices/companies/${companyId}/invoices/billing-opportunities?${params}`
    );
    return response.data;
  },

  async removeDraftInvoiceLine(companyId, invoiceId, lineId, options = {}) {
    const url = `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/lines/${lineId}`;
    const cfg =
      options.expected_updated_at != null && String(options.expected_updated_at).trim()
        ? { data: { expected_updated_at: options.expected_updated_at } }
        : {};
    const response = await apiClient.delete(url, cfg);
    return response.data;
  },

  async updateDraftInvoiceLine(companyId, invoiceId, lineId, body) {
    const response = await apiClient.patch(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/lines/${lineId}`,
      body
    );
    return response.data;
  },

  async applyDraftGlobalDiscount(companyId, invoiceId, payload) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/apply-global-discount`,
      payload
    );
    return response.data;
  },

  /** Remises % par ligne transport (pourcentages distincts). Liste vide = retirer toutes les remises % auto. */
  async applyDraftPerLineDiscounts(companyId, invoiceId, payload) {
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/apply-per-line-discounts`,
      payload
    );
    return response.data;
  },

  async removeDraftGlobalDiscount(companyId, invoiceId, options = {}) {
    const body = {};
    if (options.expected_updated_at != null && String(options.expected_updated_at).trim()) {
      body.expected_updated_at = options.expected_updated_at;
    }
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/remove-global-discount`,
      Object.keys(body).length ? body : {}
    );
    return response.data;
  },

  async addDraftCustomLine(
    companyId,
    invoiceId,
    {
      description,
      line_total,
      qty = 1,
      custom_mode,
      time_unit,
      expected_updated_at,
      service_date_iso,
    }
  ) {
    const mode = custom_mode === 'quantity' ? 'quantity' : 'time';
    const unitOk = (u) =>
      u != null && ['min', 'h', 'd', 'mois'].includes(String(u).trim());
    const body = {
      description,
      line_total,
      qty,
      custom_mode: mode,
    };
    if (mode === 'time') {
      body.time_unit = unitOk(time_unit) ? String(time_unit).trim() : 'h';
    }
    if (service_date_iso != null && String(service_date_iso).trim()) {
      body.service_date_iso = String(service_date_iso).trim();
    }
    if (expected_updated_at != null && String(expected_updated_at).trim()) {
      body.expected_updated_at = expected_updated_at;
    }
    const response = await apiClient.post(
      `${API_BASE}/invoices/companies/${companyId}/invoices/${invoiceId}/custom-line`,
      body
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

/** Format montant CHF pour S2 override dialog/footer : "40.00 CHF". Utilise toFixed(2). */
export const formatCurrencyCHF = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return '0.00 CHF';
  return `${n.toFixed(2)} CHF`;
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

/** Statut facture normalisé (liste + détail API peuvent varier). */
export const invoiceStatusLower = (invoice) => {
  const s = invoice?.status;
  return typeof s === 'string' ? s.toLowerCase() : '';
};

/** Édition lignes / remises : brouillon ou facture émise tant qu’elle n’est pas payée / annulée. */
export const canEditDraft = (invoice) => {
  const s = invoiceStatusLower(invoice);
  return ['draft', 'sent', 'partially_paid', 'overdue'].includes(s);
};

export const canSendInvoice = (invoice) => {
  return invoiceStatusLower(invoice) === 'draft';
};

export const canAddPayment = (invoice) => {
  return !['paid', 'cancelled'].includes(invoiceStatusLower(invoice));
};

export const canGenerateReminder = (invoice) => {
  return (
    !['paid', 'cancelled'].includes(invoiceStatusLower(invoice)) && invoice.balance_due > 0
  );
};

export const canRegeneratePdf = (invoice) => {
  // Les factures payées ne peuvent plus être régénérées, seule la visualisation est possible
  const s = invoiceStatusLower(invoice);
  return s !== 'cancelled' && s !== 'paid';
};

export const canCancelInvoice = (invoice) => {
  return invoiceStatusLower(invoice) === 'draft' && invoice.amount_paid === 0;
};

export const canDuplicateInvoice = (invoice) => {
  // Les factures payées ne peuvent plus avoir de correctif
  const s = invoiceStatusLower(invoice);
  return ['sent', 'partially_paid', 'overdue'].includes(s) || s === 'cancelled';
};

export const duplicateInvoice = async (companyId, invoiceId) => {
  return invoiceService.duplicateInvoice(companyId, invoiceId);
};

export const getNextReminderLevel = (invoice) => {
  return Math.min(invoice.reminder_level + 1, 3);
};

export const getEffectiveDueDate = (invoice) => {
  if (invoice?.effective_due_date) {
    return invoice.effective_due_date;
  }
  const openReminders = (invoice?.reminders || []).filter(
    (r) => (r.status || 'OPEN') === 'OPEN'
  );
  if (openReminders.length > 0) {
    const latest = [...openReminders].sort(
      (a, b) =>
        new Date(b.generated_at || 0) - new Date(a.generated_at || 0) ||
        (b.level || 0) - (a.level || 0)
    )[0];
    if (latest?.due_date) {
      return latest.due_date;
    }
  }
  return invoice?.due_date;
};

export const isOverdue = (invoice) => {
  const status = invoiceStatusLower(invoice);
  if (status === 'paid' || status === 'cancelled' || status === 'draft') return false;
  if ((invoice?.balance_due ?? 0) <= 0) return false;
  const dueDate = getEffectiveDueDate(invoice);
  return Boolean(dueDate && new Date(dueDate) < new Date());
};

export const getDaysOverdue = (invoice) => {
  if (!isOverdue(invoice)) return 0;
  const dueDate = new Date(getEffectiveDueDate(invoice));
  const now = new Date();
  return Math.floor((now - dueDate) / (1000 * 60 * 60 * 24));
};

// Export des fonctions principales pour compatibilité
export const {
  fetchInvoices,
  getInvoice,
  generateInvoice,
  sendInvoice,
  sendInvoiceByEmail,
  markInvoiceAsSent,
  bulkMarkAsSent,
  sendReminderByEmail,
  postPayment,
  postReminder,
  regenerateInvoicePdf,
  cancelInvoice,
  fetchBillingSettings,
  updateBillingSettings,
  exportInvoicesCSV,
  exportPaymentsCSV,
  fetchPeriodPreview,
  fetchBillingOpportunities,
  removeDraftInvoiceLine,
  updateDraftInvoiceLine,
  applyDraftGlobalDiscount,
  applyDraftPerLineDiscounts,
  removeDraftGlobalDiscount,
  addDraftCustomLine,
} = invoiceService;
