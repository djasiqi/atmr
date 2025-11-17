import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './InvoicesRegistry.module.css';
import {
  fetchInvoices,
  sendInvoice,
  postPayment,
  postReminder,
  regenerateInvoicePdf,
  cancelInvoice,
  duplicateInvoice,
} from '../../../../services/invoiceService';
import useCompanyData from '../../../../hooks/useCompanyData';
import Filters from './components/Filters';
import InvoiceRowActions from './components/InvoiceRowActions';
import PaymentModal from './components/PaymentModal';
import ReminderModal from './components/ReminderModal';
import NewInvoiceModal from './components/NewInvoiceModal';

const InvoicesRegistry = () => {
  const { company } = useCompanyData();
  const navigate = useNavigate();
  const [invoices, setInvoices] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pagination, setPagination] = useState({});
  const [stats, setStats] = useState({});
  const [filters, setFilters] = useState({
    status: '',
    client_id: '',
    year: new Date().getFullYear(),
    month: '',
    q: '',
    with_balance: false,
    with_reminders: false,
    page: 1,
    per_page: 20,
  });

  // Modals state
  const [paymentModal, setPaymentModal] = useState({
    open: false,
    invoice: null,
  });
  const [reminderModal, setReminderModal] = useState({
    open: false,
    invoice: null,
  });
  const [newInvoiceModal, setNewInvoiceModal] = useState({ open: false, invoiceDraft: null });

  // Charger les factures
  const loadInvoices = useCallback(async () => {
    if (!company?.id) return;

    try {
      setLoading(true);
      setError(null);

      const response = await fetchInvoices(company.id, filters);
      setInvoices(response.invoices);
      setPagination(response.pagination);
      setStats(response.stats);
    } catch (err) {
      setError(err.message || 'Erreur lors du chargement des factures');
    } finally {
      setLoading(false);
    }
  }, [company?.id, filters]);

  useEffect(() => {
    loadInvoices();
  }, [loadInvoices]);

  // Handlers
  const handleFilterChange = (newFilters) => {
    setFilters((prev) => ({ ...prev, ...newFilters, page: 1 }));
  };

  const handlePageChange = (page) => {
    setFilters((prev) => ({ ...prev, page }));
  };

  const handleSendInvoice = async (invoiceId) => {
    try {
      await sendInvoice(company.id, invoiceId);
      await loadInvoices();
    } catch (err) {
      setError(err.message || "Erreur lors de l'envoi de la facture");
    }
  };

  const handlePayment = async (invoiceId, paymentData) => {
    try {
      await postPayment(company.id, invoiceId, paymentData);
      await loadInvoices();
      setPaymentModal({ open: false, invoice: null });
    } catch (err) {
      setError(err.message || "Erreur lors de l'enregistrement du paiement");
    }
  };

  const handleReminder = async (invoiceId, level) => {
    try {
      await postReminder(company.id, invoiceId, { level });
      await loadInvoices();
      setReminderModal({ open: false, invoice: null });
    } catch (err) {
      setError(err.message || 'Erreur lors de la génération du rappel');
    }
  };

  const handleRegeneratePdf = async (invoiceId) => {
    try {
      await regenerateInvoicePdf(company.id, invoiceId);
      await loadInvoices();
    } catch (err) {
      setError(err.message || 'Erreur lors de la régénération du PDF');
    }
  };

  const handleCancelInvoice = async (invoiceId) => {
    if (!window.confirm('Êtes-vous sûr de vouloir annuler cette facture ?')) return;

    try {
      await cancelInvoice(company.id, invoiceId);
      await loadInvoices();
    } catch (err) {
      setError(err.message || "Erreur lors de l'annulation de la facture");
    }
  };

  const handleDuplicateInvoice = async (invoiceId) => {
    if (
      !window.confirm(
        'Créer un brouillon correctif pour cette facture ? Les courses originales seront libérées.'
      )
    )
      return;

    try {
      const response = await duplicateInvoice(company.id, invoiceId);
      await loadInvoices();

      const draftContext = response?.draft;
      if (draftContext) {
        setNewInvoiceModal({
          open: true,
          invoiceDraft: draftContext,
        });
      } else {
        window.alert(
          'Les transports ont été libérés. Vous pouvez créer un nouveau brouillon manuellement.'
        );
      }
    } catch (err) {
      setError(err.message || 'Erreur lors de la duplication de la facture');
    }
  };

  const handleOpenSettings = () => {
    if (company?.public_id) {
      navigate(`/dashboard/company/${company.public_id}/settings#billing`);
    }
  };

  const handleNewInvoiceGenerated = (invoice) => {
    // Recharger la liste des factures
    loadInvoices();
    // Optionnel: afficher un message de succès
    // eslint-disable-next-line no-console
    console.log('Nouvelle facture générée:', invoice);
  };

  // Formatage des statuts
  const getStatusBadge = (status) => {
    const statusConfig = {
      draft: { label: 'Brouillon', className: styles.badgeDraft },
      sent: { label: 'Envoyée', className: styles.badgeSent },
      partially_paid: {
        label: 'Partiellement payée',
        className: styles.badgePartiallyPaid,
      },
      paid: { label: 'Payée', className: styles.badgePaid },
      overdue: { label: 'En retard', className: styles.badgeOverdue },
      cancelled: { label: 'Annulée', className: styles.badgeCancelled },
    };

    const config = statusConfig[status] || {
      label: status,
      className: styles.badgeDefault,
    };
    return <span className={`${styles.badge} ${config.className}`}>{config.label}</span>;
  };

  const getReminderBadge = (level) => {
    if (level === 0) return null;

    const config = {
      1: { label: '1er rappel', className: styles.reminder1 },
      2: { label: '2e rappel', className: styles.reminder2 },
      3: { label: 'Dernier', className: styles.reminder3 },
    };

    const reminderConfig = config[level] || {
      label: `Rappel ${level}`,
      className: styles.reminderDefault,
    };
    return (
      <span className={`${styles.reminderBadge} ${reminderConfig.className}`}>
        {reminderConfig.label}
      </span>
    );
  };

  return (
    <>
      {/* Section Header + Filtres */}
      <section className={styles.headerSection}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <h1>📄 Suivi des factures</h1>
            <p className={styles.subtitle}>
              Gestion complète de la facturation et suivi des paiements
            </p>
          </div>
          <div className={styles.headerActions}>
            <button
              className={styles.settingsBtn}
              onClick={handleOpenSettings}
            >
              ⚙️ Paramètres
            </button>
            <button
              className={styles.newInvoiceBtn}
              onClick={() => setNewInvoiceModal({ open: true, invoiceDraft: null })}
            >
              ➕ Nouvelle facture
            </button>
          </div>
        </div>

        {/* Filtres dans le même conteneur */}
        <Filters filters={filters} onFilterChange={handleFilterChange} companyId={company?.id} />
      </section>

      {/* Statistiques KPI */}
      <div className={styles.stats}>
        <div className={styles.statCard}>
          <span className={styles.statIcon}>📄</span>
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>Total émis</h3>
            <p className={styles.statValue}>{stats.total_issued?.toFixed(2) || '0.00'} CHF</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statIcon}>✅</span>
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>Payé</h3>
            <p className={styles.statValue}>{stats.total_paid?.toFixed(2) || '0.00'} CHF</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statIcon}>💰</span>
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>Solde</h3>
            <p className={styles.statValue}>{stats.total_balance?.toFixed(2) || '0.00'} CHF</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statIcon}>⚠️</span>
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>En retard</h3>
            <p className={styles.statValue}>{stats.overdue_count || 0}</p>
          </div>
        </div>
      </div>

      {/* Messages d'erreur */}
      {error && (
        <div className={styles.error}>
          {error}
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* Tableau des factures */}
      <div className={styles.tableContainer}>
        {loading ? (
          <div className={styles.loading}>Chargement...</div>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>N° facture</th>
                <th>Client</th>
                <th>Période</th>
                <th>Émise le</th>
                <th>Échéance</th>
                <th>Montant</th>
                <th>Payé</th>
                <th>Solde</th>
                <th>Statut</th>
                <th>Rappel</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {invoices.map((invoice) => (
                <tr key={invoice.id}>
                  <td>{invoice.invoice_number}</td>
                  <td>
                    {/* Afficher l'institution si facturation tierce, sinon le client */}
                    {invoice.bill_to_client_id &&
                    invoice.bill_to_client_id !== invoice.client_id &&
                    invoice.bill_to_client
                      ? invoice.bill_to_client.institution_name ||
                        `${invoice.bill_to_client.first_name || ''} ${
                          invoice.bill_to_client.last_name || ''
                        }`.trim()
                      : invoice.client
                      ? `${invoice.client.first_name || ''} ${
                          invoice.client.last_name || ''
                        }`.trim() || invoice.client.username
                      : 'Client inconnu'}
                  </td>
                  <td>
                    {invoice.period_month.toString().padStart(2, '0')}.{invoice.period_year}
                  </td>
                  <td>{new Date(invoice.issued_at).toLocaleDateString('fr-FR')}</td>
                  <td>{new Date(invoice.due_date).toLocaleDateString('fr-FR')}</td>
                  <td>{invoice.total_amount.toFixed(2)} CHF</td>
                  <td>{invoice.amount_paid.toFixed(2)} CHF</td>
                  <td>{invoice.balance_due.toFixed(2)} CHF</td>
                  <td>{getStatusBadge(invoice.status)}</td>
                  <td>{getReminderBadge(invoice.reminder_level)}</td>
                  <td>
                    <InvoiceRowActions
                      invoice={invoice}
                      onSend={() => handleSendInvoice(invoice.id)}
                      onPayment={() => setPaymentModal({ open: true, invoice })}
                      onReminder={() => setReminderModal({ open: true, invoice })}
                      onRegeneratePdf={() => handleRegeneratePdf(invoice.id)}
                      onCancel={() => handleCancelInvoice(invoice.id)}
                      onDuplicate={() => handleDuplicateInvoice(invoice.id)}
                      onViewPdf={() => window.open(invoice.pdf_url, '_blank')}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {pagination.pages > 1 && (
        <div className={styles.pagination}>
          <button
            disabled={!pagination.has_prev}
            onClick={() => handlePageChange(pagination.page - 1)}
          >
            ← Précédent
          </button>
          <span>
            Page {pagination.page} sur {pagination.pages}
          </span>
          <button
            disabled={!pagination.has_next}
            onClick={() => handlePageChange(pagination.page + 1)}
          >
            Suivant →
          </button>
        </div>
      )}

      {/* Modals */}
      <PaymentModal
        open={paymentModal.open}
        invoice={paymentModal.invoice}
        onClose={() => setPaymentModal({ open: false, invoice: null })}
        onPayment={handlePayment}
      />

      <ReminderModal
        open={reminderModal.open}
        invoice={reminderModal.invoice}
        onClose={() => setReminderModal({ open: false, invoice: null })}
        onReminder={handleReminder}
      />

      <NewInvoiceModal
        open={newInvoiceModal.open}
        initialDraft={newInvoiceModal.invoiceDraft}
        onClose={() => setNewInvoiceModal({ open: false, invoiceDraft: null })}
        onInvoiceGenerated={handleNewInvoiceGenerated}
        companyId={company?.id}
      />
    </>
  );
};

export default InvoicesRegistry;
