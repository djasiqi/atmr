import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './InvoicesRegistry.module.css';
import {
  fetchInvoices,
  sendInvoiceByEmail,
  markInvoiceAsSent,
  sendReminderByEmail,
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
import SendEmailModal from './components/SendEmailModal';
import ExportPaymentsModal from './components/ExportPaymentsModal';
import useUrlSearchSync from '../../../../hooks/useUrlSearchSync';

const InvoicesRegistry = () => {
  const { company } = useCompanyData();
  const navigate = useNavigate();
  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();
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
  /** Incrémenté uniquement après annulation de facture; déclenche refetch eligible + S2 dans le modal. */
  const [invoiceDataRefreshTrigger, setInvoiceDataRefreshTrigger] = useState(0);
  const [sendEmailModal, setSendEmailModal] = useState({
    open: false,
    invoice: null,
    isReminder: false,
    reminderId: null,
  });
  const [exportPaymentsModal, setExportPaymentsModal] = useState({
    open: false,
  });

  // Charger les factures
  const loadInvoices = useCallback(async () => {
    if (!company?.id) return;

    try {
      setLoading(true);
      setError(null);

      const response = await fetchInvoices(company.id, filters);
      // ✅ Le backend renvoie {"data": [...], "pagination": {...}, "stats": {...}}
      // Correction: utiliser response.data au lieu de response.invoices
      const invoicesData = response?.data || response?.invoices || [];
      setInvoices(invoicesData);
      setPagination(response?.pagination || {});
      setStats(response?.stats || {});
    } catch (err) {
      setError(err.message || 'Erreur lors du chargement des factures');
    } finally {
      setLoading(false);
    }
  }, [company?.id, filters]);

  useEffect(() => {
    loadInvoices();
  }, [loadInvoices]);

  useEffect(() => {
    if (!initialized) return;
    if (initialSearch && initialSearch !== filters.q) {
      setFilters((prev) => ({ ...prev, q: initialSearch, page: 1 }));
    }
    if (shouldFocus) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      requestAnimationFrame(() => {
        searchInputRef.current?.focus();
      });
      consumeFocus();
    }
  }, [initialized, initialSearch, shouldFocus, consumeFocus, filters.q]);

  // Handlers
  const handleFilterChange = (newFilters) => {
    setFilters((prev) => ({ ...prev, ...newFilters, page: 1 }));
  };

  const handlePageChange = (page) => {
    setFilters((prev) => ({ ...prev, page }));
  };

  // Marquer comme envoyée (papier) sans email
  const handleMarkAsSent = async (invoiceId) => {
    if (!window.confirm('Marquer cette facture comme envoyée par courrier papier ?')) {
      return;
    }
    try {
      await markInvoiceAsSent(company.id, invoiceId);
      await loadInvoices();
    } catch (err) {
      setError(err.message || "Erreur lors de l'envoi de la facture");
    }
  };

  // Ouvrir le modal d'envoi par email
  const handleOpenSendEmail = (invoice) => {
    setSendEmailModal({
      open: true,
      invoice,
      isReminder: false,
      reminderId: null,
    });
  };

  // Ouvrir le modal d'envoi de rappel par email
  const handleOpenSendReminderEmail = (invoice) => {
    // Trouver le dernier rappel
    const latestReminder = invoice.reminders?.[invoice.reminders.length - 1];
    if (!latestReminder) {
      setError('Aucun rappel trouvé pour cette facture');
      return;
    }
    
    setSendEmailModal({
      open: true,
      invoice,
      isReminder: true,
      reminderId: latestReminder.id,
    });
  };

  // Envoyer par email
  const handleSendEmail = async (options) => {
    try {
      if (options.reminder_id) {
        // Envoi d'un rappel
        await sendReminderByEmail(
          company.id,
          sendEmailModal.invoice.id,
          options.reminder_id,
          {
            recipient_email: options.recipient_email,
            force_regenerate_pdf: options.force_regenerate_pdf,
          }
        );
      } else {
        // Envoi d'une facture
        await sendInvoiceByEmail(company.id, sendEmailModal.invoice.id, {
          recipient_email: options.recipient_email,
          force_regenerate_pdf: options.force_regenerate_pdf,
        });
      }
      
      await loadInvoices();
      setSendEmailModal({ open: false, invoice: null, isReminder: false, reminderId: null });
    } catch (err) {
      throw err; // Laisser le modal gérer l'erreur
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
      setInvoiceDataRefreshTrigger((t) => t + 1);
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

  const getReminderBadge = (invoice) => {
    // ✅ NOUVEAU : Utiliser les données du rappel consolidé si disponibles
    if (invoice.reminders && invoice.reminders.length > 0) {
      // Trouver le rappel le plus récent
      const latestReminder = invoice.reminders
        .sort((a, b) => new Date(b.generated_at || 0) - new Date(a.generated_at || 0))[0];
      
      if (latestReminder.status === 'PAID') {
        return (
          <span className={`${styles.reminderBadge} ${styles.reminderBadgePaid}`}>
            Rappel payé
          </span>
        );
      }
      
      if (latestReminder.status === 'OPEN' && latestReminder.reminder_fee_amount) {
        return (
          <span className={`${styles.reminderBadge} ${styles.reminderBadgeOpen}`}>
            Rappel {parseFloat(latestReminder.reminder_fee_amount).toFixed(2)} CHF
          </span>
        );
      }
    }
    
    // ✅ Fallback : utiliser reminder_level pour rétrocompatibilité
    const level = invoice.reminder_level || 0;
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
            <button className={styles.settingsBtn} onClick={handleOpenSettings}>
              ⚙️ Paramètres
            </button>
            <button
              className={styles.exportBtn}
              onClick={() => setExportPaymentsModal({ open: true })}
            >
              ⬇️ Export compta (paiements)
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
        <Filters
          filters={filters}
          onFilterChange={handleFilterChange}
          companyId={company?.id}
          searchInputRef={searchInputRef}
        />
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
              {(invoices || []).map((invoice) => (
                <tr key={invoice.id}>
                  <td>{invoice.invoice_number}</td>
                  <td>
                    {/* ✅ S2: Afficher la clinique si billed_to_company_id est présent */}
                    {invoice.billed_to_company_id && invoice.billed_to_company
                      ? invoice.billed_to_company.name || 'Clinique'
                      : invoice.bill_to_client_id &&
                        invoice.bill_to_client_id !== invoice.client_id &&
                        invoice.bill_to_client
                      ? invoice.bill_to_client.institution_name ||
                        `${invoice.bill_to_client.first_name || ''} ${
                          invoice.bill_to_client.last_name || ''
                        }`.trim()
                      : invoice.client
                        ? invoice.client.institution_name ||
                          `${invoice.client.first_name || ''} ${
                            invoice.client.last_name || ''
                          }`.trim() ||
                          invoice.client.username
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
                  <td>{getReminderBadge(invoice)}</td>
                  <td>
                    <InvoiceRowActions
                      invoice={invoice}
                      onSend={() => handleMarkAsSent(invoice.id)}
                      onSendEmail={() => handleOpenSendEmail(invoice)}
                      onPayment={() => setPaymentModal({ open: true, invoice })}
                      onReminder={() => setReminderModal({ open: true, invoice })}
                      onSendReminderEmail={() => handleOpenSendReminderEmail(invoice)}
                      onRegeneratePdf={() => handleRegeneratePdf(invoice.id)}
                      onCancel={() => handleCancelInvoice(invoice.id)}
                      onDuplicate={() => handleDuplicateInvoice(invoice.id)}
                      onViewPdf={(url) => window.open(url, '_blank')}
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
        refreshTrigger={invoiceDataRefreshTrigger}
      />

      {sendEmailModal.open && (
        <SendEmailModal
          invoice={sendEmailModal.invoice}
          isReminder={sendEmailModal.isReminder}
          reminderId={sendEmailModal.reminderId}
          onClose={() => setSendEmailModal({ open: false, invoice: null, isReminder: false, reminderId: null })}
          onSend={handleSendEmail}
        />
      )}

      <ExportPaymentsModal
        open={exportPaymentsModal.open}
        onClose={() => setExportPaymentsModal({ open: false })}
        companyId={company?.id}
        companyName={company?.name}
        initialYear={filters.year}
        initialMonth={filters.month || null}
      />
    </>
  );
};

export default InvoicesRegistry;
