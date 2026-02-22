import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  FiFileText,
  FiPlus,
  FiDownload,
  FiSettings,
  FiCheckCircle,
  FiCreditCard,
  FiAlertTriangle,
  FiChevronLeft,
  FiChevronRight,
  FiX,
} from 'react-icons/fi';
import styles from './InvoicesRegistry.module.css';
import {
  fetchInvoices,
  getInvoice,
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
import CommandBar from './components/CommandBar';
import InvoiceRowActions from './components/InvoiceRowActions';
import PaymentModal from './components/PaymentModal';
import ReminderModal from './components/ReminderModal';
import NewInvoiceModal from './components/NewInvoiceModal';
import SendEmailModal from './components/SendEmailModal';
import ExportPaymentsModal from './components/ExportPaymentsModal';
import useUrlSearchSync from '../../../../hooks/useUrlSearchSync';

const InvoicesRegistry = () => {
  const { company } = useCompanyData();
  const { public_id: routePublicId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();
  const urlInvoiceId = searchParams.get('invoice_id');
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
  const [confirmDialog, setConfirmDialog] = useState({ open: false, title: '', message: '', variant: 'default', onConfirm: null });
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
      // Le backend renvoie {"data": [...], "pagination": {...}, "stats": {...}}
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

  // Si invoice_id dans l'URL et facture absente de la liste, la charger et l'afficher
  useEffect(() => {
    if (!urlInvoiceId || !company?.id || loading) return;
    const invoiceId = parseInt(urlInvoiceId, 10);
    if (Number.isNaN(invoiceId)) return;

    const fetchAndPrependIfMissing = async () => {
      try {
        const res = await getInvoice(company.id, invoiceId);
        const inv = res?.data ?? res;
        if (inv?.id) {
          setInvoices((prev) => {
            if (prev.some((i) => i.id === invoiceId)) return prev;
            return [inv, ...prev];
          });
        }
      } catch (e) {
        // Ignorer si la facture n'existe pas ou n'est pas accessible
      }
    };
    fetchAndPrependIfMissing();
  }, [urlInvoiceId, company?.id, loading]);

  // Handlers
  const handleFilterChange = (newFilters) => {
    setFilters((prev) => ({ ...prev, ...newFilters, page: 1 }));
  };

  const handlePageChange = (page) => {
    setFilters((prev) => ({ ...prev, page }));
  };

  // Marquer comme envoyée (papier) sans email
  const handleMarkAsSent = (invoiceId) => {
    setConfirmDialog({
      open: true,
      title: 'Marquer comme envoyée',
      message: 'Marquer cette facture comme envoyée par courrier papier ?',
      variant: 'default',
      onConfirm: async () => {
        setConfirmDialog((d) => ({ ...d, open: false }));
        try {
          await markInvoiceAsSent(company.id, invoiceId);
          await loadInvoices();
        } catch (err) {
          setError(err.message || "Erreur lors de l'envoi de la facture");
        }
      },
    });
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

  const handleCancelInvoice = (invoiceId) => {
    setConfirmDialog({
      open: true,
      title: 'Annuler la facture',
      message: 'Cette action est irréversible. La facture sera définitivement annulée et les courses associées seront libérées.',
      variant: 'danger',
      onConfirm: async () => {
        setConfirmDialog((d) => ({ ...d, open: false }));
        try {
          await cancelInvoice(company.id, invoiceId);
          await loadInvoices();
          setInvoiceDataRefreshTrigger((t) => t + 1);
        } catch (err) {
          setError(err.message || "Erreur lors de l'annulation de la facture");
        }
      },
    });
  };

  const handleDuplicateInvoice = (invoiceId) => {
    setConfirmDialog({
      open: true,
      title: 'Brouillon correctif',
      message: 'Créer un brouillon correctif pour cette facture ? Les courses originales seront libérées.',
      variant: 'default',
      onConfirm: async () => {
        setConfirmDialog((d) => ({ ...d, open: false }));
        try {
          const response = await duplicateInvoice(company.id, invoiceId);
          await loadInvoices();
          const draftContext = response?.draft;
          if (draftContext) {
            setNewInvoiceModal({ open: true, invoiceDraft: draftContext });
          }
        } catch (err) {
          setError(err.message || 'Erreur lors de la duplication de la facture');
        }
      },
    });
  };

  const handleOpenSettings = () => {
    const companyId = company?.public_id ?? routePublicId;
    if (companyId) {
      navigate(`/dashboard/company/${companyId}/settings#billing`);
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
    // Utiliser les donnees du rappel consolide si disponibles
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
    
    // Fallback : utiliser reminder_level pour retrocompatibilite
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

  const defaultFilters = useMemo(() => ({
    status: '',
    client_id: '',
    year: new Date().getFullYear(),
    month: '',
    q: '',
    with_balance: false,
    with_reminders: false,
    page: 1,
    per_page: 20,
  }), []);

  const displayedInvoices = useMemo(() => {
    if (!invoices) return [];
    if (filters.status === 'cancelled') return invoices;
    return invoices.filter((inv) => inv.status !== 'cancelled');
  }, [invoices, filters.status]);

  const unknownClientCount = useMemo(() => {
    if (displayedInvoices.length === 0) return 0;
    return displayedInvoices.filter(
      (inv) =>
        inv.status !== 'paid' &&
        !inv.client &&
        !inv.billed_to_company_id &&
        !inv.bill_to_client_id
    ).length;
  }, [displayedInvoices]);

  const getClientName = (invoice) => {
    if (invoice.billed_to_company_id && invoice.billed_to_company) {
      return invoice.billed_to_company.name || 'Clinique';
    }
    if (invoice.bill_to_client_id && invoice.bill_to_client_id !== invoice.client_id && invoice.bill_to_client) {
      return invoice.bill_to_client.institution_name ||
        `${invoice.bill_to_client.first_name || ''} ${invoice.bill_to_client.last_name || ''}`.trim();
    }
    if (invoice.client) {
      return invoice.client.patient_display_name ||
        invoice.client.institution_name ||
        `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
        invoice.client.username;
    }
    if (invoice.client_id) {
      return `Client #${invoice.client_id}`;
    }
    return null;
  };

  const formatDateCH = (dateStr) => {
    if (!dateStr) return '\u2014';
    const d = new Date(dateStr);
    const day = d.getDate().toString().padStart(2, '0');
    const month = (d.getMonth() + 1).toString().padStart(2, '0');
    const year = d.getFullYear();
    return `${day}.${month}.${year}`;
  };

  const getDaysOverdue = (invoice) => {
    if (invoice.status === 'paid' || invoice.status === 'cancelled' || invoice.status === 'draft') return 0;
    if (invoice.balance_due <= 0) return 0;
    const due = new Date(invoice.due_date);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    due.setHours(0, 0, 0, 0);
    const diff = Math.floor((today - due) / (1000 * 60 * 60 * 24));
    return diff > 0 ? diff : 0;
  };

  const getRowClassName = (invoice) => {
    if (invoice.status === 'cancelled') return styles.rowCancelled;
    if (invoice.status === 'draft') return styles.rowDraft;
    if (getDaysOverdue(invoice) > 0) return styles.rowOverdue;
    return '';
  };

  const paginationStart = pagination.total > 0
    ? (pagination.page - 1) * pagination.per_page + 1
    : 0;
  const paginationEnd = pagination.total > 0
    ? Math.min(pagination.page * pagination.per_page, pagination.total)
    : 0;

  return (
    <>
      {/* Zone A — Header */}
      <section className={styles.headerSection}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <h1 className={styles.title}>
              <FiFileText size={24} className={styles.titleIcon} />
              Factures
            </h1>
            <p className={styles.subtitle}>Suivi des emissions et paiements</p>
          </div>
          <div className={styles.headerActions}>
            <button className={styles.settingsBtn} onClick={handleOpenSettings}>
              <FiSettings size={14} />
              Parametres
            </button>
            <button
              className={styles.exportBtn}
              onClick={() => setExportPaymentsModal({ open: true })}
            >
              <FiDownload size={14} />
              Exporter
            </button>
            <button
              className={styles.newInvoiceBtn}
              onClick={() => setNewInvoiceModal({ open: true, invoiceDraft: null })}
            >
              <FiPlus size={14} />
              Nouvelle facture
            </button>
          </div>
        </div>
      </section>

      {/* Zone B — Command Bar */}
      <CommandBar
        filters={filters}
        defaultFilters={defaultFilters}
        onFilterChange={handleFilterChange}
        companyId={company?.id}
        searchInputRef={searchInputRef}
      />

      {/* Zone C — KPIs + Alertes */}
      <div className={styles.stats}>
        <div className={styles.statCard}>
          <FiFileText size={20} className={styles.statIcon} />
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>Total emis</h3>
            <p className={styles.statValue}>{stats.total_issued?.toFixed(2) || '0.00'} CHF</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <FiCheckCircle size={20} className={styles.statIcon} />
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>Paye</h3>
            <p className={styles.statValue}>{stats.total_paid?.toFixed(2) || '0.00'} CHF</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <FiCreditCard size={20} className={styles.statIcon} />
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>Solde</h3>
            <p className={styles.statValue}>{stats.total_balance?.toFixed(2) || '0.00'} CHF</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <FiAlertTriangle size={20} className={styles.statIcon} />
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>En retard</h3>
            <p className={styles.statValue}>{stats.overdue_count || 0}</p>
          </div>
        </div>
      </div>

      {/* Alertes conditionnelles */}
      {(stats.overdue_count > 0 || unknownClientCount > 0) && (
        <div className={styles.alertsBar}>
          {stats.overdue_count > 0 && (
            <div className={styles.alertItem} data-type="danger">
              <FiAlertTriangle size={14} />
              <span>{stats.overdue_count} facture{stats.overdue_count > 1 ? 's' : ''} en retard</span>
              <button
                className={styles.alertLink}
                onClick={() => handleFilterChange({ status: 'overdue' })}
              >
                Voir
              </button>
            </div>
          )}
          {unknownClientCount > 0 && (
            <div className={styles.alertItem} data-type="warning">
              <FiAlertTriangle size={14} />
              <span>{unknownClientCount} facture{unknownClientCount > 1 ? 's' : ''} sans client sur cette page</span>
            </div>
          )}
        </div>
      )}

      {/* Messages d'erreur */}
      {error && (
        <div className={styles.error}>
          {error}
          <button onClick={() => setError(null)} aria-label="Fermer">
            <FiX size={16} />
          </button>
        </div>
      )}

      {/* Zone D — Table */}
      <div className={styles.tableContainer}>
        {loading ? (
          <div className={styles.loading}>Chargement...</div>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>N&#xB0; facture</th>
                <th>Client</th>
                <th>Echeance</th>
                <th>Montant</th>
                <th>Paiement</th>
                <th>Statut</th>
                <th>Rappel</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayedInvoices.map((invoice) => {
                const clientName = getClientName(invoice);
                const daysOverdue = getDaysOverdue(invoice);
                return (
                  <tr key={invoice.id} className={getRowClassName(invoice)}>
                    <td>
                      <div className={styles.cellInvoiceNum}>
                        <span className={styles.invoiceNumber}>{invoice.invoice_number}</span>
                        <span className={styles.invoicePeriod}>
                          {invoice.period_month.toString().padStart(2, '0')}.{invoice.period_year}
                        </span>
                      </div>
                    </td>
                    <td>
                      <div className={styles.cellClient}>
                        <span className={styles.clientName} title={clientName || ''}>
                          {clientName || '\u2014'}
                        </span>
                        {!clientName && (
                          <span className={styles.badgeWarning}>Client manquant</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <div className={styles.cellDueDate}>
                        <span>{formatDateCH(invoice.due_date)}</span>
                        {daysOverdue > 0 && (
                          <span className={styles.badgeOverdueSmall}>J+{daysOverdue}</span>
                        )}
                      </div>
                    </td>
                    <td className={styles.cellAmount}>
                      {invoice.total_amount.toFixed(2)} CHF
                    </td>
                    <td>
                      <div className={styles.cellPayment}>
                        <span className={styles.paymentPaid}>
                          Paye: {invoice.amount_paid.toFixed(2)} CHF
                        </span>
                        <span className={invoice.balance_due > 0 ? styles.paymentBalance : styles.paymentPaid}>
                          Solde: {invoice.balance_due.toFixed(2)} CHF
                        </span>
                      </div>
                    </td>
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
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {pagination.pages > 1 && (
        <div className={styles.pagination}>
          <span className={styles.paginationInfo}>
            Affichage {paginationStart}&ndash;{paginationEnd} sur {pagination.total || 0}
          </span>
          <div className={styles.paginationControls}>
            <button
              disabled={!pagination.has_prev}
              onClick={() => handlePageChange(pagination.page - 1)}
              aria-label="Page precedente"
            >
              <FiChevronLeft size={16} />
            </button>
            <span className={styles.paginationPage}>
              Page {pagination.page} sur {pagination.pages}
            </span>
            <button
              disabled={!pagination.has_next}
              onClick={() => handlePageChange(pagination.page + 1)}
              aria-label="Page suivante"
            >
              <FiChevronRight size={16} />
            </button>
          </div>
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

      {/* Confirm dialog */}
      {confirmDialog.open && (
        <div className={styles.confirmOverlay} onClick={() => setConfirmDialog((d) => ({ ...d, open: false }))}>
          <div className={styles.confirmModal} onClick={(e) => e.stopPropagation()}>
            <h3 className={styles.confirmTitle}>{confirmDialog.title}</h3>
            <p className={styles.confirmMessage}>{confirmDialog.message}</p>
            <div className={styles.confirmActions}>
              <button
                type="button"
                className={styles.confirmCancel}
                onClick={() => setConfirmDialog((d) => ({ ...d, open: false }))}
              >
                Annuler
              </button>
              <button
                type="button"
                className={`${styles.confirmBtn} ${confirmDialog.variant === 'danger' ? styles.confirmBtnDanger : styles.confirmBtnDefault}`}
                onClick={confirmDialog.onConfirm}
              >
                Confirmer
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default InvoicesRegistry;
