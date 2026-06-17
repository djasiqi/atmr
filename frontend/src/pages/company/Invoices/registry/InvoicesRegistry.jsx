import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useQuery, keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  FiFileText,
  FiDownload,
  FiSettings,
  FiCheckCircle,
  FiCreditCard,
  FiAlertTriangle,
  FiChevronLeft,
  FiChevronRight,
  FiX,
  FiSend,
  FiMail,
  FiCheck,
  FiList,
} from 'react-icons/fi';
import styles from './InvoicesRegistry.module.css';
import {
  fetchInvoices,
  getInvoice,
  sendInvoiceByEmail,
  markInvoiceAsSent,
  bulkMarkAsSent,
  sendReminderByEmail,
  postPayment,
  postReminder,
  regenerateInvoicePdf,
  cancelInvoice,
  duplicateInvoice,
  fetchBillingOpportunities,
  getEffectiveDueDate,
  getDaysOverdue,
} from '../../../../services/invoiceService';
import { useLirieCompany } from '../../../../hooks/useLirieCompany';
import { lirieKeys, invoiceFiltersHash } from '../../../../queryKeys/lirie';
import CommandBar from './components/CommandBar';
import InvoiceRowActions from './components/InvoiceRowActions';
import PaymentModal from './components/PaymentModal';
import ReminderModal from './components/ReminderModal';
import NewInvoiceModal from './components/NewInvoiceModal';
import BillPeriodModal from './components/BillPeriodModal';
import InvoiceDraftEditModal from './components/InvoiceDraftEditModal';
import SendEmailModal from './components/SendEmailModal';
import ExportPaymentsModal from './components/ExportPaymentsModal';
import useUrlSearchSync from '../../../../hooks/useUrlSearchSync';
import InvoicesTableSkeleton from './components/InvoicesTableSkeleton';

const extractApiError = (err, fallback = 'Erreur inconnue') => {
  const data = err?.response?.data;
  if (data) {
    if (typeof data === 'string') return data;
    if (typeof data.error === 'string') return data.error;
    if (typeof data.message === 'string') return data.message;
  }
  return err?.message || fallback;
};

const InvoicesRegistry = () => {
  const { company } = useLirieCompany();
  const { public_id: routePublicId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const searchInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();
  const urlInvoiceId = searchParams.get('invoice_id');
  const [actionError, setActionError] = useState(null);
  const [listErrorDismissed, setListErrorDismissed] = useState(false);
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
  const [billPeriodOpen, setBillPeriodOpen] = useState(false);
  const [draftEditInvoice, setDraftEditInvoice] = useState(null);
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

  // Bulk selection state
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [bulkLoading, setBulkLoading] = useState(false);

  const queryClient = useQueryClient();
  const canLoadInvoices = Boolean(company?.id);
  const filtersHash = useMemo(() => invoiceFiltersHash(filters), [filters]);

  const opportunityYear = filters.year || new Date().getFullYear();
  const opportunityMonth = filters.month
    ? Number(filters.month)
    : new Date().getMonth() + 1;

  const { data: billingOpportunitiesPayload } = useQuery({
    queryKey: ['billingOpportunities', company?.id, opportunityYear, opportunityMonth],
    queryFn: async () => {
      const res = await fetchBillingOpportunities(
        company.id,
        opportunityYear,
        opportunityMonth
      );
      if (res && typeof res === 'object' && res.data && typeof res.data === 'object') {
        return res.data;
      }
      return res;
    },
    enabled: Boolean(company?.id),
    staleTime: 60_000,
  });

  const {
    data: listData,
    error: invoicesListError,
    isError: invoicesListIsError,
    isLoading: listInitialLoading,
    isRefetching: listRefetching,
    refetch: refetchInvoicesList,
  } = useQuery({
    queryKey: canLoadInvoices
      ? lirieKeys.companyInvoices(company.id, filtersHash)
      : ['lirie', 'company-invoices', 'disabled'],
    enabled: canLoadInvoices,
    queryFn: async () => {
      const response = await fetchInvoices(company.id, filters);
      const invoicesData = response?.data || response?.invoices || [];
      return {
        invoices: Array.isArray(invoicesData) ? invoicesData : [],
        pagination: response?.pagination || {},
        stats: response?.stats || {},
      };
    },
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });

  const invoices = useMemo(
    () => (Array.isArray(listData?.invoices) ? listData.invoices : []),
    [listData]
  );
  const pagination = listData?.pagination ?? {};
  const stats = listData?.stats ?? {};

  const listErrorMessage = useMemo(
    () =>
      invoicesListIsError
        ? extractApiError(invoicesListError, 'Erreur lors du chargement des factures')
        : null,
    [invoicesListIsError, invoicesListError]
  );

  useEffect(() => {
    if (!invoicesListIsError) setListErrorDismissed(false);
  }, [invoicesListIsError]);

  const loadInvoices = useCallback(() => refetchInvoicesList(), [refetchInvoicesList]);
  const showListTableSkeleton = canLoadInvoices && listInitialLoading;
  const showListRefetching = listRefetching && !listInitialLoading;
  const showListError = invoicesListIsError && !listErrorDismissed;

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

  // Si invoice_id dans l'URL et facture absente de la liste, la charger et l'ajouter au cache
  useEffect(() => {
    if (!urlInvoiceId || !company?.id || listInitialLoading) return;
    const invoiceId = parseInt(urlInvoiceId, 10);
    if (Number.isNaN(invoiceId)) return;

    const key = lirieKeys.companyInvoices(company.id, filtersHash);
    const fetchAndPrependIfMissing = async () => {
      try {
        const res = await getInvoice(company.id, invoiceId);
        const inv = res?.data ?? res;
        if (inv?.id) {
          queryClient.setQueryData(key, (old) => {
            if (!old?.invoices) return old;
            if (old.invoices.some((i) => i.id === invoiceId)) return old;
            return { ...old, invoices: [inv, ...old.invoices] };
          });
        }
      } catch (e) {
        // Ignorer si la facture n'existe pas ou n'est pas accessible
      }
    };
    fetchAndPrependIfMissing();
  }, [urlInvoiceId, company?.id, listInitialLoading, filtersHash, queryClient]);

  // Handlers
  const handleFilterChange = (newFilters) => {
    setFilters((prev) => ({ ...prev, ...newFilters, page: 1 }));
  };

  const handlePageChange = (page) => {
    setFilters((prev) => ({ ...prev, page }));
  };

  const displayedInvoices = useMemo(() => {
    if (!invoices) return [];
    if (filters.status === 'cancelled') return invoices;
    return invoices.filter((inv) => inv.status !== 'cancelled');
  }, [invoices, filters.status]);

  // Clear selection when invoices change (page change, filter change, etc.)
  useEffect(() => {
    setSelectedIds(new Set());
  }, [invoices]);

  // Selection helpers
  const toggleSelect = useCallback((id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds((prev) => {
      if (prev.size === displayedInvoices.length && displayedInvoices.length > 0) {
        return new Set();
      }
      return new Set(displayedInvoices.map((inv) => inv.id));
    });
  }, [displayedInvoices]);

  const selectAllDrafts = useCallback(() => {
    const draftIds = displayedInvoices
      .filter((inv) => inv.status === 'draft')
      .map((inv) => inv.id);
    setSelectedIds(new Set(draftIds));
  }, [displayedInvoices]);

  const selectTodaysDrafts = useCallback(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const draftIds = displayedInvoices
      .filter((inv) => {
        if (inv.status !== 'draft') return false;
        const issued = new Date(inv.issued_at || inv.created_at);
        issued.setHours(0, 0, 0, 0);
        return issued.getTime() === today.getTime();
      })
      .map((inv) => inv.id);
    setSelectedIds(new Set(draftIds));
  }, [displayedInvoices]);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  const selectedInvoices = useMemo(() => {
    return displayedInvoices.filter((inv) => selectedIds.has(inv.id));
  }, [displayedInvoices, selectedIds]);

  const selectedDraftCount = useMemo(() => {
    return selectedInvoices.filter((inv) => inv.status === 'draft').length;
  }, [selectedInvoices]);

  // Bulk: mark selected drafts as sent (paper)
  const handleBulkMarkAsSent = (method = 'paper') => {
    const draftIds = selectedInvoices
      .filter((inv) => inv.status === 'draft')
      .map((inv) => inv.id);

    if (draftIds.length === 0) return;

    const label = method === 'email' ? 'par email' : 'par courrier papier';
    const bulkTitle =
      method === 'email'
        ? `Envoyer ${draftIds.length} facture(s) par email`
        : `Marquer ${draftIds.length} facture(s) comme envoyée(s)`;
    const bulkMessage =
      method === 'email'
        ? `Envoyer ${draftIds.length} brouillon(s) par email (une facture par message) ? Les échecs seront listés sans bloquer les autres.`
        : `Marquer ${draftIds.length} brouillon(s) comme envoyé(s) ${label} ?`;
    setConfirmDialog({
      open: true,
      title: bulkTitle,
      message: bulkMessage,
      variant: 'default',
      onConfirm: async () => {
        setConfirmDialog((d) => ({ ...d, open: false }));
        try {
          setBulkLoading(true);
          const payload = await bulkMarkAsSent(company.id, draftIds, method);
          const inner = payload?.data ?? payload;
          const sent = Array.isArray(inner?.sent) ? inner.sent : [];
          const failed = Array.isArray(inner?.failed) ? inner.failed : [];
          setSelectedIds((prev) => {
            const next = new Set(prev);
            sent.forEach((id) => next.delete(id));
            return next;
          });
          await loadInvoices();
          if (failed.length > 0) {
            const preview = failed
              .slice(0, 5)
              .map((f) => `#${f.invoice_id}: ${String(f.error || '').slice(0, 96)}`)
              .join(' ');
            setActionError(
              `${failed.length} facture(s) en échec${
                sent.length ? ` (${sent.length} réussite(s))` : ''
              }. ${preview}${failed.length > 5 ? ' …' : ''}`
            );
          }
        } catch (err) {
          setActionError(extractApiError(err, "Erreur lors de l'envoi groupé"));
        } finally {
          setBulkLoading(false);
        }
      },
    });
  };

  // Marquer comme envoyée (papier) sans email
  const handleMarkAsSent = (invoiceId, { afterSuccess } = {}) => {
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
          afterSuccess?.();
        } catch (err) {
          setActionError(extractApiError(err, "Erreur lors de l'envoi de la facture"));
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
      setActionError('Aucun rappel trouvé pour cette facture');
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
    const sentInvoiceId = sendEmailModal.invoice?.id;
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
      if (draftEditInvoice && sentInvoiceId && draftEditInvoice.id === sentInvoiceId) {
        closeDraftEdit();
      }
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
      setActionError(extractApiError(err, "Erreur lors de l'enregistrement du paiement"));
    }
  };

  const handleReminder = async (invoiceId, level) => {
    try {
      await postReminder(company.id, invoiceId, { level });
      await loadInvoices();
      setReminderModal({ open: false, invoice: null });
    } catch (err) {
      setActionError(extractApiError(err, 'Erreur lors de la génération du rappel'));
    }
  };

  const handleRegeneratePdf = async (invoiceId) => {
    try {
      await regenerateInvoicePdf(company.id, invoiceId);
      await loadInvoices();
    } catch (err) {
      setActionError(extractApiError(err, 'Erreur lors de la régénération du PDF'));
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
          setActionError(extractApiError(err, "Erreur lors de l'annulation de la facture"));
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
          setActionError(extractApiError(err, 'Erreur lors de la duplication de la facture'));
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

  const handleNewInvoiceGenerated = (_invoice) => {
    loadInvoices();
  };

  const clearDraftEditParams = useCallback(() => {
    setSearchParams((prev) => {
      const p = new URLSearchParams(prev);
      p.delete('draft_edit');
      p.delete('invoice_id');
      return p;
    });
  }, [setSearchParams]);

  /** Rafraîchissement liste après génération depuis le compositeur « Nouvelle facture » (édition dans la même modale). */
  const handleComposerInvoiceGenerated = useCallback(() => {
    loadInvoices();
  }, [loadInvoices]);

  const closeDraftEdit = useCallback(() => {
    setDraftEditInvoice(null);
    clearDraftEditParams();
  }, [clearDraftEditParams]);

  const handleOpenDraftEdit = useCallback(
    (invoice) => {
      if (!invoice?.id) return;
      setDraftEditInvoice(invoice);
      setSearchParams((prev) => {
        const p = new URLSearchParams(prev);
        p.set('invoice_id', String(invoice.id));
        p.set('draft_edit', '1');
        return p;
      });
    },
    [setSearchParams]
  );

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

  const getPayerName = (invoice) => {
    if (invoice.billing_party?.display_name) {
      return invoice.billing_party.display_name;
    }
    if (invoice.billed_to_company_id && invoice.billed_to_company) {
      return invoice.billed_to_company.name || 'Clinique';
    }
    if (
      invoice.bill_to_client_id &&
      invoice.bill_to_client_id !== invoice.client_id &&
      invoice.bill_to_client
    ) {
      return (
        invoice.bill_to_client.institution_name ||
        `${invoice.bill_to_client.first_name || ''} ${invoice.bill_to_client.last_name || ''}`.trim()
      );
    }
    return null;
  };

  const getClientName = (invoice) => {
    // For S2 clinic monthly invoices, the primary label must remain the clinic/payer.
    if (invoice.billing_strategy === 's2_clinic_monthly') {
      return getPayerName(invoice) || 'Clinique';
    }
    if (invoice.client) {
      return invoice.client.patient_display_name ||
        `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
        invoice.client.institution_name ||
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

  const getDaysOverdueLocal = (invoice) => getDaysOverdue(invoice);

  const getRowClassName = (invoice) => {
    if (invoice.status === 'cancelled') return styles.rowCancelled;
    if (invoice.status === 'draft') return styles.rowDraft;
    if (getDaysOverdueLocal(invoice) > 0) return styles.rowOverdue;
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
              type="button"
              className={styles.newInvoiceBtn}
              onClick={() => setBillPeriodOpen(true)}
              data-tour-id="invoice-new-invoice-button"
              title="Nouvelle facture — par période ou assistant complet (dans la fenêtre)"
            >
              <FiFileText size={14} />
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
      <div className={styles.stats} data-tour-id="invoice-stats">
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

      {billingOpportunitiesPayload && typeof billingOpportunitiesPayload.total_draft_would_create === 'number' && (
        <div className={styles.opportunitiesBar} data-tour-id="billing-opportunities">
          <FiList size={16} />
          <span>
            Période {String(opportunityMonth).padStart(2, '0')}/{opportunityYear} (filtre mois) :{' '}
            <strong>{billingOpportunitiesPayload.total_draft_would_create}</strong> payeur
            {billingOpportunitiesPayload.total_draft_would_create > 1 ? 's' : ''} avec courses à
            facturer (patients + cliniques).
          </span>
          <button
            type="button"
            className={styles.alertLink}
            onClick={() => setBillPeriodOpen(true)}
          >
            Nouvelle facture
          </button>
        </div>
      )}

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

      {/* Erreur liste (query) ou erreur d'action (mutations) */}
      {(showListError || actionError) && (
        <div className={styles.error} role="alert">
          {showListError && (
            <span>
              {listErrorMessage}
              <button
                type="button"
                className={styles.errorRetryBtn}
                onClick={() => {
                  setListErrorDismissed(false);
                  void refetchInvoicesList();
                }}
              >
                Réessayer
              </button>
            </span>
          )}
          {showListError && actionError && ' '}
          {actionError && <span>{actionError}</span>}
          <button
            onClick={() => {
              if (invoicesListIsError) setListErrorDismissed(true);
              setActionError(null);
            }}
            type="button"
            aria-label="Fermer"
          >
            <FiX size={16} />
          </button>
        </div>
      )}

      {/* Bulk action bar */}
      {selectedIds.size > 0 && (
        <div className={styles.bulkBar}>
          <div className={styles.bulkBarLeft}>
            <span className={styles.bulkCount}>
              <FiCheck size={14} />
              {selectedIds.size} sélectionnée{selectedIds.size > 1 ? 's' : ''}
            </span>
            <div className={styles.bulkQuickSelect}>
              <button
                type="button"
                className={styles.bulkQuickBtn}
                onClick={selectAllDrafts}
              >
                Tous les brouillons
              </button>
              <button
                type="button"
                className={styles.bulkQuickBtn}
                onClick={selectTodaysDrafts}
              >
                Brouillons du jour
              </button>
              <button
                type="button"
                className={styles.bulkQuickBtn}
                onClick={toggleSelectAll}
              >
                {selectedIds.size === displayedInvoices.length ? 'Tout désélectionner' : 'Tout sélectionner'}
              </button>
            </div>
          </div>
          <div className={styles.bulkBarRight}>
            {selectedDraftCount > 0 && (
              <>
                <button
                  type="button"
                  className={styles.bulkActionBtn}
                  onClick={() => handleBulkMarkAsSent('paper')}
                  disabled={bulkLoading}
                >
                  <FiSend size={13} />
                  Envoyée (papier) ({selectedDraftCount})
                </button>
                <button
                  type="button"
                  className={`${styles.bulkActionBtn} ${styles.bulkActionBtnEmail}`}
                  onClick={() => handleBulkMarkAsSent('email')}
                  disabled={bulkLoading}
                >
                  <FiMail size={13} />
                  Envoyer par email ({selectedDraftCount})
                </button>
              </>
            )}
            <button
              type="button"
              className={styles.bulkCancelBtn}
              onClick={clearSelection}
            >
              <FiX size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Zone D — Table : premier chargement = squelette ; rechargement = contenu + barre */}
      {showListTableSkeleton ? (
        <InvoicesTableSkeleton rowCount={Math.min(10, Math.max(5, filters.per_page || 7))} />
      ) : (
        <div
          className={showListRefetching ? styles.listBlockRefreshing : styles.listBlock}
          aria-busy={showListRefetching}
        >
          {showListRefetching && (
            <div
              className={styles.listRefreshBar}
              role="status"
              aria-label="Mise à jour des factures"
            />
          )}
          <div className={styles.tableContainer}>
            <table className={styles.table} data-tour-id="invoice-table">
            <thead>
              <tr>
                <th className={styles.thCheckbox}>
                  <input
                    type="checkbox"
                    className={styles.checkbox}
                    checked={displayedInvoices.length > 0 && selectedIds.size === displayedInvoices.length}
                    onChange={toggleSelectAll}
                    title="Tout sélectionner"
                  />
                </th>
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
              {displayedInvoices.map((invoice, index) => {
                const clientName = getClientName(invoice);
                const payerName = getPayerName(invoice);
                const daysOverdue = getDaysOverdueLocal(invoice);
                const displayDueDate = getEffectiveDueDate(invoice);
                return (
                  <tr key={invoice.id} className={`${getRowClassName(invoice)} ${selectedIds.has(invoice.id) ? styles.rowSelected : ''}`}>
                    <td className={styles.tdCheckbox}>
                      <input
                        type="checkbox"
                        className={styles.checkbox}
                        checked={selectedIds.has(invoice.id)}
                        onChange={() => toggleSelect(invoice.id)}
                      />
                    </td>
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
                        {payerName && payerName !== clientName && (
                          <span className={styles.clientMeta} title={`Payeur: ${payerName}`}>
                            Payeur: {payerName}
                          </span>
                        )}
                        {!clientName && (
                          <span className={styles.badgeWarning}>Client manquant</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <div className={styles.cellDueDate}>
                        <span>{formatDateCH(displayDueDate)}</span>
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
                        isGuideAnchor={index === 0}
                        onSend={() => handleMarkAsSent(invoice.id)}
                        onSendEmail={() => handleOpenSendEmail(invoice)}
                        onPayment={() => setPaymentModal({ open: true, invoice })}
                        onReminder={() => setReminderModal({ open: true, invoice })}
                        onSendReminderEmail={() => handleOpenSendReminderEmail(invoice)}
                        onRegeneratePdf={() => handleRegeneratePdf(invoice.id)}
                        onCancel={() => handleCancelInvoice(invoice.id)}
                        onDuplicate={() => handleDuplicateInvoice(invoice.id)}
                        onEditDraft={() => handleOpenDraftEdit(invoice)}
                        onViewPdf={(url) => window.open(url, '_blank')}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
        </div>
      )}

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

      <BillPeriodModal
        open={billPeriodOpen}
        onClose={() => setBillPeriodOpen(false)}
        companyId={company?.id}
        onInvoiceGenerated={handleComposerInvoiceGenerated}
        onOpenSendEmail={handleOpenSendEmail}
        onMarkAsSent={(inv) =>
          handleMarkAsSent(inv.id, { afterSuccess: () => setBillPeriodOpen(false) })
        }
      />

      <InvoiceDraftEditModal
        open={Boolean(draftEditInvoice)}
        initialInvoice={draftEditInvoice}
        companyId={company?.id}
        onClose={closeDraftEdit}
        onUpdated={loadInvoices}
        onOpenSendEmail={(inv) => {
          setSendEmailModal({ open: true, invoice: inv, isReminder: false, reminderId: null });
        }}
        onMarkAsSent={(inv) => {
          if (inv?.id) handleMarkAsSent(inv.id, { afterSuccess: closeDraftEdit });
        }}
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
