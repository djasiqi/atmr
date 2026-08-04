import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  closePlatformBillingContract,
  createPlatformBillingContract,
  downloadPartnerAgreementDocxUrl,
  downloadPartnerAgreementFile,
  downloadPartnerAgreementPackageUrl,
  downloadPartnerAgreementParticularPdfUrl,
  downloadPartnerAgreementPreviewUrl,
  downloadPartnerAgreementSignedUrl,
  fetchPlatformBillingCompaniesConfig,
  fetchPlatformBillingContracts,
  fetchPlatformBillingCreditor,
  fetchPlatformPricingGrids,
  generatePartnerAgreement,
  markPartnerAgreementSent,
  migratePartnerAgreementToV120,
  putPlatformBillingDebtorAddress,
  uploadPartnerAgreementSigned,
  voidPartnerAgreement,
} from '../../../services/adminService';
import styles from './AdminBillingTransportConfig.module.css';
import { adminPaths } from '../routing/adminRoutePaths';
import AdminActionDialog from '../components/AdminActionDialog';

const emptyContract = {
  is_billing_enabled: false,
  own_portfolio_billing_enabled: false,
  lirie_commission_enabled: false,
  support_enabled: false,
  subscription_pricing_mode: 'volume',
  custom_subscription_amount: '',
  use_global_pricing_grid: true,
  commission_rate: '',
  commission_cancellation_policy: 'exclude',
  free_license_max_months: '',
  statement_dispute_days: '10',
  support_hourly_rate_default: '',
  payment_terms_days: '30',
  automated_dunning_enabled: false,
  reminder_delay_days_after_due: '0',
  reminder_grace_days: '10',
  full_suspend_days_after_due: '30',
  full_suspend_overdue_invoice_count: '2',
  termination_notice_days: '10',
  partial_block_marketplace_offers: true,
  partial_block_marketplace_acceptance: true,
  partial_block_billable_support: true,
  partial_block_billable_configuration: true,
  effective_from: '',
  notes: '',
  contract_special_conditions: '',
};

const LEGAL_FORMS = [
  { value: 'sole_proprietorship', label: 'Indépendant' },
  { value: 'sarl', label: 'Sàrl' },
  { value: 'sa', label: 'SA' },
  { value: 'association', label: 'Association' },
  { value: 'foundation', label: 'Fondation' },
  { value: 'other', label: 'Autre' },
];

const CANCEL_POLICIES = [
  { value: 'exclude', label: 'Exclure les annulations' },
  { value: 'on_cancellation_fees', label: 'Commission sur frais d’annulation' },
  { value: 'on_billed_amount', label: 'Commission sur montant facturé' },
];

const CONTRACT_TABS = [
  { id: 'identity', label: 'Identité' },
  { id: 'products', label: 'Produits' },
  { id: 'dunning', label: 'Recouvrement' },
  { id: 'document', label: 'Contrat' },
];

const agreementStatusClass = (status) => {
  if (status === 'signed') return 'docBadgeSigned';
  if (status === 'sent') return 'docBadgeSent';
  if (status === 'void') return 'docBadgeVoid';
  return 'docBadgeDraft';
};

const statusLabel = (s) =>
  ({
    draft: 'Brouillon',
    sent: 'Envoyé',
    signed: 'Signé',
    void: 'Annulé',
  }[s] || s || '—');

const fmtDec = (s) => (s == null || s === '' ? '—' : `${s} CHF`);

const portfolioLabel = (c) => {
  if (!c?.is_billing_enabled || !c?.own_portfolio_billing_enabled) return 'Désactivé';
  if (c.subscription_pricing_mode === 'fixed') return 'Montant fixe';
  if (c.subscription_pricing_mode === 'free') return 'Gratuit';
  return 'Par volume';
};

/** commission_rate API = décimal "0.030000" → affichage "3 %". */
const commissionLabel = (c) => {
  if (!c?.is_billing_enabled || !c?.lirie_commission_enabled) return 'Désactivée';
  if (c.commission_rate == null || c.commission_rate === '') return '—';
  const n = Number(String(c.commission_rate).replace(',', '.'));
  if (Number.isNaN(n)) return String(c.commission_rate);
  return `${(n * 100).toLocaleString('fr-CH', { maximumFractionDigits: 2 })} %`;
};

const configLabel = (c) => {
  if (!c) return 'Non configurée';
  if (!c.is_billing_enabled) return 'Inactive';
  return 'Active';
};

const isLikelyTestCompany = (name) => {
  const n = (name || '').trim().toLowerCase();
  if (!n) return true;
  return (
    n.startsWith('test ') ||
    n.startsWith('test company') ||
    n.startsWith('transport ') ||
    n.includes('test co') ||
    n.includes('footer test') ||
    n.includes('header gate') ||
    n.includes('lines gate') ||
    /^test company [0-9a-f]{6,}/i.test(name) ||
    /^transport [0-9a-f]{6,}/i.test(name)
  );
};

/** Décimal API → pourcentage affichable (vide si absent). */
const rateToPercent = (rate) => {
  if (rate == null || rate === '') return '';
  const n = Number(String(rate).replace(',', '.'));
  if (Number.isNaN(n)) return '';
  return String(Number((n * 100).toFixed(4)));
};

/** Pourcentage UI → décimal API. */
const percentToRate = (percent) => {
  if (percent == null || percent === '') return null;
  const n = Number(String(percent).replace(',', '.'));
  if (Number.isNaN(n)) return null;
  return (n / 100).toFixed(6);
};

const isoToMonth = (iso) => {
  if (!iso) return '';
  const m = String(iso).match(/^(\d{4}-\d{2})/);
  if (m) return m[1];
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
};

/** Mois calendaire YYYY-MM → { effective_year, effective_month }. */
const monthToYearMonth = (ym) => {
  if (!ym || !/^\d{4}-\d{2}$/.test(ym)) return null;
  const [y, m] = ym.split('-').map(Number);
  if (!y || !m || m < 1 || m > 12) return null;
  return { effective_year: y, effective_month: m };
};

const fmtPeriod = (from, to) => {
  const a = isoToMonth(from) || 'ouvert';
  const b = isoToMonth(to) || '∞';
  return `${a} → ${b}`;
};

const isContractOpen = (c) => c && (c.effective_to == null || c.effective_to === '');

const contractFromRow = (c) => ({
  ...emptyContract,
  is_billing_enabled: !!c?.is_billing_enabled,
  own_portfolio_billing_enabled: !!c?.own_portfolio_billing_enabled,
  lirie_commission_enabled: !!c?.lirie_commission_enabled,
  support_enabled: !!c?.support_enabled,
  subscription_pricing_mode: c?.subscription_pricing_mode || 'volume',
  custom_subscription_amount: c?.custom_subscription_amount || '',
  commission_rate: c?.commission_rate || '',
  commission_cancellation_policy: c?.commission_cancellation_policy || 'exclude',
  free_license_max_months:
    c?.free_license_max_months != null ? String(c.free_license_max_months) : '',
  statement_dispute_days:
    c?.statement_dispute_days != null ? String(c.statement_dispute_days) : '10',
  support_hourly_rate_default: c?.support_hourly_rate_default || '',
  payment_terms_days:
    c?.payment_terms_days != null ? String(c.payment_terms_days) : '30',
  automated_dunning_enabled: !!c?.automated_dunning_enabled,
  reminder_delay_days_after_due: String(c?.reminder_delay_days_after_due ?? 0),
  reminder_grace_days: String(c?.reminder_grace_days ?? 10),
  full_suspend_days_after_due: String(c?.full_suspend_days_after_due ?? 30),
  full_suspend_overdue_invoice_count: String(
    c?.full_suspend_overdue_invoice_count ?? 2
  ),
  termination_notice_days: String(c?.termination_notice_days ?? 10),
  partial_block_marketplace_offers: c?.partial_block_marketplace_offers !== false,
  partial_block_marketplace_acceptance:
    c?.partial_block_marketplace_acceptance !== false,
  partial_block_billable_support: c?.partial_block_billable_support !== false,
  partial_block_billable_configuration:
    c?.partial_block_billable_configuration !== false,
  effective_from: c?.effective_from || '',
  notes: c?.notes || '',
  contract_special_conditions: c?.contract_special_conditions || '',
});

/** Affiche uniquement les points encore à compléter (masque les « Prêt »). */
const ReadinessItem = ({ ok, label, hint, errors }) => {
  if (ok) return null;
  return (
    <li className={`${styles.readinessItem} ${styles.readinessItemWarn}`}>
      <div className={styles.readinessRow}>
        <span className={styles.readinessLabel}>{label}</span>
        <span className={styles.badgeOff}>À compléter</span>
      </div>
      {hint ? <p className={styles.readinessHint}>{hint}</p> : null}
      {errors?.length ? (
        <p className={styles.readinessHint}>{errors.join(' · ')}</p>
      ) : null}
    </li>
  );
};

const emptyDebtor = {
  legal_name: '',
  street_name: '',
  building_number: '',
  postal_code: '',
  city: '',
  country_code: 'CH',
  uid_ide: '',
  legal_form: 'sarl',
  signatory_name: '',
  signatory_title: '',
};

const AdminBillingDualProductConfig = () => {
  const { public_id: adminId } = useParams();
  const settingsPath = adminPaths.configuration(adminId);

  const [items, setItems] = useState([]);
  const [grids, setGrids] = useState([]);
  const [creditor, setCreditor] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [showTestCompanies, setShowTestCompanies] = useState(false);
  const [modalCompany, setModalCompany] = useState(null);
  const [contracts, setContracts] = useState([]);
  const [readiness, setReadiness] = useState(null);
  const [debtorForm, setDebtorForm] = useState(emptyDebtor);
  const [form, setForm] = useState(emptyContract);
  const [commissionPercent, setCommissionPercent] = useState('');
  const [effectiveMonth, setEffectiveMonth] = useState('');
  const [saving, setSaving] = useState(false);
  const [modalError, setModalError] = useState(null);
  const [partnerIdentity, setPartnerIdentity] = useState(null);
  const [selectedContractId, setSelectedContractId] = useState(null);
  const [savedSnapshot, setSavedSnapshot] = useState('');
  const [initialDebtorSnapshot, setInitialDebtorSnapshot] = useState('');
  const [initialCommercialSnapshot, setInitialCommercialSnapshot] = useState('');
  const [signedOn, setSignedOn] = useState('');
  const [docBusy, setDocBusy] = useState(false);
  const [actionDialog, setActionDialog] = useState(null);
  const [modalTab, setModalTab] = useState('identity');
  const [rcAttested, setRcAttested] = useState(false);
  const [deliveryChannel, setDeliveryChannel] = useState('email');
  const [deliveryRecipient, setDeliveryRecipient] = useState('');
  const [signedAdditionalPagesConfirmed, setSignedAdditionalPagesConfirmed] =
    useState(false);
  const [rcSignatureMode, setRcSignatureMode] = useState('individual');
  const [rcCoSignatoryName, setRcCoSignatoryName] = useState('');
  const [rcCoSignatoryFunction, setRcCoSignatoryFunction] = useState('');
  const [rcRegisterName, setRcRegisterName] = useState(
    'Registre du commerce / Zefix'
  );

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [listRes, gridRes, credRes] = await Promise.all([
        fetchPlatformBillingCompaniesConfig(
          showTestCompanies ? { include_unapproved: true } : {}
        ),
        fetchPlatformPricingGrids(),
        fetchPlatformBillingCreditor(),
      ]);
      setItems(listRes?.items || []);
      setGrids(gridRes?.items || []);
      setCreditor(credRes?.creditor || null);
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement');
    } finally {
      setLoading(false);
    }
  }, [showTestCompanies]);

  useEffect(() => {
    load();
  }, [load]);

  const visibleItems = useMemo(() => {
    const q = search.trim().toLowerCase();
    return items.filter((row) => {
      if (!showTestCompanies && isLikelyTestCompany(row.company_name)) return false;
      if (q && !(row.company_name || '').toLowerCase().includes(q)) return false;
      return true;
    });
  }, [items, search, showTestCompanies]);

  const buildSnapshot = (f, d, pct, month) =>
    JSON.stringify({ f, d, pct, month });

  const requestCloseModal = useCallback(() => {
    const dirty =
      Boolean(savedSnapshot) &&
      buildSnapshot(form, debtorForm, commissionPercent, effectiveMonth) !==
        savedSnapshot;
    if (dirty) {
      setActionDialog({
        title: 'Modifications non enregistrées',
        description:
          'Des modifications ne sont pas enregistrées. Abandonner les modifications ?',
        confirmationLabel: 'Abandonner les modifications',
        danger: true,
        onConfirm: async () => {
          setModalCompany(null);
          setModalError(null);
          setContracts([]);
          setReadiness(null);
          setActionDialog(null);
        },
      });
      return;
    }
    setModalCompany(null);
    setModalError(null);
    setContracts([]);
    setReadiness(null);
  }, [savedSnapshot, form, debtorForm, commissionPercent, effectiveMonth]);

  const closeModal = requestCloseModal;

  useEffect(() => {
    if (!modalCompany || actionDialog) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') requestCloseModal();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [modalCompany, actionDialog, requestCloseModal]);

  const openCompany = async (row) => {
    setModalCompany(row);
    setModalError(null);
    setSignedOn('');
    setModalTab('identity');
    const c = row.config;
    let nextForm = contractFromRow(c);
    // Entreprise sans config : défauts inactifs (emptyContract)
    if (!c) nextForm = { ...emptyContract };
    const pct = rateToPercent(nextForm.commission_rate);
    const month = isoToMonth(c?.effective_from);
    setForm(nextForm);
    setCommissionPercent(pct);
    setEffectiveMonth(month);
    setDebtorForm({
      ...emptyDebtor,
      legal_name: row.company_name || '',
    });
    try {
      const res = await fetchPlatformBillingContracts(row.company_id);
      const list = res?.contracts || [];
      setContracts(list);
      const active =
        list.find((x) => isContractOpen(x)) || list[0] || null;
      setSelectedContractId(active?.id ?? null);
      setReadiness(res?.readiness || null);
      setPartnerIdentity(res?.partner_identity || null);
      const addr = res?.debtor_address;
      const cf = res?.partner_identity?.company_fields || {};
      const nextDebtor = {
        legal_name: addr?.legal_name || row.company_name || '',
        street_name: addr?.street_name || '',
        building_number: addr?.building_number || '',
        postal_code: addr?.postal_code || '',
        city: addr?.city || '',
        country_code: addr?.country_code || 'CH',
        uid_ide: cf.uid_ide || '',
        legal_form: cf.legal_form || 'sarl',
        signatory_name: cf.signatory_name || '',
        signatory_title: cf.signatory_title || '',
      };
      if (active) {
        nextForm = contractFromRow(active);
        setForm(nextForm);
        setCommissionPercent(rateToPercent(active.commission_rate));
        setEffectiveMonth(
          active.effective_year && active.effective_month
            ? `${active.effective_year}-${String(active.effective_month).padStart(2, '0')}`
            : isoToMonth(active.effective_from)
        );
      }
      setDebtorForm(nextDebtor);
      const snapPct = active
        ? rateToPercent(active.commission_rate)
        : pct;
      const snapMonth = active
        ? active.effective_year && active.effective_month
          ? `${active.effective_year}-${String(active.effective_month).padStart(2, '0')}`
          : isoToMonth(active.effective_from)
        : month;
      setSavedSnapshot(buildSnapshot(nextForm, nextDebtor, snapPct, snapMonth));
      setInitialDebtorSnapshot(JSON.stringify(nextDebtor));
      setInitialCommercialSnapshot(
        JSON.stringify({ f: nextForm, pct: snapPct, month: snapMonth })
      );
    } catch (e) {
      setModalError(e?.message || 'Erreur contrats');
    }
  };

  const isDirty = useMemo(() => {
    if (!savedSnapshot) return false;
    return (
      buildSnapshot(form, debtorForm, commissionPercent, effectiveMonth) !==
      savedSnapshot
    );
  }, [form, debtorForm, commissionPercent, effectiveMonth, savedSnapshot]);

  const isDebtorDirty = useMemo(
    () =>
      Boolean(initialDebtorSnapshot) &&
      JSON.stringify(debtorForm) !== initialDebtorSnapshot,
    [debtorForm, initialDebtorSnapshot]
  );

  const isCommercialDirty = useMemo(
    () =>
      Boolean(initialCommercialSnapshot) &&
      JSON.stringify({ f: form, pct: commissionPercent, month: effectiveMonth }) !==
        initialCommercialSnapshot,
    [form, commissionPercent, effectiveMonth, initialCommercialSnapshot]
  );

  const activeContract = useMemo(
    () => contracts.find((c) => isContractOpen(c)) || null,
    [contracts]
  );

  const selectedContract = useMemo(
    () => contracts.find((c) => c.id === selectedContractId) || activeContract,
    [contracts, selectedContractId, activeContract]
  );

  const isHistoricalSelection = Boolean(
    selectedContract && !isContractOpen(selectedContract)
  );

  const formReadOnly = isHistoricalSelection;

  const activeAgreement = selectedContract?.active_agreement || null;
  const dunningReady = selectedContract?.dunning_automation_ready || null;

  const calcOk = Boolean(readiness?.contract_calculation_ready);
  const debtorOk = Boolean(readiness?.debtor_identity_ready);
  const creditorOk = Boolean(readiness?.creditor_qr_ready);
  const allReady = !readiness || (calcOk && debtorOk && creditorOk);

  const dunningPartialDay =
    Number(form.reminder_delay_days_after_due || 0) +
    Number(form.reminder_grace_days || 10);

  const selectContractVersion = (c) => {
    setSelectedContractId(c.id);
    setModalError(null);
    if (!isContractOpen(c) && activeContract && c.id !== activeContract.id) {
      // Historique : affichage lecture seule hydraté depuis la version
      const hist = contractFromRow(c);
      setForm(hist);
      setCommissionPercent(rateToPercent(c.commission_rate));
      setEffectiveMonth(
        c.effective_year && c.effective_month
          ? `${c.effective_year}-${String(c.effective_month).padStart(2, '0')}`
          : isoToMonth(c.effective_from)
      );
      return;
    }
    // Version active : formulaire d'édition (nouvelle version basée sur l'active)
    const next = contractFromRow(c);
    setForm(next);
    setCommissionPercent(rateToPercent(c.commission_rate));
    setEffectiveMonth(
      c.effective_year && c.effective_month
        ? `${c.effective_year}-${String(c.effective_month).padStart(2, '0')}`
        : isoToMonth(c.effective_from)
    );
  };

  const performSave = async ({ saveAddress, saveCommercial }) => {
    if (!modalCompany) return;
    setSaving(true);
    setModalError(null);
    try {
      if (saveAddress) {
        if (!debtorForm.street_name?.trim() || !debtorForm.postal_code?.trim()) {
          throw new Error(
            'Adresse incomplète : rue et NPA requis pour enregistrer l’adresse.'
          );
        }
        await putPlatformBillingDebtorAddress(modalCompany.company_id, debtorForm);
      }
      if (saveCommercial) {
        const now = new Date();
        const month =
          effectiveMonth ||
          `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
        const ym = monthToYearMonth(month);
        if (!ym) throw new Error('Mois d’effet invalide');
        const rate = percentToRate(commissionPercent);
        const created = await createPlatformBillingContract(modalCompany.company_id, {
          ...form,
          commission_rate: rate,
          commission_cancellation_policy: form.commission_cancellation_policy,
          free_license_max_months:
            form.subscription_pricing_mode === 'free'
              ? Number(form.free_license_max_months || 60)
              : null,
          statement_dispute_days: Number(form.statement_dispute_days || 10),
          custom_subscription_amount: form.custom_subscription_amount || null,
          support_hourly_rate_default: form.support_hourly_rate_default || null,
          payment_terms_days: form.payment_terms_days
            ? Number(form.payment_terms_days)
            : null,
          automated_dunning_enabled: !!form.automated_dunning_enabled,
          reminder_delay_days_after_due: Number(
            form.reminder_delay_days_after_due || 0
          ),
          reminder_grace_days: Number(form.reminder_grace_days || 10),
          full_suspend_days_after_due: Number(
            form.full_suspend_days_after_due || 30
          ),
          full_suspend_overdue_invoice_count: Number(
            form.full_suspend_overdue_invoice_count || 2
          ),
          termination_notice_days: Number(form.termination_notice_days || 10),
          partial_block_marketplace_offers: !!form.partial_block_marketplace_offers,
          partial_block_marketplace_acceptance:
            !!form.partial_block_marketplace_acceptance,
          partial_block_billable_support: !!form.partial_block_billable_support,
          partial_block_billable_configuration:
            !!form.partial_block_billable_configuration,
          effective_year: ym.effective_year,
          effective_month: ym.effective_month,
          auto_close_overlapping: true,
        });
        setEffectiveMonth(month);
        if (created?.contract?.id) {
          setSelectedContractId(created.contract.id);
        }
      }
      await openCompany(modalCompany);
      await load();
      setActionDialog(null);
    } catch (e) {
      setModalError(
        e?.response?.data?.message ||
          e?.response?.data?.error ||
          e?.message ||
          'Erreur sauvegarde'
      );
      setActionDialog(null);
    } finally {
      setSaving(false);
    }
  };

  const saveContract = async () => {
    if (!modalCompany || formReadOnly) return;
    const saveAddress = isDebtorDirty;
    const saveCommercial = isCommercialDirty || (!contracts.length && isDirty);
    if (!saveAddress && !saveCommercial) {
      setModalError('Aucune modification à enregistrer.');
      return;
    }
    const lines = [];
    if (saveAddress) lines.push('✓ modifier l’adresse de facturation');
    if (saveCommercial) {
      lines.push('✓ créer une nouvelle version commerciale');
      if (activeContract) {
        lines.push(
          `✓ clôturer la version active nº ${activeContract.id} à l’effet de la nouvelle version`
        );
      }
    }
    setActionDialog({
      title: 'Confirmer l’enregistrement',
      description: `Cette opération va :\n${lines.join('\n')}`,
      confirmationLabel: saveCommercial
        ? 'Créer la version / enregistrer'
        : 'Enregistrer l’adresse',
      onConfirm: async () => {
        await performSave({ saveAddress, saveCommercial });
      },
    });
  };

  const refreshContractsOnly = async () => {
    if (!modalCompany) return;
    const res = await fetchPlatformBillingContracts(modalCompany.company_id);
    setContracts(res?.contracts || []);
    setPartnerIdentity(res?.partner_identity || null);
    setReadiness(res?.readiness || null);
  };

  const buildSignatoryAuthorityVerification = () => {
    const identity = partnerIdentity?.partner || partnerIdentity || {};
    const signatoryName =
      debtorForm.signatory_name || identity.signatory_name || '';
    const signatoryFunction =
      debtorForm.signatory_title || identity.signatory_title || '';
    return {
      source: 'registre_du_commerce',
      register_name: rcRegisterName,
      checked_at: new Date().toISOString(),
      company_uid: debtorForm.uid_ide || identity.uid_ide || null,
      signatory_name: signatoryName,
      signatory_function: signatoryFunction,
      signature_mode: rcSignatureMode,
      co_signatory_name:
        rcSignatureMode === 'collective' ? rcCoSignatoryName : null,
      co_signatory_function:
        rcSignatureMode === 'collective' ? rcCoSignatoryFunction : null,
      attested: rcAttested,
    };
  };

  const onGenerateAgreement = async () => {
    if (!selectedContract || isDirty || formReadOnly) return;
    if (!rcAttested) {
      setModalError(
        'Attestez le pouvoir de signature sur la base du Registre du commerce.'
      );
      return;
    }
    setDocBusy(true);
    setModalError(null);
    try {
      await generatePartnerAgreement(
        selectedContract.id,
        buildSignatoryAuthorityVerification()
      );
      await refreshContractsOnly();
    } catch (e) {
      setModalError(
        e?.response?.data?.error || e?.message || 'Erreur génération contrat'
      );
    } finally {
      setDocBusy(false);
    }
  };

  const onMigrateAgreementV120 = async () => {
    if (!activeAgreement) return;
    if (!rcAttested) {
      setModalError(
        'Attestez le pouvoir de signature avant de migrer vers v1.20.'
      );
      return;
    }
    setDocBusy(true);
    setModalError(null);
    try {
      await migratePartnerAgreementToV120(
        activeAgreement.id,
        buildSignatoryAuthorityVerification()
      );
      await refreshContractsOnly();
    } catch (e) {
      setModalError(
        e?.response?.data?.error || e?.message || 'Erreur migration v1.20'
      );
    } finally {
      setDocBusy(false);
    }
  };

  const onMarkSent = async () => {
    if (!activeAgreement) return;
    const recipient =
      deliveryRecipient.trim() ||
      activeAgreement?.parties_snapshot?.partner?.contractual_email ||
      '';
    const ok = window.confirm(
      'Je confirme que ce dossier a été ou va immédiatement être remis au partenaire.\n\n' +
        `Canal : ${deliveryChannel}\n` +
        `Destinataire déclaré : ${recipient || '—'}`
    );
    if (!ok) return;
    setDocBusy(true);
    try {
      await markPartnerAgreementSent(activeAgreement.id, {
        confirmed: true,
        channel: deliveryChannel,
        recipient: recipient || null,
      });
      await refreshContractsOnly();
    } catch (e) {
      setModalError(e?.response?.data?.error || e?.message || 'Erreur envoi');
    } finally {
      setDocBusy(false);
    }
  };

  const onVoidAgreement = () => {
    if (!activeAgreement) return;
    setActionDialog({
      title: 'Annuler le document contractuel',
      description: 'Annuler le document partenaire actif.',
      confirmationLabel: 'Annuler le document',
      danger: true,
      reason: {
        required: true,
        label: 'Motif d’annulation',
        minLength: 3,
      },
      onConfirm: async ({ reason }) => {
        setDocBusy(true);
        try {
          await voidPartnerAgreement(activeAgreement.id, reason);
          await refreshContractsOnly();
          setActionDialog(null);
        } finally {
          setDocBusy(false);
        }
      },
    });
  };

  const onUploadSigned = async (file) => {
    if (!activeAgreement || !file || !signedOn) {
      setModalError('Date de signature et fichier PDF requis');
      return;
    }
    setDocBusy(true);
    try {
      await uploadPartnerAgreementSigned(activeAgreement.id, file, signedOn, {
        additionalPagesConfirmed: signedAdditionalPagesConfirmed,
      });
      await refreshContractsOnly();
      setSignedAdditionalPagesConfirmed(false);
    } catch (e) {
      setModalError(e?.response?.data?.error || e?.message || 'Erreur upload');
    } finally {
      setDocBusy(false);
    }
  };

  const closeActiveContract = () => {
    if (!activeContract) return;
    if (selectedContract?.id !== activeContract.id) return;
    const now = new Date();
    const defaultMonth = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
    const ym = monthToYearMonth(effectiveMonth || defaultMonth);
    setActionDialog({
      title: `Clôturer la version active nº ${activeContract.id}`,
      description: [
        `Version active nº ${activeContract.id}`,
        `Période actuelle : ${fmtPeriod(activeContract.effective_from, activeContract.effective_to)}`,
        ym
          ? `Fin prévue : ${String(ym.effective_month).padStart(2, '0')}/${ym.effective_year}`
          : 'Fin : début du mois courant (Zurich)',
        'Après cette date, aucune configuration de facturation ne sera applicable, sauf si une nouvelle version prend le relais.',
      ].join('\n'),
      confirmationLabel: 'Clôturer la version active',
      danger: true,
      onConfirm: async () => {
        setSaving(true);
        try {
          await closePlatformBillingContract(
            activeContract.id,
            ym
              ? {
                  effective_to_year: ym.effective_year,
                  effective_to_month: ym.effective_month,
                }
              : {}
          );
          await openCompany(modalCompany);
          setActionDialog(null);
        } catch (e) {
          setModalError(
            e?.response?.data?.message ||
              e?.response?.data?.error ||
              e?.message ||
              'Erreur clôture'
          );
        } finally {
          setSaving(false);
        }
      },
    });
  };

  const defaultGrid = grids.find((g) => g.grid_key === 'default') || grids[0];

  return (
    <div className={`${styles.layout} ${styles.layoutWithAside}`}>
      <div>
        <h2>Configuration commerciale transporteurs</h2>
        <p className={styles.lead}>
          Deux produits distincts : <strong>abonnement portefeuille</strong> (selon le volume) et{' '}
          <strong>commission LIRIE</strong> (sur les transports reçus via le réseau). Support
          optionnel. Les contrats sont versionnés dans le temps.
        </p>

        {error && <div className={styles.errorBanner}>{error}</div>}

        <div className={styles.toolbar}>
          <input
            type="search"
            className={styles.search}
            placeholder="Filtrer par nom d’entreprise…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Filtrer"
          />
          <label className={styles.muted} style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={showTestCompanies}
              onChange={(e) => setShowTestCompanies(e.target.checked)}
            />
            Inclure non approuvées / tests
          </label>
          <button type="button" className={styles.btn} onClick={() => load()} disabled={loading}>
            Actualiser
          </button>
        </div>

        {loading ? (
          <p className={styles.muted}>Chargement…</p>
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Entreprise</th>
                  <th>Abonnement portefeuille</th>
                  <th>Commission LIRIE</th>
                  <th>Configuration</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {visibleItems.length === 0 ? (
                  <tr>
                    <td colSpan={5} className={styles.muted}>
                      Aucune entreprise approuvée. Activez « Inclure non approuvées / tests » ou
                      élargissez le filtre.
                    </td>
                  </tr>
                ) : (
                  visibleItems.map((row) => {
                    const c = row.config;
                    return (
                      <tr key={row.company_id}>
                        <td>
                          <strong>{row.company_name}</strong>
                        </td>
                        <td>{portfolioLabel(c)}</td>
                        <td>{commissionLabel(c)}</td>
                        <td>
                          {configLabel(c) === 'Active' ? (
                            <span className={styles.badgeOn}>Active</span>
                          ) : (
                            <span className={styles.badgeOff}>{configLabel(c)}</span>
                          )}
                        </td>
                        <td>
                          <button
                            type="button"
                            className={styles.btn}
                            onClick={() => openCompany(row)}
                          >
                            {c ? 'Ouvrir' : 'Configurer'}
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <aside className={styles.aside}>
        <h3 className={styles.asideTitle}>Grille volume (globale)</h3>
        <p className={styles.asideLead}>
          Paliers d’abonnement portefeuille — sans mode dispatch.
        </p>
        {defaultGrid ? (
          <table className={styles.pricingTable}>
            <thead>
              <tr>
                <th>Volume mensuel</th>
                <th>Prix</th>
              </tr>
            </thead>
            <tbody>
              {(defaultGrid.tiers || []).map((t) => (
                <tr key={t.id}>
                  <td>
                    {t.volume_min}–{t.volume_max ?? '∞'}
                  </td>
                  <td>{fmtDec(t.price_monthly)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className={styles.muted}>Aucune grille chargée.</p>
        )}

        <h3 className={styles.asideTitle} style={{ marginTop: '1.25rem' }}>
          Créancier LIRIE
        </h3>
        {creditor ? (
          <>
            <p className={styles.asideLead}>
              {creditor.legal_name}
              <br />
              {[creditor.street_name, creditor.building_number].filter(Boolean).join(' ')}
              <br />
              {creditor.postal_code} {creditor.city}
              <br />
              IBAN : {creditor.iban || creditor.qr_iban || '—'}
              <br />
              TVA :{' '}
              {Number(creditor.default_tax_rate) === 0
                ? 'non applicable (franchise)'
                : `${creditor.default_tax_rate} %`}
            </p>
            <Link className={styles.btn} to={settingsPath}>
              Modifier dans Paramètres
            </Link>
          </>
        ) : (
          <p className={styles.muted}>
            Aucun créancier configuré.{' '}
            <Link to={settingsPath}>Configurer l’adresse et l’IBAN dans Paramètres</Link>.
          </p>
        )}
      </aside>

      {modalCompany && (
        <div
          className={styles.modalOverlay}
          role="presentation"
          onClick={(e) => {
            if (e.target === e.currentTarget) closeModal();
          }}
        >
          <div
            className={styles.modal}
            role="dialog"
            aria-modal="true"
            aria-labelledby="billing-contract-title"
          >
            <header className={styles.modalHeader}>
              <div className={styles.modalHeaderText}>
                <h2 id="billing-contract-title">{modalCompany.company_name}</h2>
                <p className={styles.modalSubtitle}>
                  Contrat commercial
                  {activeContract ? ` · version active nº ${activeContract.id}` : ''}
                  {formReadOnly ? ' · lecture seule' : ' · nouvelle version à l’enregistrement'}
                </p>
                <div className={styles.headerMeta}>
                  {form.is_billing_enabled ? (
                    <span className={styles.metaChipOn}>Facturation active</span>
                  ) : (
                    <span className={styles.metaChipOff}>Facturation inactive</span>
                  )}
                  {activeAgreement ? (
                    <span
                      className={`${styles.docBadge} ${styles[agreementStatusClass(activeAgreement.status)]}`}
                    >
                      {statusLabel(activeAgreement.status)}
                    </span>
                  ) : null}
                  {isDirty ? (
                    <span className={styles.metaChipWarn}>Non enregistré</span>
                  ) : null}
                </div>
              </div>
              <button
                type="button"
                className={styles.modalClose}
                onClick={closeModal}
                aria-label="Fermer"
              >
                ×
              </button>
            </header>

            <nav className={styles.tabs} aria-label="Sections du contrat">
              {CONTRACT_TABS.map((tab) => {
                const warn =
                  (tab.id === 'identity' && readiness && !debtorOk) ||
                  (tab.id === 'products' && readiness && !calcOk);
                return (
                  <button
                    key={tab.id}
                    type="button"
                    className={modalTab === tab.id ? styles.tabActive : styles.tab}
                    onClick={() => setModalTab(tab.id)}
                    aria-current={modalTab === tab.id ? 'page' : undefined}
                  >
                    {tab.label}
                    {warn ? <span className={styles.tabDot} aria-hidden /> : null}
                  </button>
                );
              })}
            </nav>

            <div className={styles.modalBody}>
              {formReadOnly ? (
                <div className={styles.bannerWarn} role="status">
                  <p className={styles.readinessHint}>
                    Version historique nº {selectedContract?.id} — lecture seule.
                    Revenez à la version active nº {activeContract?.id} pour créer
                    une nouvelle version commerciale.
                  </p>
                  {activeContract ? (
                    <button
                      type="button"
                      className={styles.btn}
                      onClick={() => {
                        selectContractVersion(activeContract);
                        setModalTab('products');
                      }}
                    >
                      Revenir à la version active
                    </button>
                  ) : null}
                </div>
              ) : null}

              {!allReady ? (
                <section className={styles.readinessBlock} aria-label="À compléter">
                  <ul className={styles.readinessList}>
                    <ReadinessItem
                      ok={calcOk}
                      label="Calcul du relevé"
                      hint="Activez au moins un produit puis enregistrez."
                      errors={readiness?.contract_calculation_errors}
                    />
                    <ReadinessItem
                      ok={debtorOk}
                      label="Adresse de facturation"
                      hint="Raison sociale, rue, NPA et localité requis."
                      errors={readiness?.debtor_identity_errors}
                    />
                    <ReadinessItem
                      ok={creditorOk}
                      label="Créancier LIRIE"
                      hint="IBAN et adresse LIRIE dans Paramètres."
                      errors={readiness?.creditor_errors}
                    />
                  </ul>
                  <div className={styles.readinessActions}>
                    {!debtorOk ? (
                      <button
                        type="button"
                        className={`${styles.btn} ${styles.btnGhost}`}
                        onClick={() => setModalTab('identity')}
                      >
                        Compléter l’identité
                      </button>
                    ) : null}
                    {!calcOk ? (
                      <button
                        type="button"
                        className={`${styles.btn} ${styles.btnGhost}`}
                        onClick={() => setModalTab('products')}
                      >
                        Configurer les produits
                      </button>
                    ) : null}
                    {!creditorOk ? (
                      <Link className={styles.btn} to={settingsPath}>
                        Créancier LIRIE
                      </Link>
                    ) : null}
                  </div>
                </section>
              ) : null}

              {modalError && (
                <div className={styles.errorBanner} role="alert">
                  {modalError}
                </div>
              )}

              {modalTab === 'identity' ? (
              <div
                className={`${styles.tabPanel}${formReadOnly ? ` ${styles.tabPanelReadonly}` : ''}`}
                inert={formReadOnly || undefined}
              >
              <section className={styles.formSection}>
                <h3 className={styles.formSectionTitle}>Adresse de facturation</h3>
                <p className={styles.formSectionLead}>
                  Débiteur sur la QR-facture émise par LIRIE.
                </p>
                <div className={styles.formGridTwo}>
                  <label className={styles.field}>
                    Raison sociale
                    <input
                      value={debtorForm.legal_name}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, legal_name: e.target.value }))
                      }
                    />
                  </label>
                  <label className={styles.field}>
                    Pays
                    <input
                      value={debtorForm.country_code}
                      maxLength={2}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, country_code: e.target.value }))
                      }
                    />
                  </label>
                </div>
                <div className={`${styles.formGridTwo} ${styles.addressRow}`}>
                  <label className={`${styles.field} ${styles.fieldGrow}`}>
                    Rue
                    <input
                      value={debtorForm.street_name}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, street_name: e.target.value }))
                      }
                    />
                  </label>
                  <label className={`${styles.field} ${styles.fieldNarrow}`}>
                    N°
                    <input
                      value={debtorForm.building_number}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, building_number: e.target.value }))
                      }
                    />
                  </label>
                </div>
                <div className={styles.formGridTwo}>
                  <label className={styles.field}>
                    NPA
                    <input
                      value={debtorForm.postal_code}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, postal_code: e.target.value }))
                      }
                    />
                  </label>
                  <label className={styles.field}>
                    Localité
                    <input
                      value={debtorForm.city}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, city: e.target.value }))
                      }
                    />
                  </label>
                </div>
                <div className={styles.formGridTwo}>
                  <label className={styles.field}>
                    IDE
                    <input
                      value={debtorForm.uid_ide}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, uid_ide: e.target.value }))
                      }
                      placeholder="CHE-XXX.XXX.XXX"
                    />
                  </label>
                  <label className={styles.field}>
                    Forme juridique
                    <select
                      value={debtorForm.legal_form}
                      onChange={(e) =>
                        setDebtorForm((f) => ({ ...f, legal_form: e.target.value }))
                      }
                    >
                      {LEGAL_FORMS.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className={styles.formGridTwo}>
                  <label className={styles.field}>
                    Représentant / signataire
                    <input
                      value={debtorForm.signatory_name}
                      onChange={(e) =>
                        setDebtorForm((f) => ({
                          ...f,
                          signatory_name: e.target.value,
                        }))
                      }
                    />
                  </label>
                  <label className={styles.field}>
                    Titre / pouvoir de signature
                    <input
                      value={debtorForm.signatory_title}
                      onChange={(e) =>
                        setDebtorForm((f) => ({
                          ...f,
                          signatory_title: e.target.value,
                        }))
                      }
                      placeholder="associé-gérant, avec signature individuelle"
                    />
                  </label>
                </div>
                {partnerIdentity?.divergence_warnings?.length ? (
                  <p className={styles.readinessHint}>
                    Divergence profil / entreprise détectée (
                    {partnerIdentity.divergence_warnings.join(', ')}). La génération
                    utilisera un seul bloc d’identité, sans mélange.
                  </p>
                ) : null}
                {partnerIdentity && !partnerIdentity.is_complete ? (
                  <p className={styles.readinessHint}>
                    Identité contractuelle incomplète :{' '}
                    {(partnerIdentity.missing_fields || []).join(', ')}
                  </p>
                ) : null}
              </section>
              </div>
              ) : null}

              {modalTab === 'products' ? (
              <div
                className={`${styles.tabPanel}${formReadOnly ? ` ${styles.tabPanelReadonly}` : ''}`}
                inert={formReadOnly || undefined}
              >
              <section className={`${styles.productCard} ${form.is_billing_enabled ? styles.productCardOn : ''}`}>
                <label className={styles.productToggle}>
                  <input
                    type="checkbox"
                    checked={form.is_billing_enabled}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, is_billing_enabled: e.target.checked }))
                    }
                  />
                  <span>
                    <strong>Facturation active</strong>
                    <small>Sans activation, aucun relevé n’est généré.</small>
                  </span>
                </label>
              </section>

              <div className={styles.productGrid}>
              <section className={`${styles.productCard} ${form.own_portfolio_billing_enabled ? styles.productCardOn : ''}`}>
                <label className={styles.productToggle}>
                  <input
                    type="checkbox"
                    checked={form.own_portfolio_billing_enabled}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        own_portfolio_billing_enabled: e.target.checked,
                      }))
                    }
                  />
                  <span>
                    <strong>Abonnement portefeuille</strong>
                    <small>Volume mensuel — paliers 79 / 149 / 249 CHF</small>
                  </span>
                </label>
                {form.own_portfolio_billing_enabled ? (
                  <div className={styles.productBody}>
                    <label className={styles.field}>
                      Tarification
                      <select
                        value={form.subscription_pricing_mode}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            subscription_pricing_mode: e.target.value,
                          }))
                        }
                      >
                        <option value="volume">Selon le volume</option>
                        <option value="fixed">Montant fixe</option>
                        <option value="free">Gratuit</option>
                      </select>
                    </label>
                    {form.subscription_pricing_mode === 'fixed' ? (
                      <label className={styles.field}>
                        Montant mensuel
                        <div className={styles.inputWithSuffix}>
                          <input
                            inputMode="decimal"
                            value={form.custom_subscription_amount}
                            onChange={(e) =>
                              setForm((f) => ({
                                ...f,
                                custom_subscription_amount: e.target.value,
                              }))
                            }
                          />
                          <span className={styles.inputSuffix}>CHF</span>
                        </div>
                      </label>
                    ) : null}
                    {form.subscription_pricing_mode === 'free' ? (
                      <label className={styles.field}>
                        Durée max. gratuité
                        <div className={styles.inputWithSuffix}>
                          <input
                            inputMode="numeric"
                            value={form.free_license_max_months}
                            onChange={(e) =>
                              setForm((f) => ({
                                ...f,
                                free_license_max_months: e.target.value,
                              }))
                            }
                          />
                          <span className={styles.inputSuffix}>mois</span>
                        </div>
                      </label>
                    ) : null}
                  </div>
                ) : null}
              </section>

              <section className={`${styles.productCard} ${form.lirie_commission_enabled ? styles.productCardOn : ''}`}>
                <label className={styles.productToggle}>
                  <input
                    type="checkbox"
                    checked={form.lirie_commission_enabled}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        lirie_commission_enabled: e.target.checked,
                      }))
                    }
                  />
                  <span>
                    <strong>Commission LIRIE</strong>
                    <small>Sur les transports marketplace terminés</small>
                  </span>
                </label>
                {form.lirie_commission_enabled ? (
                  <div className={styles.productBody}>
                    <label className={styles.field}>
                      Taux
                      <div className={styles.inputWithSuffix}>
                        <input
                          inputMode="decimal"
                          value={commissionPercent}
                          onChange={(e) => setCommissionPercent(e.target.value)}
                        />
                        <span className={styles.inputSuffix}>%</span>
                      </div>
                    </label>
                    <label className={styles.field}>
                      Politique d’annulation
                      <select
                        value={form.commission_cancellation_policy}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            commission_cancellation_policy: e.target.value,
                          }))
                        }
                      >
                        {CANCEL_POLICIES.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                ) : null}
              </section>

              <section className={`${styles.productCard} ${form.support_enabled ? styles.productCardOn : ''}`}>
                <label className={styles.productToggle}>
                  <input
                    type="checkbox"
                    checked={form.support_enabled}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, support_enabled: e.target.checked }))
                    }
                  />
                  <span>
                    <strong>Support</strong>
                    <small>Heures saisies sur le relevé</small>
                  </span>
                </label>
                {form.support_enabled ? (
                  <div className={styles.productBody}>
                    <label className={styles.field}>
                      Tarif horaire
                      <div className={styles.inputWithSuffix}>
                        <input
                          inputMode="decimal"
                          value={form.support_hourly_rate_default}
                          onChange={(e) =>
                            setForm((f) => ({
                              ...f,
                              support_hourly_rate_default: e.target.value,
                            }))
                          }
                        />
                        <span className={styles.inputSuffix}>CHF/h</span>
                      </div>
                    </label>
                  </div>
                ) : null}
              </section>
              </div>

              <section className={styles.formSection}>
                <h3 className={styles.formSectionTitle}>Conditions</h3>
                <p className={styles.formSectionLead}>
                  Mois d’effet de la nouvelle version et délais contractuels.
                </p>
                <div className={styles.formGridTwo}>
                  <label className={styles.field}>
                    Mois d’effet
                    <input
                      type="month"
                      value={effectiveMonth}
                      onChange={(e) => setEffectiveMonth(e.target.value)}
                    />
                  </label>
                  <label className={styles.field}>
                    Délai de paiement
                    <div className={styles.inputWithSuffix}>
                      <input
                        inputMode="numeric"
                        value={form.payment_terms_days}
                        onChange={(e) =>
                          setForm((f) => ({ ...f, payment_terms_days: e.target.value }))
                        }
                      />
                      <span className={styles.inputSuffix}>jours</span>
                    </div>
                  </label>
                </div>
                <div className={styles.formGridTwo}>
                  <label className={styles.field}>
                    Délai de contestation relevé
                    <div className={styles.inputWithSuffix}>
                      <input
                        inputMode="numeric"
                        value={form.statement_dispute_days}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            statement_dispute_days: e.target.value,
                          }))
                        }
                      />
                      <span className={styles.inputSuffix}>jours</span>
                    </div>
                  </label>
                </div>
              </section>
              </div>
              ) : null}

              {modalTab === 'dunning' ? (
              <div
                className={`${styles.tabPanel}${formReadOnly ? ` ${styles.tabPanelReadonly}` : ''}`}
                inert={formReadOnly || undefined}
              >
              <section className={styles.formSection}>
                <h3 className={styles.formSectionTitle}>
                  Défaut de paiement (art. 6 bis)
                </h3>
                <p className={styles.formSectionLead}>
                  Rappel, suspension partielle Marketplace / support, puis
                  restriction commerciale complète. Les courses déjà engagées,
                  GPS, factures, paiement et export restent toujours disponibles.
                </p>
                <div className={styles.dunningStatusRow}>
                  <span
                    className={
                      form.automated_dunning_enabled
                        ? styles.metaChipOn
                        : styles.metaChipOff
                    }
                  >
                    Config. {form.automated_dunning_enabled ? 'activée' : 'désactivée'}
                  </span>
                  <span
                    className={
                      dunningReady?.ready ? styles.metaChipOn : styles.metaChipWarn
                    }
                  >
                    Exécution{' '}
                    {dunningReady?.ready
                      ? 'autorisée'
                      : `inactive${
                          (dunningReady?.reasons || []).length
                            ? ` — ${(dunningReady.reasons || []).join(' · ')}`
                            : ''
                        }`}
                  </span>
                </div>
                <label className={styles.productToggle}>
                  <input
                    type="checkbox"
                    checked={!!form.automated_dunning_enabled}
                    disabled={formReadOnly}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        automated_dunning_enabled: e.target.checked,
                      }))
                    }
                  />
                  <span>
                    <strong>Mesures automatisées</strong>
                    <small>Rappel, suspension partielle puis complète</small>
                  </span>
                </label>
                {form.automated_dunning_enabled ? (
                  <>
                    <ol className={styles.dunningTimeline} aria-label="Calendrier de recouvrement">
                      <li>
                        <span className={styles.dunningStepDay}>
                          J{form.reminder_delay_days_after_due || 0}
                        </span>
                        <span className={styles.dunningStepLabel}>Rappel</span>
                      </li>
                      <li>
                        <span className={styles.dunningStepDay}>J{dunningPartialDay}</span>
                        <span className={styles.dunningStepLabel}>Suspension partielle</span>
                      </li>
                      <li>
                        <span className={styles.dunningStepDay}>
                          J{form.full_suspend_days_after_due || 30}
                        </span>
                        <span className={styles.dunningStepLabel}>
                          Suspension complète
                          <small>
                            ou {form.full_suspend_overdue_invoice_count || 2} factures
                            échues
                          </small>
                        </span>
                      </li>
                    </ol>
                    <div className={styles.formGridTwo}>
                      <label className={styles.field}>
                        Délai rappel après échéance
                        <div className={styles.inputWithSuffix}>
                          <input
                            inputMode="numeric"
                            value={form.reminder_delay_days_after_due}
                            onChange={(e) =>
                              setForm((f) => ({
                                ...f,
                                reminder_delay_days_after_due: e.target.value,
                              }))
                            }
                          />
                          <span className={styles.inputSuffix}>jours</span>
                        </div>
                      </label>
                      <label className={styles.field}>
                        Grâce après rappel
                        <div className={styles.inputWithSuffix}>
                          <input
                            inputMode="numeric"
                            value={form.reminder_grace_days}
                            onChange={(e) =>
                              setForm((f) => ({
                                ...f,
                                reminder_grace_days: e.target.value,
                              }))
                            }
                          />
                          <span className={styles.inputSuffix}>jours</span>
                        </div>
                      </label>
                    </div>
                    <div className={styles.formGridTwo}>
                      <label className={styles.field}>
                        Suspension complète après
                        <div className={styles.inputWithSuffix}>
                          <input
                            inputMode="numeric"
                            value={form.full_suspend_days_after_due}
                            onChange={(e) =>
                              setForm((f) => ({
                                ...f,
                                full_suspend_days_after_due: e.target.value,
                              }))
                            }
                          />
                          <span className={styles.inputSuffix}>jours</span>
                        </div>
                      </label>
                      <label className={styles.field}>
                        Ou nb factures échues
                        <input
                          inputMode="numeric"
                          value={form.full_suspend_overdue_invoice_count}
                          onChange={(e) =>
                            setForm((f) => ({
                              ...f,
                              full_suspend_overdue_invoice_count: e.target.value,
                            }))
                          }
                        />
                      </label>
                    </div>
                    <div className={styles.formGridTwo}>
                      <label className={styles.field}>
                        Mise en demeure finale
                        <div className={styles.inputWithSuffix}>
                          <input
                            inputMode="numeric"
                            value={form.termination_notice_days}
                            onChange={(e) =>
                              setForm((f) => ({
                                ...f,
                                termination_notice_days: e.target.value,
                              }))
                            }
                          />
                          <span className={styles.inputSuffix}>jours</span>
                        </div>
                      </label>
                    </div>
                    <h4 className={styles.subSectionTitle}>Blocages en suspension partielle</h4>
                    <div className={styles.blockGrid}>
                    <label className={styles.productToggle}>
                      <input
                        type="checkbox"
                        checked={!!form.partial_block_marketplace_offers}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            partial_block_marketplace_offers: e.target.checked,
                          }))
                        }
                      />
                      Nouvelles offres Marketplace
                    </label>
                    <label className={styles.productToggle}>
                      <input
                        type="checkbox"
                        checked={!!form.partial_block_marketplace_acceptance}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            partial_block_marketplace_acceptance: e.target.checked,
                          }))
                        }
                      />
                      Acceptation d’offres Marketplace
                    </label>
                    <label className={styles.productToggle}>
                      <input
                        type="checkbox"
                        checked={!!form.partial_block_billable_support}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            partial_block_billable_support: e.target.checked,
                          }))
                        }
                      />
                      Support facturable
                    </label>
                    <label className={styles.productToggle}>
                      <input
                        type="checkbox"
                        checked={!!form.partial_block_billable_configuration}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            partial_block_billable_configuration: e.target.checked,
                          }))
                        }
                      />
                      Configuration facturable
                    </label>
                    </div>
                  </>
                ) : (
                  <p className={styles.readinessHint}>
                    Automation désactivée : LIRIE conserve ses droits de
                    recouvrement manuel ; aucune suspension automatique.
                  </p>
                )}
              </section>
              </div>
              ) : null}

              {modalTab === 'document' ? (
              <div className={styles.tabPanel}>
              {selectedContract ? (
                <section className={`${styles.formSection} ${styles.docPanel}`}>
                  <div className={styles.docHeader}>
                    <div className={styles.docHeaderMain}>
                      <h3 className={styles.formSectionTitle}>
                        Contrat partenaire
                      </h3>
                      {activeAgreement?.reference ? (
                        <p className={styles.docReference}>
                          {activeAgreement.reference}
                        </p>
                      ) : (
                        <p className={styles.formSectionLead}>
                          Aucun document généré pour cette version
                        </p>
                      )}
                    </div>
                    <span
                      className={`${styles.docStatus} ${styles[agreementStatusClass(activeAgreement?.status)]}`}
                    >
                      {statusLabel(activeAgreement?.status)}
                    </span>
                  </div>

                  <dl className={styles.docMeta}>
                    <div>
                      <dt>Version commerciale</dt>
                      <dd>n°{selectedContract.id}</dd>
                    </div>
                    <div>
                      <dt>Période</dt>
                      <dd>
                        {fmtPeriod(
                          selectedContract.effective_from,
                          selectedContract.effective_to
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt>Commission</dt>
                      <dd>
                        {form.lirie_commission_enabled
                          ? `${commissionPercent || '—'} %`
                          : 'désactivée'}
                      </dd>
                    </div>
                    <div>
                      <dt>Licence</dt>
                      <dd>
                        {form.subscription_pricing_mode === 'free'
                          ? `Gratuit · ${form.free_license_max_months || 60} mois`
                          : form.subscription_pricing_mode || '—'}
                      </dd>
                    </div>
                    <div>
                      <dt>Support</dt>
                      <dd>
                        {form.support_enabled
                          ? `${form.support_hourly_rate_default || '—'} CHF/h`
                          : 'désactivé'}
                      </dd>
                    </div>
                    <div>
                      <dt>Effet</dt>
                      <dd>{effectiveMonth || '—'}</dd>
                    </div>
                  </dl>
                  {selectedContract.commercially_frozen ? (
                    <p className={styles.docNote}>
                      Conditions commerciales gelées (accord envoyé ou signé).
                    </p>
                  ) : null}

                  <div className={styles.docBlock}>
                    <h4 className={styles.docBlockTitle}>Conditions &amp; pouvoir</h4>
                    <label className={styles.field}>
                      Conditions particulières
                      <textarea
                        rows={2}
                        value={form.contract_special_conditions || ''}
                        disabled={formReadOnly}
                        onChange={(e) =>
                          setForm((f) => ({
                            ...f,
                            contract_special_conditions: e.target.value,
                          }))
                        }
                        placeholder="Texte optionnel — distinct des notes internes"
                      />
                    </label>
                    <div className={styles.formGridTwo}>
                      <label className={styles.field}>
                        Mode de signature
                        <select
                          value={rcSignatureMode}
                          onChange={(e) => setRcSignatureMode(e.target.value)}
                        >
                          <option value="individual">Individuelle</option>
                          <option value="collective">Collective à deux</option>
                        </select>
                      </label>
                      <label className={styles.field}>
                        Registre consulté
                        <input
                          type="text"
                          value={rcRegisterName}
                          onChange={(e) => setRcRegisterName(e.target.value)}
                        />
                      </label>
                    </div>
                    {rcSignatureMode === 'collective' ? (
                      <div className={styles.formGridTwo}>
                        <label className={styles.field}>
                          Co-signataire (nom)
                          <input
                            type="text"
                            value={rcCoSignatoryName}
                            onChange={(e) =>
                              setRcCoSignatoryName(e.target.value)
                            }
                          />
                        </label>
                        <label className={styles.field}>
                          Co-signataire (fonction)
                          <input
                            type="text"
                            value={rcCoSignatoryFunction}
                            onChange={(e) =>
                              setRcCoSignatoryFunction(e.target.value)
                            }
                          />
                        </label>
                      </div>
                    ) : null}
                    <label className={styles.checkRow}>
                      <input
                        type="checkbox"
                        checked={rcAttested}
                        onChange={(e) => setRcAttested(e.target.checked)}
                      />
                      Pouvoir de signature vérifié au Registre du commerce
                    </label>
                  </div>

                  {isDirty ? (
                    <p className={styles.docNote}>
                      Enregistrez d’abord la version commerciale avant de générer
                      le contrat.
                    </p>
                  ) : null}

                  <div className={styles.docBlock}>
                    <h4 className={styles.docBlockTitle}>1. Génération</h4>
                    <div className={styles.docActions}>
                      <button
                        type="button"
                        className={`${styles.btn} ${styles.btnPrimary}`}
                        disabled={
                          docBusy ||
                          isDirty ||
                          !selectedContract ||
                          formReadOnly ||
                          !rcAttested
                        }
                        onClick={onGenerateAgreement}
                      >
                        Générer
                      </button>
                      {activeAgreement?.needs_v120_migration ? (
                        <button
                          type="button"
                          className={styles.btn}
                          disabled={docBusy || !rcAttested}
                          onClick={onMigrateAgreementV120}
                        >
                          Migrer vers le pack
                        </button>
                      ) : null}
                      <button
                        type="button"
                        className={styles.btn}
                        disabled={
                          docBusy ||
                          !activeAgreement?.has_generated_particular_pdf ||
                          activeAgreement?.status !== 'draft'
                        }
                        onClick={() =>
                          downloadPartnerAgreementFile(
                            downloadPartnerAgreementPreviewUrl(
                              activeAgreement.id
                            ),
                            `${(activeAgreement.reference || 'contrat').replaceAll('/', '_')}_BROUILLON.pdf`
                          )
                        }
                      >
                        Prévisualiser
                      </button>
                      <button
                        type="button"
                        className={`${styles.btn} ${styles.btnGhost}`}
                        disabled={
                          docBusy ||
                          !(
                            activeAgreement?.has_internal_docx ||
                            activeAgreement?.has_generated_docx
                          )
                        }
                        onClick={() =>
                          downloadPartnerAgreementFile(
                            downloadPartnerAgreementDocxUrl(activeAgreement.id),
                            `${(activeAgreement.reference || 'contrat').replaceAll('/', '_')}_interne.docx`
                          )
                        }
                      >
                        DOCX interne
                      </button>
                    </div>
                  </div>

                  <div className={styles.docBlock}>
                    <h4 className={styles.docBlockTitle}>2. Remise</h4>
                    {activeAgreement?.status === 'draft' ? (
                      <div className={styles.formGridTwo}>
                        <label className={styles.field}>
                          Canal
                          <select
                            value={deliveryChannel}
                            onChange={(e) => setDeliveryChannel(e.target.value)}
                          >
                            <option value="email">E-mail</option>
                            <option value="hand_delivery">
                              Remise en main propre
                            </option>
                            <option value="other">Autre</option>
                          </select>
                        </label>
                        <label className={styles.field}>
                          Destinataire
                          <input
                            type="text"
                            value={deliveryRecipient}
                            onChange={(e) =>
                              setDeliveryRecipient(e.target.value)
                            }
                            placeholder="ex. contact@partenaire.ch"
                          />
                        </label>
                      </div>
                    ) : null}
                    <div className={styles.docActions}>
                      <button
                        type="button"
                        className={`${styles.btn} ${styles.btnPrimary}`}
                        disabled={
                          docBusy ||
                          !activeAgreement ||
                          activeAgreement.status !== 'draft'
                        }
                        onClick={onMarkSent}
                      >
                        Marquer envoyé
                      </button>
                      <button
                        type="button"
                        className={styles.btn}
                        disabled={
                          docBusy ||
                          !activeAgreement?.particular_pdf_available_for_signature
                        }
                        onClick={() =>
                          downloadPartnerAgreementFile(
                            downloadPartnerAgreementParticularPdfUrl(
                              activeAgreement.id
                            ),
                            `${(activeAgreement.reference || 'contrat').replaceAll('/', '_')}_contrat-particulier.pdf`
                          )
                        }
                      >
                        PDF à signer
                      </button>
                      <button
                        type="button"
                        className={styles.btn}
                        disabled={
                          docBusy || !activeAgreement?.has_delivery_package
                        }
                        onClick={() =>
                          downloadPartnerAgreementFile(
                            downloadPartnerAgreementPackageUrl(
                              activeAgreement.id
                            ),
                            `${(activeAgreement.reference || 'dossier').replaceAll('/', '_')}_Dossier-remise.zip`
                          )
                        }
                      >
                        Dossier ZIP
                      </button>
                      <button
                        type="button"
                        className={`${styles.btn} ${styles.btnGhost}`}
                        disabled={
                          docBusy ||
                          !activeAgreement ||
                          !['draft', 'sent'].includes(activeAgreement.status)
                        }
                        onClick={onVoidAgreement}
                      >
                        Annuler
                      </button>
                    </div>
                    {activeAgreement?.status === 'draft' ? (
                      <p className={styles.docNote}>
                        Le PDF à signer et le dossier ZIP s’activent après «
                        Marquer envoyé ».
                      </p>
                    ) : null}
                  </div>

                  {(activeAgreement?.status === 'sent' ||
                    activeAgreement?.has_signed_pdf) ? (
                    <div className={styles.docBlock}>
                      <h4 className={styles.docBlockTitle}>3. Signature</h4>
                      {activeAgreement?.status === 'sent' ? (
                        <>
                          <div className={styles.formGridTwo}>
                            <label className={styles.field}>
                              Date de signature
                              <input
                                type="date"
                                value={signedOn}
                                onChange={(e) => setSignedOn(e.target.value)}
                              />
                            </label>
                            <label className={styles.field}>
                              PDF signé
                              <input
                                type="file"
                                accept="application/pdf,.pdf"
                                disabled={docBusy || !signedOn}
                                onChange={(e) => {
                                  const file = e.target.files?.[0];
                                  if (file) onUploadSigned(file);
                                  e.target.value = '';
                                }}
                              />
                            </label>
                          </div>
                          <label className={styles.checkRow}>
                            <input
                              type="checkbox"
                              checked={signedAdditionalPagesConfirmed}
                              onChange={(e) =>
                                setSignedAdditionalPagesConfirmed(
                                  e.target.checked
                                )
                              }
                            />
                            Pages supplémentaires = certificat / journal de
                            signature
                          </label>
                        </>
                      ) : null}
                      {activeAgreement?.has_signed_pdf ? (
                        <div className={styles.docActions}>
                          <button
                            type="button"
                            className={styles.btn}
                            onClick={() =>
                              downloadPartnerAgreementFile(
                                downloadPartnerAgreementSignedUrl(
                                  activeAgreement.id
                                ),
                                activeAgreement.signed_original_filename ||
                                  'contrat-signe.pdf'
                              )
                            }
                          >
                            Télécharger le PDF signé
                          </button>
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  <p className={styles.docFootnote}>
                    Validation juridique externe recommandée avant signature.
                  </p>
                </section>
              ) : (
                <p className={styles.docNote}>
                  Enregistrez une première version commerciale pour générer le
                  contrat.
                </p>
              )}

              {contracts.length > 0 ? (
                <section className={`${styles.formSection} ${styles.docVersions}`}>
                  <h3 className={styles.formSectionTitle}>
                    Historique commercial
                  </h3>
                  <ul className={styles.versionsList}>
                    {contracts.map((c) => {
                      const selected = c.id === selectedContract?.id;
                      const open = isContractOpen(c);
                      return (
                        <li key={c.id}>
                          <button
                            type="button"
                            className={`${styles.versionBtn} ${
                              selected ? styles.versionBtnActive : ''
                            }`}
                            onClick={() => selectContractVersion(c)}
                          >
                            <span className={styles.versionId}>#{c.id}</span>
                            <span className={styles.versionMeta}>
                              {fmtPeriod(c.effective_from, c.effective_to)}
                              {open ? ' · active' : ' · remplacée'}
                            </span>
                            <span className={styles.versionProducts}>
                              {portfolioLabel(c)} · {commissionLabel(c)}
                              {c.active_agreement
                                ? ` · ${statusLabel(c.active_agreement.status)}`
                                : ''}
                            </span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </section>
              ) : null}
              </div>
              ) : null}
            </div>

            <footer className={styles.modalFooter}>
              {activeContract &&
              selectedContract?.id === activeContract.id &&
              !formReadOnly ? (
                <button
                  type="button"
                  className={`${styles.btn} ${styles.btnGhost}`}
                  onClick={closeActiveContract}
                  disabled={saving}
                >
                  {`Clôturer nº ${activeContract.id}`}
                </button>
              ) : (
                <span />
              )}
              <div className={styles.footerRight}>
                <button
                  type="button"
                  className={`${styles.btn} ${styles.btnGhost}`}
                  onClick={closeModal}
                  disabled={saving}
                >
                  Annuler
                </button>
                <button
                  type="button"
                  className={`${styles.btn} ${styles.btnPrimary}`}
                  onClick={saveContract}
                  disabled={saving || formReadOnly}
                  title={
                    formReadOnly
                      ? 'Version historique en lecture seule — revenez à la version active pour éditer'
                      : undefined
                  }
                >
                  {saving
                    ? 'Enregistrement…'
                    : formReadOnly
                      ? 'Lecture seule'
                      : 'Enregistrer'}
                </button>
              </div>
            </footer>
          </div>
        </div>
      )}

      {actionDialog ? (
        <AdminActionDialog
          open
          title={actionDialog.title}
          description={actionDialog.description}
          confirmationLabel={actionDialog.confirmationLabel}
          reason={actionDialog.reason}
          danger={Boolean(actionDialog.danger)}
          loading={saving || docBusy}
          onConfirm={actionDialog.onConfirm}
          onClose={() => setActionDialog(null)}
        />
      ) : null}
    </div>
  );
};

export default AdminBillingDualProductConfig;
