import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link, useParams } from 'react-router-dom';
import { FiExternalLink, FiX } from 'react-icons/fi';
import {
  fetchPlatformBillingCompanyConfig,
  fetchPlatformBillingContracts,
} from '../../../services/adminService';
import { adminPaths } from '../routing/adminRoutePaths';
import styles from './AdminCompanyDetailDrawer.module.css';

const BILLING_ACCESS_LABELS = {
  active: 'Aucune restriction',
  partial: 'Restriction partielle',
  full: 'Restriction complète',
};

const formatBillingAccessLabel = (state) =>
  BILLING_ACCESS_LABELS[String(state || 'active').toLowerCase()] ||
  'Aucune restriction';

const formatPricingMode = (mode) => {
  const key = String(mode || '').toLowerCase();
  if (key === 'fixed') return 'Montant fixe';
  if (key === 'free') return 'Gratuit';
  if (key === 'volume') return 'Selon volume';
  return mode || '—';
};

const formatPercent = (rate) => {
  if (rate == null || rate === '') return '—';
  const n = Number(rate);
  if (Number.isNaN(n)) return '—';
  return `${(n * 100).toFixed(n >= 0.01 ? 1 : 2)} %`;
};

const formatMoney = (amount) => {
  if (amount == null || amount === '') return '—';
  const n = Number(amount);
  if (Number.isNaN(n)) return String(amount);
  return new Intl.NumberFormat('fr-CH', {
    style: 'currency',
    currency: 'CHF',
  }).format(n);
};

const formatDate = (value) => {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleDateString('fr-CH');
};

const billingBadgeClass = (state) => {
  const key = String(state || 'active').toLowerCase();
  if (key === 'partial') return styles.badgePartial;
  if (key === 'full') return styles.badgeFull;
  return styles.badgeActive;
};

/**
 * Panneau latéral : détail entreprise (identité, abonnement, contrats, restriction commerciale).
 */
const AdminCompanyDetailDrawer = ({
  user,
  isOpen,
  onClose,
  onSetBillingAccess,
  onPauseDunning,
}) => {
  const { public_id: adminId } = useParams();
  const drawerRef = useRef(null);
  const overlayRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [config, setConfig] = useState(null);
  const [contractsPayload, setContractsPayload] = useState(null);

  const companyId = user?.company_id;
  const billingState = String(user?.platform_billing_access_state || 'active').toLowerCase();
  const dunningPaused = Boolean(user?.dunning_paused_until);

  useEffect(() => {
    if (!isOpen || !companyId) return undefined;

    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [cfgRes, contractsRes] = await Promise.all([
          fetchPlatformBillingCompanyConfig(companyId),
          fetchPlatformBillingContracts(companyId),
        ]);
        if (cancelled) return;
        setConfig(cfgRes?.config ?? null);
        setContractsPayload(contractsRes || null);
      } catch (err) {
        if (cancelled) return;
        setError(
          err?.response?.data?.message ||
            err?.message ||
            'Impossible de charger le détail entreprise.'
        );
        setConfig(null);
        setContractsPayload(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [isOpen, companyId]);

  useEffect(() => {
    if (!isOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (!isOpen) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !drawerRef.current) return undefined;
    const closeBtn = drawerRef.current.querySelector('button[aria-label="Fermer"]');
    closeBtn?.focus();
    return undefined;
  }, [isOpen]);

  if (!isOpen || !user) return null;

  const contracts = contractsPayload?.contracts || [];
  const latest = contracts[0] || null;
  const readiness = contractsPayload?.readiness || null;
  const debtor = contractsPayload?.debtor_address || null;
  const partner = contractsPayload?.partner_identity || null;
  const companyFields = partner?.company_fields || {};
  const effective = latest || config;
  const financeConfigPath = adminId ? adminPaths.financeConfig(adminId) : null;

  const drawer = (
    <div
      ref={overlayRef}
      className={styles.overlay}
      role="presentation"
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <aside
        ref={drawerRef}
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="admin-company-drawer-title"
      >
        <header className={styles.header}>
          <div className={styles.headerText}>
            <p className={styles.eyebrow}>Entreprise</p>
            <h2 id="admin-company-drawer-title" className={styles.title}>
              {user.company_name || debtor?.legal_name || (companyId ? `Entreprise #${companyId}` : user.username || 'Entreprise')}
            </h2>
            <p className={styles.subtitle}>
              {companyId ? `ID ${companyId}` : 'Aucune entreprise liée'}
              {user.username ? ` · Compte ${user.username}` : ''}
            </p>
          </div>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={18} aria-hidden />
          </button>
        </header>

        <div className={styles.body}>
          {!companyId ? (
            <p className={styles.error} role="alert">
              Ce compte a le rôle Entreprise, mais aucune fiche entreprise n&apos;y est liée.
              Impossible de charger abonnements et accès commerciaux.
            </p>
          ) : null}

          {loading ? <p className={styles.loading}>Chargement du détail…</p> : null}
          {error ? (
            <p className={styles.error} role="alert">
              {error}
            </p>
          ) : null}

          <section className={styles.section} aria-labelledby="drawer-sec-account">
            <h3 id="drawer-sec-account" className={styles.sectionTitle}>
              Compte utilisateur
            </h3>
            <dl className={styles.kv}>
              <div>
                <dt>Nom</dt>
                <dd>{user.username || '—'}</dd>
              </div>
              <div>
                <dt>E-mail</dt>
                <dd>{user.email || '—'}</dd>
              </div>
              <div>
                <dt>Inscription</dt>
                <dd>
                  {user.created_at
                    ? new Date(user.created_at).toLocaleString('fr-CH')
                    : '—'}
                </dd>
              </div>
            </dl>
          </section>

          {companyId ? (
            <>
          <section className={styles.section} aria-labelledby="drawer-sec-access">
            <h3 id="drawer-sec-access" className={styles.sectionTitle}>
              Restriction commerciale LIRIE
            </h3>
            <div className={styles.accessRow}>
              <span className={`${styles.badge} ${billingBadgeClass(billingState)}`}>
                {formatBillingAccessLabel(billingState)}
              </span>
              {dunningPaused ? (
                <span className={styles.hint}>
                  Pause recouvrement jusqu&apos;au{' '}
                  {formatDate(user.dunning_paused_until)}
                </span>
              ) : null}
            </div>
            <div className={styles.actions}>
              {billingState !== 'active' ? (
                <button
                  type="button"
                  className={styles.actionBtn}
                  onClick={() => onSetBillingAccess?.(user, 'active')}
                >
                  Lever
                </button>
              ) : null}
              {billingState !== 'partial' ? (
                <button
                  type="button"
                  className={styles.actionBtn}
                  onClick={() => onSetBillingAccess?.(user, 'partial')}
                >
                  Partiel
                </button>
              ) : null}
              {billingState !== 'full' ? (
                <button
                  type="button"
                  className={styles.actionBtnWarn}
                  onClick={() => onSetBillingAccess?.(user, 'full')}
                >
                  Complet
                </button>
              ) : null}
              <button
                type="button"
                className={styles.actionBtn}
                onClick={() => onPauseDunning?.(user)}
              >
                Pause dunning
              </button>
            </div>
          </section>

          <section className={styles.section} aria-labelledby="drawer-sec-identity">
            <h3 id="drawer-sec-identity" className={styles.sectionTitle}>
              Identité &amp; facturation
            </h3>
            <dl className={styles.kv}>
              <div>
                <dt>Raison sociale</dt>
                <dd>{debtor?.legal_name || user.company_name || '—'}</dd>
              </div>
              <div>
                <dt>Adresse</dt>
                <dd>
                  {[
                    [debtor?.street_name, debtor?.building_number].filter(Boolean).join(' '),
                    [debtor?.postal_code, debtor?.city].filter(Boolean).join(' '),
                    debtor?.country_code,
                  ]
                    .filter(Boolean)
                    .join(', ') || '—'}
                </dd>
              </div>
              <div>
                <dt>IDE</dt>
                <dd>{companyFields.uid_ide || '—'}</dd>
              </div>
              <div>
                <dt>Forme juridique</dt>
                <dd>{companyFields.legal_form || '—'}</dd>
              </div>
              <div>
                <dt>Signataire</dt>
                <dd>
                  {[companyFields.signatory_name, companyFields.signatory_title]
                    .filter(Boolean)
                    .join(' — ') || '—'}
                </dd>
              </div>
            </dl>
          </section>

          <section className={styles.section} aria-labelledby="drawer-sec-sub">
            <h3 id="drawer-sec-sub" className={styles.sectionTitle}>
              Abonnement &amp; produits
            </h3>
            {!effective && !loading ? (
              <p className={styles.empty}>Aucune configuration commerciale enregistrée.</p>
            ) : (
              <dl className={styles.kv}>
                <div>
                  <dt>Facturation activée</dt>
                  <dd>{effective?.is_billing_enabled ? 'Oui' : 'Non'}</dd>
                </div>
                <div>
                  <dt>Mode abonnement</dt>
                  <dd>{formatPricingMode(effective?.subscription_pricing_mode)}</dd>
                </div>
                <div>
                  <dt>Montant fixe</dt>
                  <dd>{formatMoney(effective?.custom_subscription_amount)}</dd>
                </div>
                <div>
                  <dt>Commission</dt>
                  <dd>{formatPercent(effective?.commission_rate)}</dd>
                </div>
                <div>
                  <dt>Support (taux horaire)</dt>
                  <dd>{formatMoney(effective?.support_hourly_rate_default)}</dd>
                </div>
                <div>
                  <dt>Délai de paiement</dt>
                  <dd>
                    {effective?.payment_terms_days != null
                      ? `${effective.payment_terms_days} j`
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Dunning auto</dt>
                  <dd>{effective?.automated_dunning_enabled === false ? 'Non' : 'Oui'}</dd>
                </div>
                <div>
                  <dt>Effectif depuis</dt>
                  <dd>{formatDate(effective?.effective_from)}</dd>
                </div>
              </dl>
            )}
            {effective ? (
              <ul className={styles.flagList}>
                <li className={effective.partial_block_marketplace_offers !== false ? styles.flagOn : styles.flagOff}>
                  Blocage offres marketplace (accès partiel)
                </li>
                <li
                  className={
                    effective.partial_block_marketplace_acceptance !== false
                      ? styles.flagOn
                      : styles.flagOff
                  }
                >
                  Blocage acceptation marketplace
                </li>
                <li
                  className={
                    effective.partial_block_billable_support !== false
                      ? styles.flagOn
                      : styles.flagOff
                  }
                >
                  Blocage support facturable
                </li>
                <li
                  className={
                    effective.partial_block_billable_configuration !== false
                      ? styles.flagOn
                      : styles.flagOff
                  }
                >
                  Blocage configuration facturable
                </li>
              </ul>
            ) : null}
          </section>

          <section className={styles.section} aria-labelledby="drawer-sec-contracts">
            <h3 id="drawer-sec-contracts" className={styles.sectionTitle}>
              Versions de contrat
            </h3>
            {contracts.length === 0 && !loading ? (
              <p className={styles.empty}>Aucun contrat commercial.</p>
            ) : (
              <ul className={styles.contractList}>
                {contracts.map((c, idx) => (
                  <li key={c.id || idx} className={styles.contractItem}>
                    <div className={styles.contractHead}>
                      <strong>Version {c.version_number ?? c.id ?? idx + 1}</strong>
                      {idx === 0 ? <span className={styles.latestTag}>Courante</span> : null}
                    </div>
                    <p className={styles.contractMeta}>
                      {formatPricingMode(c.subscription_pricing_mode)}
                      {' · '}
                      Commission {formatPercent(c.commission_rate)}
                      {' · '}
                      Dès {formatDate(c.effective_from)}
                      {c.closed_at ? ` · Clôturé ${formatDate(c.closed_at)}` : ''}
                    </p>
                    {c.active_agreement ? (
                      <p className={styles.contractMeta}>
                        Accord partenaire : {c.active_agreement.status || 'présent'}
                      </p>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </section>

          {readiness ? (
            <section className={styles.section} aria-labelledby="drawer-sec-ready">
              <h3 id="drawer-sec-ready" className={styles.sectionTitle}>
                Préparation facturation
              </h3>
              <ul className={styles.readinessList}>
                <li
                  className={
                    readiness.contract_calculation_ready ? styles.readyOk : styles.readyKo
                  }
                >
                  Calcul du relevé
                </li>
                <li
                  className={
                    readiness.debtor_identity_ready ? styles.readyOk : styles.readyKo
                  }
                >
                  Identité débiteur
                </li>
                <li className={readiness.creditor_qr_ready ? styles.readyOk : styles.readyKo}>
                  Créancier LIRIE (QR)
                </li>
              </ul>
            </section>
          ) : null}
            </>
          ) : null}
        </div>

        <footer className={styles.footer}>
          {financeConfigPath && companyId ? (
            <Link to={financeConfigPath} className={styles.footerLink} onClick={onClose}>
              Ouvrir la config Finance
              <FiExternalLink size={14} aria-hidden />
            </Link>
          ) : (
            <span />
          )}
          <button type="button" className={styles.footerClose} onClick={onClose}>
            Fermer
          </button>
        </footer>
      </aside>
    </div>
  );

  return createPortal(drawer, document.body);
};

export default AdminCompanyDetailDrawer;
