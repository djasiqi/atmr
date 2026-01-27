import React, { useEffect } from 'react';
import styles from './BillingReviewDrawer.module.css';
import {
  getRecipientLabel,
  getRecipientSourceLabel,
  getRecipientWarningText,
} from '../../../../utils/billingRecipient';

const BillingReviewDrawer = ({
  booking,
  isOpen,
  onClose,
  onOpenSetPayer,
  companyPublicId,
}) => {
  useEffect(() => {
    if (!isOpen) return;
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onClose?.();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen || !booking) return null;

  const bookingId = booking?.booking_id;
  const clientId = booking?.client_id || booking?.client?.id;
  const patientName = String(booking?.patient_name || '').trim();
  const canNavigate = Boolean(companyPublicId);
  const canOpenClient = Boolean(clientId || patientName);
  const canOpenBooking = Boolean(bookingId || patientName);
  const rawReturnTo = `${window.location.pathname}${window.location.search}`;
  const isAllowedReturnTo = (value) =>
    value.startsWith('/dashboard/company/') || value.startsWith('/company/');
  const returnTo = isAllowedReturnTo(rawReturnTo)
    ? encodeURIComponent(rawReturnTo)
    : null;

  const getBasePaths = () => {
    const baseDashboard = companyPublicId ? `/dashboard/company/${companyPublicId}` : null;
    const baseCompany = '/company';
    const preferDashboard = Boolean(baseDashboard)
      && window.location.pathname.includes('/dashboard/company/');
    const bases = preferDashboard
      ? [baseDashboard, baseCompany]
      : [baseCompany, baseDashboard];
    return bases.filter(Boolean);
  };

  const buildSearchQuery = () => {
    if (!patientName) return null;
    const encoded = encodeURIComponent(patientName);
    return `search=${encoded}&focusSearch=1`;
  };

  const buildSearchFallbacks = (basePath, resource) => {
    const searchQuery = buildSearchQuery();
    if (!searchQuery) return [];
    return [`${basePath}/${resource}?${searchQuery}`];
  };

  const navigateWithFallbacks = (paths, { includeReturnTo = false } = {}) => {
    const target = paths.find(Boolean);
    if (target) {
      if (includeReturnTo && returnTo) {
        const joiner = target.includes('?') ? '&' : '?';
        window.location.assign(`${target}${joiner}returnTo=${returnTo}`);
      } else {
        window.location.assign(target);
      }
    }
  };

  const openClient = () => {
    const bases = getBasePaths();
    const candidates = [];
    bases.forEach((base) => {
      if (clientId) {
        candidates.push(`${base}/clients?openClientId=${clientId}`);
        candidates.push(`${base}/clients?clientId=${clientId}`);
        candidates.push(`${base}/clients/${clientId}`);
      }
      candidates.push(`${base}/clients`);
      candidates.push(...buildSearchFallbacks(base, 'clients'));
    });
    navigateWithFallbacks(candidates, { includeReturnTo: true });
  };

  const openBooking = () => {
    const bases = getBasePaths();
    const candidates = [];
    bases.forEach((base) => {
      if (bookingId) {
        candidates.push(`${base}/bookings/${bookingId}`);
        candidates.push(`${base}/reservations/${bookingId}`);
      }
      candidates.push(...buildSearchFallbacks(base, 'bookings'));
    });
    navigateWithFallbacks(candidates, { includeReturnTo: true });
  };

  const openBillingSettings = () => {
    const bases = getBasePaths();
    const candidates = [];
    bases.forEach((base) => {
      candidates.push(`${base}/settings/billing`);
      candidates.push(`${base}/settings`);
    });
    navigateWithFallbacks(candidates);
  };

  return (
    <div className={styles.overlay} onClick={onClose} aria-hidden="true">
      <aside
        className={styles.drawer}
        onClick={(e) => e.stopPropagation()}
        aria-label="Détails facturation"
      >
        <header className={styles.header}>
          <div>
            <h2 className={styles.title}>Détails facturation</h2>
            <p className={styles.subtitle}>
              Booking #{booking.booking_id} • {booking.date}
            </p>
          </div>
          <button className={styles.closeButton} onClick={onClose} aria-label="Fermer">
            ✕
          </button>
        </header>

        <section className={styles.section}>
          <div className={styles.label}>Patient</div>
          <div className={styles.value}>{booking.patient_name || '—'}</div>
        </section>

        <section className={styles.section}>
          <div className={styles.label}>Payeur</div>
          <div className={styles.value}>{getRecipientLabel(booking)}</div>
          <div className={styles.metaText}>
            Type: {booking.payer_type || '—'}
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.label}>Source</div>
          <div className={styles.value}>{getRecipientSourceLabel(booking)}</div>
          {booking.billing_source_ref && (
            <div className={styles.metaText}>{booking.billing_source_ref}</div>
          )}
        </section>

        <section className={styles.section}>
          <div className={styles.label}>Statut</div>
          <div className={styles.value}>{booking.status || '—'}</div>
        </section>

        <section className={styles.section}>
          <div className={styles.label}>Alertes</div>
          <div className={styles.value}>
            {getRecipientWarningText(booking) || 'Aucune alerte'}
          </div>
          {booking.billing_override_reason && (
            <div className={styles.metaText}>
              Motif override: {booking.billing_override_reason}
            </div>
          )}
        </section>

        <section className={styles.section}>
          <div className={styles.label}>Actions</div>
          <div className={styles.actionList}>
            <button
              type="button"
              className={styles.actionButton}
              disabled={!canOpenBooking}
              title={
                canOpenBooking
                  ? 'Ouvrir la course (fallbacks automatiques)'
                  : 'Aucun identifiant ni nom patient disponible'
              }
              onClick={openBooking}
            >
              Ouvrir la course
            </button>
            <button
              type="button"
              className={styles.actionButton}
              disabled={!canOpenClient}
              title={
                canOpenClient
                  ? 'Ouvrir le client (fallbacks automatiques)'
                  : 'Aucun identifiant ni nom patient disponible'
              }
              onClick={openClient}
            >
              Ouvrir le client
            </button>
            <button
              type="button"
              className={styles.actionButton}
              disabled={!canNavigate}
              title={
                canNavigate
                  ? 'Ouvrir Paramètres > Facturation'
                  : 'Impossible de déterminer la route'
              }
              onClick={openBillingSettings}
            >
              Ouvrir Paramètres &gt; Facturation
            </button>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.label}>Modifier le payeur</div>
          <div className={styles.valueMuted}>
            Cette section sera activée dès que l’UI d’override est prête.
          </div>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={() => onOpenSetPayer?.(booking)}
          >
            Ouvrir le formulaire
          </button>
        </section>
      </aside>
    </div>
  );
};

export default BillingReviewDrawer;
