import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useSearchParams, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiLock, FiRefreshCw } from 'react-icons/fi';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import apiClient from '../../../utils/apiClient';
import { startSaferpayHostedCheckout } from '../../../services/clientSaferpayPaymentService';
import { toastSaferpayCheckoutError, openSupportContact } from '../../../utils/saferpayPaymentUi';
import {
  clearSaferpayPayResume,
  writeSaferpayPayResume,
} from '../../../utils/clientSaferpayPayResume';
import { getActivePublicId } from '../../../utils/webAuthSession';
import { toast } from 'sonner';
import styles from './ClientSaferpayPaymentReturn.module.css';

function formatChf(amount) {
  const n = Number(amount);
  if (!Number.isFinite(n)) return null;
  try {
    return new Intl.NumberFormat('fr-CH', { style: 'currency', currency: 'CHF' }).format(n);
  } catch {
    return `${n.toFixed(2)} CHF`;
  }
}

/**
 * Page intermédiaire : initialise la Payment Page Saferpay (redirection navigateur).
 */
const ClientSaferpayCheckoutStart = () => {
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const bookingIdRaw = searchParams.get('bookingId');
  const bookingId = bookingIdRaw ? parseInt(bookingIdRaw, 10) : NaN;

  const state = location.state && typeof location.state === 'object' ? location.state : {};
  const amountChf = state.amountChf;
  const payerLabel = state.payerLabel;
  const lifecycleLabel = state.lifecycleLabel;
  const fromDashboardPath =
    typeof state.fromDashboardPath === 'string' && state.fromDashboardPath.startsWith('/')
      ? state.fromDashboardPath
      : `/dashboard/client/${encodeURIComponent(getActivePublicId() || '')}`;

  const [phase, setPhase] = useState('idle');
  const [errorHint, setErrorHint] = useState('');
  const [fallbackAmountChf, setFallbackAmountChf] = useState(null);
  const [fallbackPayerLabel, setFallbackPayerLabel] = useState('');
  const [fallbackLifecycleLabel, setFallbackLifecycleLabel] = useState('');
  const inFlightRef = useRef(false);
  const amountFromState = Number.isFinite(Number(amountChf)) ? Number(amountChf) : null;
  const resolvedAmountChf = amountFromState ?? fallbackAmountChf;
  const resolvedPayerLabel = payerLabel || fallbackPayerLabel;
  const resolvedLifecycleLabel = lifecycleLabel || fallbackLifecycleLabel;

  useEffect(() => {
    let mounted = true;
    const hydrateFallbackAmount = async () => {
      if (!Number.isFinite(bookingId) || bookingId <= 0) return;
      if (amountFromState != null) return;
      try {
        const { data } = await apiClient.get(`/bookings/${bookingId}`, { timeout: 60000 });
        const booking = data?.data ?? data ?? {};
        const backendAmount =
          Number.isFinite(Number(booking?.payment_amount))
            ? Number(booking.payment_amount)
            : Number.isFinite(Number(booking?.amount))
              ? Number(booking.amount)
              : null;
        if (!mounted || backendAmount == null) return;
        setFallbackAmountChf(backendAmount);
        if (typeof booking?.payer_label === 'string') {
          setFallbackPayerLabel(booking.payer_label);
        } else if (typeof booking?.coverage_label === 'string') {
          setFallbackPayerLabel(booking.coverage_label);
        }
        if (typeof booking?.status === 'string' && booking.status.trim()) {
          setFallbackLifecycleLabel(booking.status);
        }
      } catch {
        // Pas bloquant : la redirection paiement peut continuer sans affichage montant.
      }
    };
    void hydrateFallbackAmount();
    return () => {
      mounted = false;
    };
  }, [amountFromState, bookingId]);

  const runCheckout = useCallback(async () => {
    if (!Number.isFinite(bookingId) || bookingId <= 0) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    clearSaferpayPayResume();
    setPhase('redirecting');
    setErrorHint('');
    try {
      await startSaferpayHostedCheckout(bookingId);
    } catch (e) {
      setPhase('error');
      const msg = e?.message || 'Le paiement en ligne n’est pas disponible pour le moment.';
      writeSaferpayPayResume({
        bookingId,
        finalAmount: resolvedAmountChf != null ? Number(resolvedAmountChf) : undefined,
        payerLabel: resolvedPayerLabel || 'Client',
        lifecycleLabel: resolvedLifecycleLabel || 'Paiement requis',
      });
      setErrorHint(msg);
      toastSaferpayCheckoutError(toast, e);
    } finally {
      inFlightRef.current = false;
    }
  }, [bookingId, resolvedAmountChf, resolvedLifecycleLabel, resolvedPayerLabel]);

  useEffect(() => {
    if (!bookingIdRaw || Number.isNaN(bookingId)) {
      setPhase('error');
      setErrorHint('Lien de paiement incomplet (réservation manquante).');
      return;
    }
    void runCheckout();
  }, [bookingId, bookingIdRaw, runCheckout]);

  const dashboardLink = fromDashboardPath || '/dashboard/client';

  const amountLabel = formatChf(resolvedAmountChf);

  return (
    <div className={styles.page}>
      <HeaderDashboard userName="Client" />
      <main className={styles.main}>
        <motion.article
          className={styles.checkoutArticle}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.36, ease: [0.25, 0.1, 0.25, 1] }}
          aria-label="Paiement sécurisé Saferpay"
        >
          <header className={styles.checkoutHero}>
            <div className={`${styles.heroIconWrap} ${styles.heroIconWrapNeutral}`} aria-hidden="true">
              <FiLock className={styles.heroIconSvg} />
            </div>
            <p className={styles.heroKicker}>Saferpay</p>
            <h1 className={styles.checkoutTitle}>Paiement sécurisé</h1>
            {!Number.isNaN(bookingId) ? (
              <span
                className={styles.heroBookingPill}
                title={`Réservation ${bookingId}`}
                aria-label={`Réservation numéro ${bookingId}`}
              >
                Course n° {bookingId}
              </span>
            ) : null}
          </header>

          <div className={styles.checkoutBody}>
            {resolvedLifecycleLabel ? (
              <p className={styles.checkoutMeta}>
                <strong>{resolvedLifecycleLabel}</strong>
              </p>
            ) : null}
            {resolvedPayerLabel ? (
              <p className={styles.checkoutMeta}>
                Payeur : <strong>{resolvedPayerLabel}</strong>
              </p>
            ) : null}
            {amountLabel ? (
              <p className={styles.checkoutMeta}>
                Montant à régler : <strong>{amountLabel}</strong>
              </p>
            ) : null}

            {phase === 'redirecting' ? (
              <p className={styles.checkoutPolling} role="status" aria-live="polite">
                <FiRefreshCw className={styles.spinnerIcon} aria-hidden />
                <span>
                  Connexion au prestataire de paiement… Vous allez être redirigé vers la page
                  sécurisée Saferpay.
                </span>
              </p>
            ) : null}

            {phase === 'error' ? (
              <div className={styles.checkoutErrorBox}>
                <p className={styles.checkoutErrorTitle}>Impossible de lancer le paiement</p>
                <p className={styles.checkoutErrorText}>{errorHint}</p>
                <p className={styles.checkoutErrorHint}>
                  Vous pouvez réessayer ci-dessous ou retourner au tableau de bord pour finaliser plus
                  tard.
                </p>
                <div className={styles.checkoutActions}>
                  <button type="button" className={styles.checkoutOutlineBtn} onClick={() => void runCheckout()}>
                    Réessayer le paiement
                  </button>
                  <button type="button" className={styles.checkoutOutlineBtn} onClick={() => openSupportContact()}>
                    Contacter le support
                  </button>
                </div>
              </div>
            ) : null}
          </div>

          <footer className={styles.checkoutFooter}>
            <Link to={dashboardLink} className={styles.checkoutBackLink}>
              ← Retour au tableau de bord
            </Link>
          </footer>
        </motion.article>
      </main>
      <Footer />
    </div>
  );
};

export default ClientSaferpayCheckoutStart;
