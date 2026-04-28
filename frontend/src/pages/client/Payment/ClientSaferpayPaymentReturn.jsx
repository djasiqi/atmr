import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import {
  FiAlertTriangle,
  FiArrowRight,
  FiCheckCircle,
  FiClock,
  FiCreditCard,
  FiRefreshCw,
  FiShield,
  FiXCircle,
} from 'react-icons/fi';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import apiClient from '../../../utils/apiClient';
import {
  SAFERPAY_FINALIZE_ASSERT_FAILED,
  SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
  SAFERPAY_FINALIZE_CAPTURE_FAILED,
  SAFERPAY_FINALIZE_PAYMENT_FAILED,
  SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
  userMessageForSaferpayFinalizeStatus,
} from '../../../constants/saferpayFinalizeStatus';
import { assertSaferpayCheckout } from '../../../services/clientSaferpayPaymentService';
import { getApiErrorMessage } from '../../../utils/apiErrorMessage';
import { getActivePublicId } from '../../../utils/webAuthSession';
import styles from './ClientSaferpayPaymentReturn.module.css';

/**
 * Retour navigateur après Payment Page Saferpay : Assert + Capture côté API, puis affichage du statut.
 */
/** Court affichage du message « Payé » avant envoi vers le tableau de bord client. */
const REDIRECT_AFTER_PAID_MS = 1200;

/**
 * Une seule explication affichée : dérivée du statut assert quand il est connu,
 * sinon formulation honnête (la banque ne transmet pas le libellé exact ici).
 */
function getSingleFailureCause(finalizeStatus) {
  const st = finalizeStatus ? String(finalizeStatus) : '';
  if (st === SAFERPAY_FINALIZE_CAPTURE_FAILED) {
    return {
      Icon: FiAlertTriangle,
      text:
        'La confirmation finale du débit a échoué côté serveur après autorisation bancaire. Aucun prélèvement définitif n’a été enregistré pour cette tentative.',
    };
  }
  if (st === SAFERPAY_FINALIZE_ASSERT_TRANSIENT) {
    return {
      Icon: FiRefreshCw,
      text:
        'Échec temporaire lors de l’échange avec le prestataire de paiement. Réessayez dans quelques instants ou actualisez le statut.',
    };
  }
  if (st === SAFERPAY_FINALIZE_PAYMENT_FAILED) {
    return {
      Icon: FiCreditCard,
      text:
        'Le prestataire indique un paiement refusé ou annulé (banque, carte ou étape de sécurité). Le détail exact n’est pas transmis sur cette page.',
    };
  }
  if (st === SAFERPAY_FINALIZE_ASSERT_FAILED) {
    return {
      Icon: FiShield,
      text:
        'La finalisation technique auprès de Saferpay n’a pas pu être terminée. Le statut affiché correspond à la dernière lecture côté serveur.',
    };
  }
  if (st === SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS) {
    return {
      Icon: FiAlertTriangle,
      text:
        'Statut de transaction inattendu renvoyé par le prestataire. Le paiement est considéré comme non abouti côté application.',
    };
  }
  return {
    Icon: FiCreditCard,
    text:
      'Le refus ou l’annulation provient de la chaîne bancaire (carte, plafond, 3-D Secure ou autre). La banque ne communique pas ici le libellé exact du refus.',
  };
}

/** Affichage type démo : « CHF 45.00 » (espace insécable). */
function formatChf(amount) {
  const n = Number(amount);
  if (!Number.isFinite(n)) return null;
  return `CHF\u00a0${n.toFixed(2)}`;
}

const bodyStaggerContainer = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.065, delayChildren: 0.05 },
  },
};

const bodyStaggerItem = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.38, ease: [0.25, 0.1, 0.25, 1] },
  },
};

const ClientSaferpayPaymentReturn = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const bookingIdRaw = searchParams.get('bookingId');
  const paymentIdRaw = searchParams.get('paymentId');
  const bookingId = bookingIdRaw ? parseInt(bookingIdRaw, 10) : NaN;
  const paymentIdFromQuery = paymentIdRaw ? parseInt(paymentIdRaw, 10) : NaN;

  /** Chargement initial (réservation + identifiant de paiement) — court. */
  const [loading, setLoading] = useState(true);
  /** Assert / Capture Saferpay encore en cours (affiché seulement si le paiement est encore « pending »). */
  const [saferpaySyncPending, setSaferpaySyncPending] = useState(false);
  const [error, setError] = useState(null);
  const [paymentBrief, setPaymentBrief] = useState(null);
  const [authRequired, setAuthRequired] = useState(false);
  /** Statut brut renvoyé par POST …/saferpay/assert (complète le statut relu sur la réservation). */
  const [finalizeStatus, setFinalizeStatus] = useState(null);
  /** Montant course (réservation), affiché à titre informatif — distinct du paiement en ligne. */
  const [bookingAmountChf, setBookingAmountChf] = useState(null);
  const redirectAfterPaidTimerRef = useRef(null);

  const loginResumeHref = useMemo(() => {
    const path = `${location.pathname}${location.search || ''}`;
    return `/login?next=${encodeURIComponent(path)}`;
  }, [location.pathname, location.search]);

  const outcomeForApp = useMemo(() => {
    const oc = (searchParams.get('outcome') || 'success').toLowerCase();
    if (oc === 'success' || oc === 'fail' || oc === 'abort') return oc;
    return 'success';
  }, [searchParams]);

  const fetchBooking = useCallback(async () => {
    const { data } = await apiClient.get(`/bookings/${bookingId}`, { timeout: 60000 });
    const booking = data?.data ?? data;
    const rawAmt = booking?.amount;
    const amtNum = Number(rawAmt);
    setBookingAmountChf(Number.isFinite(amtNum) ? amtNum : null);
    const op = booking?.online_payment ?? null;
    setPaymentBrief(op);
    const st = (op?.status || '').toLowerCase();
    const terminal = st === 'completed' || st === 'failed';
    return { terminal, onlinePayment: op };
  }, [bookingId]);

  const effectivePaymentIdForApp = useMemo(() => {
    if (Number.isFinite(paymentIdFromQuery) && paymentIdFromQuery > 0) {
      return paymentIdFromQuery;
    }
    const op = paymentBrief?.payment_id;
    if (typeof op === 'number' && op > 0) return op;
    const parsed = parseInt(String(op || ''), 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  }, [paymentIdFromQuery, paymentBrief]);

  const lirieAppReturnUrl = useMemo(() => {
    if (!Number.isFinite(bookingId) || bookingId <= 0) return null;
    if (effectivePaymentIdForApp == null) return null;
    return `lirie://payment-return?bookingId=${encodeURIComponent(String(bookingId))}&paymentId=${encodeURIComponent(String(effectivePaymentIdForApp))}&outcome=${encodeURIComponent(outcomeForApp)}`;
  }, [bookingId, effectivePaymentIdForApp, outcomeForApp]);

  useEffect(() => {
    if (!bookingIdRaw || Number.isNaN(bookingId)) {
      setLoading(false);
      setError('Paramètre bookingId manquant ou invalide.');
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const isLikelyMobile = /Android|iPhone|iPad|iPod/i.test(
          typeof navigator !== 'undefined' ? navigator.userAgent : ''
        );
        if (
          isLikelyMobile &&
          Number.isFinite(paymentIdFromQuery) &&
          paymentIdFromQuery > 0 &&
          Number.isFinite(bookingId) &&
          bookingId > 0
        ) {
          const mobileDeepLink = `lirie://payment-return?bookingId=${encodeURIComponent(
            String(bookingId)
          )}&paymentId=${encodeURIComponent(String(paymentIdFromQuery))}&outcome=${encodeURIComponent(
            outcomeForApp
          )}`;
          window.location.href = mobileDeepLink;
          await new Promise((r) => setTimeout(r, 1500));
        }
        if (cancelled) return;
        const { terminal, onlinePayment } = await fetchBooking();
        if (cancelled) return;

        let paymentId = paymentIdFromQuery;
        if (!Number.isFinite(paymentId) || paymentId <= 0) {
          const pid = onlinePayment?.payment_id;
          paymentId = typeof pid === 'number' ? pid : parseInt(String(pid || ''), 10);
        }
        if (!Number.isFinite(paymentId) || paymentId <= 0) {
          setError('Identifiant de paiement introuvable. Revenez depuis « Payer » ou le tableau de bord.');
          return;
        }

        /* Déjà enregistré (souvent après l’URL de notification Saferpay) : pas d’appel Assert/Capture. */
        if (terminal) {
          return;
        }

        setLoading(false);
        setSaferpaySyncPending(true);
        try {
          try {
            const assertPayload = await assertSaferpayCheckout(bookingId, paymentId);
            if (!cancelled && assertPayload?.status) {
              setFinalizeStatus(String(assertPayload.status));
            }
          } catch (assertErr) {
            if (!cancelled) {
              const st = assertErr?.response?.status;
              if (st === 401) {
                setAuthRequired(true);
                setError(
                  'Session expirée. Reconnectez-vous pour finaliser la lecture du paiement.'
                );
                return;
              }
              const msg = getApiErrorMessage(
                assertErr,
                'La finalisation du paiement a échoué côté serveur. Le statut ci-dessous est relu depuis la réservation.'
              );
              setError(msg);
            }
          }
          if (!cancelled) {
            await fetchBooking();
          }
        } finally {
          if (!cancelled) {
            setSaferpaySyncPending(false);
          }
        }
      } catch (e) {
        if (!cancelled) {
          const st = e?.response?.status;
          if (st === 401) {
            setAuthRequired(true);
            setError(
              'Votre session a expiré pendant le paiement. Reconnectez-vous pour afficher le statut.'
            );
          } else {
            setError(
              e?.response?.data?.message ||
                'Impossible de charger le statut de la réservation.'
            );
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [bookingId, bookingIdRaw, fetchBooking, paymentIdFromQuery, outcomeForApp]);

  const publicId = getActivePublicId();
  const dashboardLink = publicId ? `/dashboard/client/${publicId}` : loginResumeHref;
  const reservationsHref = publicId ? `/reservations/${publicId}` : loginResumeHref;

  const payStatus = paymentBrief ? (paymentBrief.status || '').toLowerCase() : null;
  const finalizeHint =
    finalizeStatus && userMessageForSaferpayFinalizeStatus(finalizeStatus);
  const showFinalizeHint = Boolean(finalizeHint);
  const showMainCard = !loading && !error;

  const singleFailureCause = useMemo(
    () => (payStatus === 'failed' ? getSingleFailureCause(finalizeStatus) : null),
    [payStatus, finalizeStatus]
  );

  const statusPresentation = useMemo(() => {
    if (!paymentBrief) return { tone: 'neutral' };
    if (payStatus === 'completed') return { tone: 'success' };
    if (payStatus === 'failed') return { tone: 'danger' };
    if (payStatus === 'pending') return { tone: 'pending' };
    return { tone: 'neutral' };
  }, [paymentBrief, payStatus]);

  const panelAccentClass =
    statusPresentation.tone === 'danger'
      ? styles.panelAccentDanger
      : statusPresentation.tone === 'success'
        ? styles.panelAccentSuccess
        : statusPresentation.tone === 'pending'
          ? styles.panelAccentPending
          : '';

  const panelToneBorderClass =
    statusPresentation.tone === 'danger'
      ? styles.panelToneDanger
      : statusPresentation.tone === 'success'
        ? styles.panelToneSuccess
        : statusPresentation.tone === 'pending'
          ? styles.panelTonePending
          : '';

  const heroBookingPillClass = useMemo(() => {
    const tone = statusPresentation.tone;
    if (tone === 'success') return `${styles.heroBookingPill} ${styles.heroBookingPillSuccess}`;
    if (tone === 'danger') return `${styles.heroBookingPill} ${styles.heroBookingPillDanger}`;
    if (tone === 'pending') return `${styles.heroBookingPill} ${styles.heroBookingPillPending}`;
    return styles.heroBookingPill;
  }, [statusPresentation.tone]);

  const heroUi = useMemo(() => {
    const tone = statusPresentation.tone;
    if (tone === 'success') {
      return {
        title: 'Paiement confirmé',
        lead:
          'Votre paiement en ligne est enregistré. Vous pouvez suivre la course depuis vos réservations.',
        Icon: FiCheckCircle,
      };
    }
    if (tone === 'danger') {
      return {
        title: 'Paiement refusé ou annulé',
        lead:
          "Le paiement n'a pas pu aboutir. Votre course peut rester visible dans vos réservations — aucun montant n'a été débité pour cette tentative.",
        Icon: FiXCircle,
      };
    }
    if (tone === 'pending') {
      return {
        title: 'En cours de traitement',
        lead:
          'La confirmation auprès du prestataire peut prendre quelques instants. Vous pouvez actualiser ou consulter vos courses.',
        Icon: FiClock,
      };
    }
    return {
      title: 'Résultat du paiement',
      lead:
        'Synthèse renvoyée par le serveur après passage sur la page de paiement sécurisée (assert / capture).',
      Icon: FiShield,
    };
  }, [statusPresentation.tone]);

  const heroBlockClass = useMemo(() => {
    const tone = statusPresentation.tone;
    if (tone === 'success') return `${styles.heroBlock} ${styles.heroBlockSuccess}`;
    if (tone === 'danger') return `${styles.heroBlock} ${styles.heroBlockDanger}`;
    if (tone === 'pending') return `${styles.heroBlock} ${styles.heroBlockPending}`;
    return `${styles.heroBlock} ${styles.heroBlockNeutral}`;
  }, [statusPresentation.tone]);

  const heroIconWrapClass = useMemo(() => {
    const tone = statusPresentation.tone;
    if (tone === 'success') return `${styles.heroIconWrap} ${styles.heroIconWrapSuccess}`;
    if (tone === 'danger') return `${styles.heroIconWrap} ${styles.heroIconWrapDanger}`;
    if (tone === 'pending') return `${styles.heroIconWrap} ${styles.heroIconWrapPending}`;
    return `${styles.heroIconWrap} ${styles.heroIconWrapNeutral}`;
  }, [statusPresentation.tone]);

  const retryPayHref =
    !Number.isNaN(bookingId) && bookingId > 0
      ? `/client/payment/saferpay/start?bookingId=${encodeURIComponent(String(bookingId))}`
      : null;

  const footerHelpLines = useMemo(() => {
    if (payStatus === 'failed') {
      return {
        before: null,
        linkLabel: 'Aide & support',
        after: ' — questions fréquentes sur le paiement.',
      };
    }
    if (payStatus === 'pending') {
      return {
        before: 'Le délai de confirmation varie selon les banques. ',
        linkLabel: 'Aide & support',
        after: ' pour le paiement et le suivi des courses.',
      };
    }
    return {
      before: null,
      linkLabel: 'Aide & support',
      after: ' — questions fréquentes sur le paiement.',
    };
  }, [payStatus]);

  useEffect(() => {
    if (loading || error || saferpaySyncPending) return;
    if (payStatus !== 'completed' || !publicId) return;

    redirectAfterPaidTimerRef.current = window.setTimeout(() => {
      redirectAfterPaidTimerRef.current = null;
      navigate(`/dashboard/client/${publicId}`, { replace: true });
    }, REDIRECT_AFTER_PAID_MS);

    return () => {
      if (redirectAfterPaidTimerRef.current != null) {
        window.clearTimeout(redirectAfterPaidTimerRef.current);
        redirectAfterPaidTimerRef.current = null;
      }
    };
  }, [loading, error, saferpaySyncPending, payStatus, publicId, navigate]);

  const HeroIcon = heroUi.Icon;

  return (
    <div className={styles.page}>
      <HeaderDashboard userName="Client" />
      <main className={styles.main}>
        {lirieAppReturnUrl ? (
          <div className={styles.appReopenBanner} role="region" aria-label="Application mobile LIRIE">
            <a href={lirieAppReturnUrl} className={styles.appReopenLink}>
              Rouvrir l&apos;application LIRIE
            </a>
          </div>
        ) : null}
        {loading ? (
          <div className={styles.stateCard} role="status" aria-live="polite">
            <p className={styles.stateText}>Chargement du statut…</p>
          </div>
        ) : null}

        {error ? (
          <div className={styles.errorCard} role="alert">
            <p className={styles.errorTitle}>Impossible d’afficher le statut</p>
            <p className={styles.error}>{error}</p>
            {authRequired ? (
              <Link to={loginResumeHref} className={styles.link}>
                Se reconnecter et revenir sur cette page
              </Link>
            ) : null}
          </div>
        ) : null}

        <AnimatePresence mode="wait">
          {showMainCard ? (
            <motion.div
              key={`${payStatus || 'none'}-${paymentBrief ? 'op' : 'noop'}`}
              className={styles.panelMotion}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.36, ease: [0.25, 0.1, 0.25, 1] }}
            >
              <article
                className={[styles.panel, panelAccentClass, panelToneBorderClass].filter(Boolean).join(' ')}
                aria-label={`Résultat du paiement — ${heroUi.title}`}
              >
                <header className={heroBlockClass}>
                  <motion.div
                    className={heroIconWrapClass}
                    aria-hidden="true"
                    initial={{ opacity: 0, scale: 0.82 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 20, delay: 0.05 }}
                  >
                    <HeroIcon
                      className={styles.heroIconSvg}
                      strokeWidth={statusPresentation.tone === 'pending' ? 1.5 : 2}
                    />
                  </motion.div>
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1, duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
                  >
                    <p className={styles.heroKicker}>Saferpay</p>
                    <h1 className={styles.heroTitle}>{heroUi.title}</h1>
                  </motion.div>
                  {!Number.isNaN(bookingId) ? (
                    <motion.span
                      className={heroBookingPillClass}
                      title={`Réservation ${bookingId}`}
                      aria-label={`Réservation numéro ${bookingId}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.18, duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
                    >
                      Course n° {bookingId}
                    </motion.span>
                  ) : null}
                </header>

                <motion.section
                  className={styles.panelBody}
                  aria-labelledby="saferpay-return-summary"
                  variants={bodyStaggerContainer}
                  initial="hidden"
                  animate="visible"
                >
                  <motion.div variants={bodyStaggerItem}>
                    <h2 id="saferpay-return-summary" className={styles.sectionEyebrow}>
                      Confirmation serveur
                    </h2>
                    <p className={styles.cardLead}>{heroUi.lead}</p>
                  </motion.div>

                  {payStatus === 'completed' && publicId ? (
                    <motion.div variants={bodyStaggerItem}>
                      <div className={`${styles.notice} ${styles.noticeSuccess}`} role="status">
                        Redirection vers votre tableau de bord…
                      </div>
                    </motion.div>
                  ) : null}
                  {saferpaySyncPending && payStatus === 'pending' ? (
                    <motion.div variants={bodyStaggerItem}>
                      <div className={`${styles.notice} ${styles.noticeInfo}`} role="status">
                        Confirmation auprès de Saferpay en cours. Vous pouvez quitter cette page : le
                        statut sera mis à jour sur le tableau de bord une fois terminé.
                      </div>
                    </motion.div>
                  ) : null}
                  {showFinalizeHint ? (
                    <motion.div variants={bodyStaggerItem}>
                      <div className={`${styles.notice} ${styles.noticeWarning}`} role="status">
                        {finalizeHint}
                      </div>
                    </motion.div>
                  ) : null}

                  {formatChf(bookingAmountChf) ? (
                    <motion.div variants={bodyStaggerItem}>
                      <div className={styles.amountRow}>
                        <span
                          className={styles.amountRowLabel}
                          title="Montant indicatif lié à la réservation (course)"
                        >
                          Montant
                        </span>
                        <span className={styles.amountRowValue}>{formatChf(bookingAmountChf)}</span>
                      </div>
                    </motion.div>
                  ) : null}

                  {!paymentBrief ? (
                    <motion.p className={styles.emptyPayment} variants={bodyStaggerItem}>
                      Aucun paiement en ligne associé à cette réservation.
                    </motion.p>
                  ) : null}

                  {payStatus === 'failed' && singleFailureCause ? (
                    <motion.div
                      className={styles.failureHints}
                      role="status"
                      aria-label="Information sur le refus du paiement"
                      variants={bodyStaggerItem}
                    >
                      <p className={styles.failureHintsTitle}>Cause indiquée</p>
                      <div className={styles.failureHintSingle}>
                        {React.createElement(singleFailureCause.Icon, {
                          className: styles.failureHintIcon,
                          'aria-hidden': true,
                        })}
                        <span>{singleFailureCause.text}</span>
                      </div>
                    </motion.div>
                  ) : null}

                  {paymentBrief && payStatus === 'completed' ? (
                    <motion.div className={styles.ctaStack} variants={bodyStaggerItem}>
                      <Link to={reservationsHref} className={`${styles.link} ${styles.ctaMotion}`}>
                        Voir mes courses
                        <FiArrowRight className={styles.ctaIcon} aria-hidden />
                      </Link>
                    </motion.div>
                  ) : null}
                  {paymentBrief && payStatus === 'failed' && retryPayHref ? (
                    <motion.div className={styles.ctaStack} variants={bodyStaggerItem}>
                      <Link to={retryPayHref} className={`${styles.ctaDanger} ${styles.ctaMotion}`}>
                        <FiRefreshCw className={styles.ctaIcon} aria-hidden />
                        Réessayer le paiement
                      </Link>
                      <Link to={reservationsHref} className={`${styles.linkSecondary} ${styles.ctaMotion}`}>
                        Voir mes courses
                        <FiArrowRight className={styles.ctaIcon} aria-hidden />
                      </Link>
                    </motion.div>
                  ) : null}
                  {paymentBrief && payStatus === 'pending' ? (
                    <motion.p className={styles.pendingCtaHint} variants={bodyStaggerItem}>
                      Le statut peut encore évoluer. Consultez « Mes courses » ou le tableau de bord dans quelques
                      instants.
                    </motion.p>
                  ) : null}
                </motion.section>

                <motion.div
                  className={styles.panelFooterMeta}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.32, duration: 0.35 }}
                >
                  <p className={styles.panelFooterHelp}>
                    {footerHelpLines.before}
                    <Link to="/aide" className={`${styles.navHelpLink} ${styles.footerHelpAccent}`}>
                      {footerHelpLines.linkLabel}
                    </Link>
                    {footerHelpLines.after}
                  </p>
                  <Link
                    to={dashboardLink}
                    className={`${styles.footerDashboardLink} ${styles.ctaMotion}`}
                  >
                    Tableau de bord
                    <FiArrowRight className={styles.footerDashboardIcon} aria-hidden />
                  </Link>
                </motion.div>
              </article>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {!showMainCard && !loading ? (
          <div className={styles.actionsBlock}>
            <nav className={styles.actions} aria-label="Navigation">
              <Link to={dashboardLink} className={styles.link}>
                Tableau de bord
              </Link>
              <Link to={reservationsHref} className={styles.linkSecondary}>
                Mes courses
              </Link>
            </nav>
          </div>
        ) : null}
      </main>
      <Footer />
    </div>
  );
};

export default ClientSaferpayPaymentReturn;
