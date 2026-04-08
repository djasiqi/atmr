import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import apiClient from '../../../utils/apiClient';
import styles from './ClientWorldlinePaymentReturn.module.css';

const POLL_INTERVAL_MS = 2000;
const MAX_POLLS = 30;

/**
 * Page de retour après MyCheckout (Worldline).
 * Polling court : les webhooks peuvent arriver quelques secondes après la redirection.
 */
const ClientWorldlinePaymentReturn = () => {
  const [searchParams] = useSearchParams();
  const bookingIdRaw = searchParams.get('bookingId');
  const bookingId = bookingIdRaw ? parseInt(bookingIdRaw, 10) : NaN;

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [paymentBrief, setPaymentBrief] = useState(null);
  const [polling, setPolling] = useState(false);
  const [pollsDone, setPollsDone] = useState(0);
  const intervalRef = useRef(null);

  const fetchBooking = useCallback(async () => {
    const { data } = await apiClient.get(`/bookings/${bookingId}`);
    const booking = data?.data ?? data;
    const wp = booking?.worldline_payment ?? null;
    setPaymentBrief(wp);
    const st = (wp?.status || '').toLowerCase();
    const terminal = st === 'completed' || st === 'failed';
    return { terminal };
  }, [bookingId]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPolling(false);
  }, []);

  useEffect(() => {
    if (!bookingIdRaw || Number.isNaN(bookingId)) {
      setLoading(false);
      setError('Paramètre bookingId manquant ou invalide.');
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const { terminal } = await fetchBooking();
        if (cancelled) return;
        setError(null);
        if (terminal) {
          setLoading(false);
          return;
        }
        setLoading(false);
        setPolling(true);
        let ticks = 0;
        intervalRef.current = setInterval(async () => {
          if (cancelled) return;
          ticks += 1;
          setPollsDone(ticks);
          try {
            const { terminal: t } = await fetchBooking();
            if (t || ticks >= MAX_POLLS) {
              stopPolling();
            }
          } catch (e) {
            if (!cancelled) {
              setError(
                e?.response?.data?.message ||
                  'Erreur lors du rafraîchissement du statut.'
              );
              stopPolling();
            }
          }
        }, POLL_INTERVAL_MS);
      } catch (e) {
        if (!cancelled) {
          setError(
            e?.response?.data?.message ||
              'Impossible de charger le statut de la réservation.'
          );
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [bookingId, bookingIdRaw, fetchBooking, stopPolling]);

  const handleRefresh = async () => {
    setError(null);
    try {
      await fetchBooking();
    } catch (e) {
      setError(
        e?.response?.data?.message || 'Impossible de rafraîchir le statut.'
      );
    }
  };

  const publicId = localStorage.getItem('public_id');
  const dashboardLink = publicId ? `/dashboard/client/${publicId}` : '/dashboard';

  const payStatus = paymentBrief ? (paymentBrief.status || '').toLowerCase() : null;

  return (
    <div className={styles.page}>
      <HeaderDashboard userName="Client" />
      <main className={styles.main}>
        <h1 className={styles.title}>Paiement</h1>
        {loading && <p>Chargement du statut…</p>}
        {error && <p className={styles.error}>{error}</p>}
        {!loading && !error && (
          <div className={styles.card}>
            <p>
              Si vous venez de finaliser un paiement sur la page sécurisée Worldline, la confirmation
              peut prendre quelques instants. Cette page se met à jour automatiquement.
            </p>
            {polling && payStatus === 'pending' && (
              <p className={styles.polling}>
                Actualisation automatique… ({pollsDone}/{MAX_POLLS})
              </p>
            )}
            {paymentBrief ? (
              <p className={styles.status}>
                Statut paiement enregistré :{' '}
                <strong>
                  {payStatus === 'completed'
                    ? 'Payé'
                    : payStatus === 'failed'
                      ? 'Refusé ou annulé'
                      : payStatus === 'pending'
                        ? 'En attente de confirmation'
                        : payStatus || '—'}
                </strong>
              </p>
            ) : (
              <p className={styles.status}>Aucun paiement Worldline associé à cette réservation.</p>
            )}
            <p className={styles.hint}>Réservation n° {bookingId}</p>
            <button type="button" className={styles.refreshOutline} onClick={handleRefresh}>
              Actualiser maintenant
            </button>
          </div>
        )}
        <div className={styles.actions}>
          <Link to={dashboardLink} className={styles.link}>
            Retour au tableau de bord
          </Link>
          <Link to="/dashboard/bookings" className={styles.linkSecondary}>
            Mes réservations
          </Link>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default ClientWorldlinePaymentReturn;
