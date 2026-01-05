/**
 * VirtualizedReservationsList.jsx
 *
 * Composant virtualisé pour la liste des réservations (cartes) utilisant react-window.
 *
 * Ce composant virtualise le rendu des cartes de réservations pour améliorer
 * les performances avec de grandes listes.
 *
 * @module components/virtualized/VirtualizedReservationsList
 */

import React, { useCallback, useMemo, useRef } from 'react';
import PropTypes from 'prop-types';
import { List } from 'react-window';
import { FaMapMarkerAlt, FaCalendarAlt, FaMoneyBillWave } from 'react-icons/fa';
import styles from './VirtualizedReservationsList.module.css';

/**
 * Composant pour une carte de réservation
 */
const ReservationCard = ({ index, style, data }) => {
  const { bookings, onCancelBooking, isPast } = data;
  const booking = bookings[index];

  const handleCancel = useCallback(() => {
    if (onCancelBooking && booking) {
      onCancelBooking(booking.id);
    }
  }, [booking, onCancelBooking]);

  if (!booking) return null;

  const status = (booking.status || '').toLowerCase();

  return (
    <div style={style}>
      <div className={styles.reservationCard}>
        <h3>
          <FaCalendarAlt /> {new Date(booking.scheduled_time).toLocaleDateString('fr-FR')}
        </h3>
        <p>
          <FaMapMarkerAlt /> <strong>Départ :</strong> {booking.pickup_location || 'Inconnu'}
        </p>
        <p>
          <FaMapMarkerAlt /> <strong>Arrivée :</strong> {booking.dropoff_location || 'Inconnu'}
        </p>
        {!isPast && (
          <>
            <p>
              🚖 <strong>Entreprise :</strong> {booking.company_name || '—'}
            </p>
            <p>
              👨‍✈️ <strong>Chauffeur :</strong> {booking.driver_name || '—'}
            </p>
          </>
        )}
        <p>
          <FaMoneyBillWave /> <strong>Montant :</strong>{' '}
          {status === 'canceled'
            ? '0 CHF'
            : booking.amount
              ? `${booking.amount} CHF`
              : 'N/A'}
        </p>
        <p>
          <strong>Statut :</strong>{' '}
          <span
            className={
              status === 'completed'
                ? styles.statusCompleted
                : status === 'in_progress'
                  ? styles.statusInProgress
                  : status === 'canceled'
                    ? styles.statusCanceled
                    : styles.statusDefault
            }
          >
            {status === 'completed'
              ? '✅ Terminé'
              : status === 'in_progress'
                ? '🚖 En cours'
                : status === 'canceled'
                  ? '❌ Annulé'
                  : '🔄 En attente'}
          </span>
        </p>
        {!isPast && status !== 'canceled' && (
          <button
            className={styles.cancelBtn}
            onClick={handleCancel}
            disabled={booking.isCancelling}
          >
            {booking.isCancelling ? 'Annulation...' : 'Annuler'}
          </button>
        )}
      </div>
    </div>
  );
};

ReservationCard.propTypes = {
  index: PropTypes.number.isRequired,
  style: PropTypes.object.isRequired,
  data: PropTypes.shape({
    bookings: PropTypes.array.isRequired,
    onCancelBooking: PropTypes.func,
    isPast: PropTypes.bool,
  }).isRequired,
};

/**
 * Composant VirtualizedReservationsList
 *
 * Version virtualisée de la liste de réservations qui n'affiche que les cartes visibles
 * dans le viewport.
 *
 * @param {Object} props - Props du composant
 * @param {Array} props.bookings - Liste des bookings/réservations (déjà filtrées/triées)
 * @param {Function} props.onCancelBooking - Callback pour annuler une réservation
 * @param {boolean} props.isPast - Indique si c'est la liste des réservations passées
 *
 * @returns {JSX.Element} Composant virtualisé
 */
const VirtualizedReservationsList = ({ bookings, onCancelBooking, isPast = false }) => {
  const listRef = useRef(null);
  const rowHeightsRef = useRef({});

  // Hauteur estimée par défaut (variable selon le contenu)
  // Cartes à venir : ~250px (avec entreprise et chauffeur)
  // Cartes passées : ~200px (sans entreprise et chauffeur)
  const DEFAULT_ROW_HEIGHT = isPast ? 200 : 250;

  const getItemSize = useCallback(
    (index) => {
      // Retourner la hauteur stockée ou la hauteur par défaut
      return rowHeightsRef.current[index] || DEFAULT_ROW_HEIGHT;
    },
    [DEFAULT_ROW_HEIGHT]
  );

  // ✅ Données pour react-window - TOUJOURS un objet valide (jamais null/undefined/array)
  const itemData = useMemo(
    () => ({
      bookings: Array.isArray(bookings) ? bookings : [],
      onCancelBooking: typeof onCancelBooking === 'function' ? onCancelBooking : () => {},
      isPast: Boolean(isPast),
    }),
    [bookings, onCancelBooking, isPast]
  );

  // Hauteur du conteneur (affiche ~3 cartes)
  const CONTAINER_HEIGHT = 600;

  if (bookings.length === 0) {
    return <p className={styles.emptyMessage}>Aucune course {isPast ? 'passée' : 'à venir'}.</p>;
  }

  return (
    <div className={styles.reservationList}>
      <List
        ref={listRef}
        height={Math.min(CONTAINER_HEIGHT, bookings.length * DEFAULT_ROW_HEIGHT)}
        itemCount={bookings.length}
        itemSize={getItemSize}
        width="100%"
        itemData={itemData}
        className={styles.virtualizedList}
      >
        {ReservationCard}
      </List>
    </div>
  );
};

VirtualizedReservationsList.propTypes = {
  bookings: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      scheduled_time: PropTypes.string.isRequired,
      pickup_location: PropTypes.string,
      dropoff_location: PropTypes.string,
      company_name: PropTypes.string,
      driver_name: PropTypes.string,
      amount: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      status: PropTypes.string,
      isCancelling: PropTypes.bool,
    })
  ).isRequired,
  onCancelBooking: PropTypes.func,
  isPast: PropTypes.bool,
};

export default VirtualizedReservationsList;

