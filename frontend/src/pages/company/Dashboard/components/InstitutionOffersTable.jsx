// src/pages/company/Dashboard/components/InstitutionOffersTable.jsx
import React, { useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FiCheckCircle, FiXCircle, FiClock, FiInbox } from 'react-icons/fi';
import styles from './ReservationTable.module.css';
import BookingIdentityCell from '../../../../components/booking/BookingIdentityCell';
import { buildOfferIdentity } from '../../../../utils/bookingIdentity';
import { getCurrentAuthEnv } from '../../../../utils/apiClient';
import { formatMissionScheduleDetail, formatLegTime, formatDepartureTime, getConfirmedScheduleParts, formatSchedulePartLabel } from '../../../../utils/formatLegTime';
import { formatWallClockDateTime } from '../../../../utils/missionTimeDisplay';
import { canRespondToInstitutionOffer, isInstitutionOfferExpired, filterVisibleInstitutionOffers } from '../../../../utils/institutionOfferResponse';
import ProposeOfferTimeModal from './ProposeOfferTimeModal';

/** Formate une date/heure mission (heure murale Genève). */
const formatMissionDateTime = (isoString) => formatWallClockDateTime(isoString);

/** Formate un instant absolu (ex. expiration d'offre). */
const formatInstantDateTime = (isoString) => {
  if (!isoString) return { date: '—', time: '' };
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return { date: '—', time: '' };
  const pad = (n) => String(n).padStart(2, '0');
  return {
    date: `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`,
    time: `${pad(d.getHours())}:${pad(d.getMinutes())}`,
  };
};
// Nom court d'un point : établissement si dispo, sinon 1er segment de l'adresse.
const shortName = (address, establishment) => {
  if (establishment && establishment.trim()) return establishment.trim();
  if (!address) return '—';
  return address.split(',')[0].trim();
};

// Reconstruit le parcours complet d'une demande à partir des legs (multi-destinations + retour).
const getRoutePoints = (req) => {
  const legs = Array.isArray(req?.legs)
    ? [...req.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];
  if (legs.length > 0) {
    return [
      {
        label: 'Départ',
        address: legs[0].pickup_location,
        short: shortName(legs[0].pickup_location),
        kind: 'start',
      },
      ...legs.map((leg, index) => {
        const isReturn = Boolean(req?.return_to_institution) && legs.length > 1 && index === legs.length - 1;
        return {
          label: isReturn ? 'Retour' : `Destination ${index + 1}`,
          address: leg.dropoff_location,
          short: isReturn ? 'Retour institution' : shortName(leg.dropoff_location, leg.dropoff_establishment),
          kind: isReturn ? 'return' : 'destination',
          time: formatLegTime(leg),
        };
      }),
    ];
  }
  return [
    { label: 'Départ', address: req?.pickup_location, short: shortName(req?.pickup_location), kind: 'start', time: formatDepartureTime(req) },
    { label: 'Arrivée', address: req?.dropoff_location, short: shortName(req?.dropoff_location, req?.dropoff_establishment), kind: 'destination', time: formatLegTime({ scheduled_time: req?.scheduled_time, time_confirmed: req?.scheduled_time_type === 'arrival' }) },
  ];
};

/**
 * Tableau des offres de transport institutionnelles.
 * Affiche les demandes reçues d'institutions avec possibilité d'accepter/proposer/refuser.
 */
const InstitutionOffersTable = ({ offers = [], loading, onAccept, onReject }) => {
  const [proposeOffer, setProposeOffer] = useState(null);
  const navigate = useNavigate();
  const { public_id: companyPublicId } = useParams();

  // Clic sur une ligne → page Réservations filtrée sur le jour de la demande,
  // pour que l'entreprise voie son organisation de la journée avant d'accepter.
  const goToReservationsForDay = useCallback(
    (missionDateIso) => {
      if (!companyPublicId) return;
      const isDemoEnv = (getCurrentAuthEnv() || '').toLowerCase() === 'demo';
      const base = `${isDemoEnv ? '/demo/dashboard' : '/dashboard'}/company/${companyPublicId}`;
      const isValidDate = /^\d{4}-\d{2}-\d{2}$/.test(missionDateIso || '');
      navigate(isValidDate ? `${base}/reservations?date=${missionDateIso}` : `${base}/reservations`);
    },
    [companyPublicId, navigate],
  );

  // Masque les offres expirées depuis plus d'1h (on garde le statut "Expiré" le temps restant).
  const visibleOffers = React.useMemo(
    () => filterVisibleInstitutionOffers(offers),
    [offers],
  );

  if (loading) {
    return (
      <div className={styles.emptyState}>
        <FiInbox className={styles.emptyIcon} size={40} />
        <p className={styles.emptyTitle}>Chargement...</p>
        <p className={styles.emptySubtitle}>Récupération des demandes institutions</p>
      </div>
    );
  }

  if (!visibleOffers.length) {
    return (
      <div className={styles.emptyState}>
        <FiInbox className={styles.emptyIcon} size={40} />
        <p className={styles.emptyTitle}>Aucune demande d'institution en attente</p>
        <p className={styles.emptySubtitle}>Les nouvelles demandes apparaîtront ici automatiquement</p>
      </div>
    );
  }

  return (
    <>
      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.dateCell}>Date / Heure</th>
              <th>Passager</th>
              <th>Trajet</th>
              <th>Type</th>
              <th>Statut</th>
              <th className={styles.actionsCell}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {visibleOffers.map((offer) => {
              const req = offer.transport_request || {};
              const routePoints = getRoutePoints(req);
              const canRespond = canRespondToInstitutionOffer(offer);
              const isExpired = isInstitutionOfferExpired(offer);
              const scheduleDetail = req.scheduling?.summary
                ? { missionDate: req.scheduling.mission_date, summary: req.scheduling.summary }
                : formatMissionScheduleDetail(req);
              const confirmedParts = getConfirmedScheduleParts(req);
              const missionDateIso = req.mission_date || req.scheduling?.mission_date || scheduleDetail.missionDate;
              const date = missionDateIso
                ? formatMissionDateTime(`${missionDateIso}T12:00:00`).date
                : formatMissionDateTime(req.next_confirmed_time || req.scheduled_time).date;
              const time = confirmedParts.length
                ? confirmedParts.map(formatSchedulePartLabel).join(' · ')
                : '';
              const scheduleKindLabel = confirmedParts.length > 1
                ? confirmedParts.map((p) => p.label).join(' · ')
                : (confirmedParts[0]?.label || (req.scheduled_time_type === 'arrival' ? 'RDV' : 'Départ'));
              const mobility = req.mobility;
              const mobilityTags = [];
              if (mobility) {
                if (mobility.wheelchair) mobilityTags.push('Fauteuil');
                if (mobility.stretcher) mobilityTags.push('Brancard');
                if (mobility.oxygen) mobilityTags.push('O₂');
                if (mobility.walking_aid) mobilityTags.push('Aide marche');
              }

              return (
                <tr
                  key={offer.id}
                  className={styles.tableRow}
                  style={{ cursor: 'pointer' }}
                  onClick={() => goToReservationsForDay(missionDateIso)}
                  title="Voir mes réservations de ce jour"
                >
                  <td className={styles.dateCell}>
                    <div>{date !== '—' ? date : '—'}</div>
                    <div className={styles.cellPrimary} title={scheduleDetail.summary || undefined}>
                      {time || scheduleKindLabel || '—'}
                    </div>
                  </td>
                  <td className={styles.clientCell}>
                    <BookingIdentityCell identity={buildOfferIdentity(offer)} />
                  </td>
                  <td
                    className={styles.locationCell}
                    title={routePoints
                      .map((point) => `${point.label} : ${point.address || '—'}`)
                      .join('\n')}
                  >
                    {(() => {
                      const tripCount = Math.max(routePoints.length - 1, 1);
                      const isMulti = req.multi_stop || req.return_to_institution || tripCount > 1;
                      return (
                        <>
                          <div className={styles.offerRouteSummaryLine}>
                            {routePoints.map((point) => `${point.short}${point.time ? ` (${point.time})` : ''}`).join(' → ')}
                          </div>
                          {isMulti && (
                            <div className={styles.offerRouteSummaryHeader}>
                              Multi-destination · {tripCount} trajet{tripCount > 1 ? 's' : ''}
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </td>
                  <td>
                    <div style={{ fontSize: '13px' }}>
                      {req.mission_type === 'patient_transport'
                        ? 'Patient'
                        : req.mission_type || '—'}
                    </div>
                    {mobilityTags.length > 0 && (
                      <div
                        style={{
                          display: 'flex',
                          gap: 4,
                          flexWrap: 'wrap',
                          marginTop: 4,
                        }}
                      >
                        {mobilityTags.map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: '10px',
                              padding: '1px 6px',
                              borderRadius: 'var(--radius-full)',
                              background: 'var(--info-bg)',
                              color: 'var(--info-primary)',
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td>
                    <span className={`${styles.statusBadge} ${styles.pending}`}>
                      {canRespond ? 'En attente' : isExpired ? 'Expiré' : 'Indisponible'}
                    </span>
                    {offer.expires_at && (
                      <div className={styles.cellMeta}>
                        Exp: {formatInstantDateTime(offer.expires_at).date}{' '}
                        {formatInstantDateTime(offer.expires_at).time}
                      </div>
                    )}
                  </td>
                  <td className={styles.actionsCell}>
                    {canRespond ? (
                      <div style={{ display: 'flex', gap: '2px', alignItems: 'center', justifyContent: 'flex-end' }}>
                        {/* Accepter (avec l'horaire demandé) */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onAccept?.(offer.id);
                          }}
                          title="Accepter (horaire demandé)"
                          className={`${styles.actionButton} ${styles.acceptButton}`}
                        >
                          <FiCheckCircle size={18} />
                        </button>

                        {/* Proposer un horaire */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setProposeOffer(offer);
                          }}
                          title="Accepter avec un horaire différent"
                          className={styles.actionButton}
                          style={{ color: 'var(--brand-primary)' }}
                        >
                          <FiClock size={18} />
                        </button>

                        {/* Refuser */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onReject?.(offer.id);
                          }}
                          title="Refuser la demande"
                          className={`${styles.actionButton} ${styles.rejectButton}`}
                        >
                          <FiXCircle size={18} />
                        </button>
                      </div>
                    ) : (
                      <span style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>
                        {isExpired
                          ? 'Offre expirée, vous ne pouvez plus répondre.'
                          : 'Aucune action'}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Modale proposition d'horaire */}
      {proposeOffer && (
        <ProposeOfferTimeModal
          offer={proposeOffer}
          onConfirm={(offerId, isoTime) => {
            onAccept?.(offerId, isoTime);
            setProposeOffer(null);
          }}
          onClose={() => setProposeOffer(null)}
        />
      )}
    </>
  );
};

export default InstitutionOffersTable;
