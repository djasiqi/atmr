// src/pages/company/Dashboard/components/InstitutionOffersTable.jsx
import React, { useState } from 'react';
import { FiCheckCircle, FiXCircle, FiClock, FiInbox } from 'react-icons/fi';
import styles from './ReservationTable.module.css';
import BookingIdentityCell from '../../../../components/booking/BookingIdentityCell';
import { buildOfferIdentity } from '../../../../utils/bookingIdentity';
import { formatMissionScheduleDetail, formatLegTime, formatDepartureTime } from '../../../../utils/formatLegTime';

/**
 * Formate une date ISO en "DD.MM.YYYY" et "HH:MM"
 */
const formatDateTime = (isoString) => {
  if (!isoString) return { date: '—', time: '' };
  const d = new Date(isoString);
  if (isNaN(d.getTime())) return { date: '—', time: '' };
  const pad = (n) => String(n).padStart(2, '0');
  return {
    date: `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`,
    time: `${pad(d.getHours())}:${pad(d.getMinutes())}`,
  };
};

/**
 * Convertit une date ISO en valeur pour input datetime-local (YYYY-MM-DDTHH:MM)
 */
/** Fallback client si can_respond absent ou incohérent avec expires_at. */
const canRespondToOffer = (offer) => {
  if (typeof offer?.can_respond === 'boolean') {
    if (!offer.can_respond) return false;
  } else if (offer?.status && offer.status !== 'PENDING') {
    return false;
  }
  if (offer?.expires_at) {
    return new Date(offer.expires_at) > new Date();
  }
  return offer?.can_respond !== false;
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
        const isReturn = Boolean(req?.return_to_institution) && index === legs.length - 1;
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

const toDatetimeLocal = (isoString) => {
  if (!isoString) return '';
  const d = new Date(isoString);
  if (isNaN(d.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

/**
 * Modale inline pour proposer un horaire de prise en charge
 */
const ProposeTimeModal = ({ offer, onConfirm, onClose }) => {
  const req = offer?.transport_request || {};
  const [proposedTime, setProposedTime] = useState(
    toDatetimeLocal(req.scheduled_time) || ''
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!proposedTime) return;
    // datetime-local → interprétation locale navigateur → ISO UTC pour le backend
    const dt = new Date(proposedTime);
    if (Number.isNaN(dt.getTime())) return;
    onConfirm(offer.id, dt.toISOString());
  };

  return (
    <div className={styles.proposeTimeOverlay} onClick={onClose}>
      <div className={styles.proposeTimeDialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ margin: '0 0 16px', fontSize: '16px' }}>
          Proposer un horaire de prise en charge
        </h3>

        {/* Résumé de la demande */}
        <div
          style={{
            background: 'var(--bg-secondary)',
            borderRadius: 'var(--radius-md)',
            padding: '12px',
            marginBottom: '16px',
            fontSize: '13px',
          }}
        >
          <div>
            <strong>{req.institution_name || 'Institution'}</strong>
          </div>
          <div style={{ marginTop: 4 }}>
            Horaire demandé :{' '}
            <strong>
              {formatDateTime(req.scheduled_time).date}{' '}
              {formatDateTime(req.scheduled_time).time}
            </strong>
          </div>
          <div style={{ marginTop: 4, color: 'var(--text-tertiary)' }}>
            {req.pickup_location} → {req.dropoff_location}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <label
            style={{
              display: 'block',
              marginBottom: '8px',
              fontWeight: 600,
              fontSize: '13px',
            }}
          >
            Horaire proposé
          </label>
          <input
            type="datetime-local"
            value={proposedTime}
            onChange={(e) => setProposedTime(e.target.value)}
            required
            style={{
              width: '100%',
              padding: '10px 12px',
              border: '1px solid var(--border-primary)',
              borderRadius: 'var(--radius-md)',
              fontSize: '14px',
              marginBottom: '16px',
              boxSizing: 'border-box',
            }}
          />

          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '8px 16px',
                border: '1px solid var(--border-primary)',
                borderRadius: 'var(--radius-md)',
                background: 'white',
                cursor: 'pointer',
                fontSize: '13px',
              }}
            >
              Annuler
            </button>
            <button
              type="submit"
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                background: 'var(--brand-primary)',
                color: 'white',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '13px',
              }}
            >
              Accepter avec cet horaire
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

/**
 * Tableau des offres de transport institutionnelles.
 * Affiche les demandes reçues d'institutions avec possibilité d'accepter/proposer/refuser.
 */
// Durée d'affichage d'une offre expirée avant masquage automatique (1h).
const EXPIRED_OFFER_VISIBLE_MS = 60 * 60 * 1000;

const InstitutionOffersTable = ({ offers = [], loading, onAccept, onReject }) => {
  const [proposeOffer, setProposeOffer] = useState(null);

  // Masque les offres expirées depuis plus d'1h (on garde le statut "Expiré" le temps restant).
  const visibleOffers = React.useMemo(() => {
    const now = Date.now();
    return (offers || []).filter((offer) => {
      if (!offer?.expires_at) return true;
      const expiresAt = new Date(offer.expires_at).getTime();
      if (Number.isNaN(expiresAt)) return true;
      if (expiresAt > now) return true; // pas encore expirée
      return now - expiresAt <= EXPIRED_OFFER_VISIBLE_MS;
    });
  }, [offers]);

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
              const canRespond = canRespondToOffer(offer);
              const isExpired =
                offer.expires_at && new Date(offer.expires_at) <= new Date();
              const scheduleDetail = req.scheduling?.summary
                ? { missionDate: req.scheduling.mission_date, summary: req.scheduling.summary }
                : formatMissionScheduleDetail(req);
              const departureTime = req.scheduling?.departure?.display_time;
              const { date, time } = departureTime
                ? { date: scheduleDetail.missionDate?.split('-').reverse().join('.') || formatDateTime(req.scheduled_time).date, time: departureTime }
                : formatDateTime(req.next_confirmed_time || req.scheduled_time);
              // Affichage compact : l'heure principale est déjà visible, on n'indique
              // que sa nature (départ vs rendez-vous) plutôt que le résumé détaillé.
              const scheduleKindLabel = (!departureTime && req.scheduled_time_type === 'arrival')
                ? 'Rendez-vous'
                : 'Départ';
              const mobility = req.mobility;
              const mobilityTags = [];
              if (mobility) {
                if (mobility.wheelchair) mobilityTags.push('Fauteuil');
                if (mobility.stretcher) mobilityTags.push('Brancard');
                if (mobility.oxygen) mobilityTags.push('O₂');
                if (mobility.walking_aid) mobilityTags.push('Aide marche');
              }

              return (
                <tr key={offer.id} className={styles.tableRow}>
                  <td className={styles.dateCell}>
                    <div>{date !== '—' ? date : (scheduleDetail.missionDate || '—')}</div>
                    <div className={styles.cellPrimary}>{time || '—'}</div>
                    {time && time !== '—' && (
                      <div className={styles.cellMeta} title={scheduleDetail.summary || undefined}>
                        {scheduleKindLabel}
                      </div>
                    )}
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
                        Exp: {formatDateTime(offer.expires_at).date}{' '}
                        {formatDateTime(offer.expires_at).time}
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
        <ProposeTimeModal
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
