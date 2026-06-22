// src/pages/company/Reservations/components/InstitutionOfferDetailPanel.jsx
import React from 'react';
import {
  FiCheckCircle, FiXCircle, FiClock, FiX, FiZap,
} from 'react-icons/fi';
import { FaRoute, FaInfoCircle, FaNotesMedical, FaWheelchair } from 'react-icons/fa';
import styles from './ReservationDetailPanel.module.css';
import { buildOfferIdentity } from '../../../../utils/bookingIdentity';
import {
  getConfirmedScheduleParts,
  formatSchedulePartLabel,
  formatRouteStopTime,
  formatReturnTimeLabel,
} from '../../../../utils/formatLegTime';
import { formatWallClockDateShort } from '../../../../utils/missionTimeDisplay';
import { institutionOfferEstimateLabel } from '../../../../utils/institutionOfferEstimateLabel';
import { canRespondToInstitutionOffer, isInstitutionOfferExpired } from '../../../../utils/institutionOfferResponse';
import { resolveInstitutionOfferActions } from '../../../../utils/institutionOfferActions';

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

const MISSION_LABELS = {
  patient_transport: 'Transport patient',
  material_delivery: 'Livraison matériel',
};

const formatMissionType = (value) => {
  if (!value) return '';
  if (MISSION_LABELS[value]) return MISSION_LABELS[value];
  const readable = String(value).replace(/_/g, ' ').trim();
  return readable ? readable.charAt(0).toUpperCase() + readable.slice(1) : '';
};

const getRoutePoints = (request) => {
  const legs = Array.isArray(request?.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];
  if (legs.length > 0) {
    return [
      {
        label: 'Départ',
        address: legs[0].pickup_location,
        kind: 'start',
        timeLabel: formatRouteStopTime({ kind: 'start', request }),
      },
      ...legs.map((leg, index) => {
        const isReturn = Boolean(request?.return_to_institution) && index === legs.length - 1;
        return {
          label: isReturn ? 'Retour' : `Destination ${index + 1}`,
          address: leg.dropoff_location,
          kind: isReturn ? 'return' : 'destination',
          timeLabel: formatRouteStopTime({
            kind: isReturn ? 'return' : 'destination',
            request,
            leg,
          }),
          details: {
            establishment: leg.dropoff_establishment,
            service: leg.dropoff_service,
            doctor: leg.dropoff_doctor,
          },
        };
      }),
    ];
  }
  const arrivalLeg = request?.scheduled_time_type === 'arrival' && request?.scheduled_time
    ? {
      scheduled_time: request.scheduled_time,
      time_confirmed: request.appointment_time_confirmed ?? true,
    }
    : null;
  return [
    {
      label: 'Départ',
      address: request?.pickup_location,
      kind: 'start',
      timeLabel: formatRouteStopTime({ kind: 'start', request }),
    },
    {
      label: 'Destination 1',
      address: request?.dropoff_location,
      kind: 'destination',
      timeLabel: formatRouteStopTime({ kind: 'destination', request, leg: arrivalLeg }),
    },
  ];
};

const getTripBadge = (request, routePoints) => {
  if (request?.return_to_institution) {
    return {
      className: 'roundTripBadge',
      label: `A/R institution — ${Math.max(routePoints.length - 1, 1)} trajet(s)`,
    };
  }
  if (request?.multi_stop || routePoints.length > 2) {
    return {
      className: 'multiStopBadge',
      label: `${routePoints.length - 1} destination(s)`,
    };
  }
  if (request?.is_round_trip || request?.round_trip) {
    const returnHint = formatReturnTimeLabel(request);
    return {
      className: 'roundTripBadge',
      label: `Aller-retour${returnHint ? ` — ${returnHint}` : ''}`,
    };
  }
  return {
    className: 'oneWayBadge',
    label: 'Aller simple',
  };
};

const renderRouteStopTime = (point) => {
  const raw = point.timeLabel;
  if (!raw) return null;
  const time = point.label && raw.startsWith(point.label)
    ? raw.slice(point.label.length).replace(/^[\s·]+/, '')
    : raw;
  return time ? (
    <span className={styles.routeStopTime}> · {time}</span>
  ) : null;
};

const InstitutionOfferDetailPanel = ({
  offer, onClose, onValidate, onPlan, onAcceptNow, onReject,
}) => {
  const req = offer?.transport_request || {};
  const identity = buildOfferIdentity(offer);
  const routePoints = getRoutePoints(req);
  const tripBadge = getTripBadge(req, routePoints);

  const parts = getConfirmedScheduleParts(req);
  const timeLabel = parts.length ? parts.map(formatSchedulePartLabel).join(' · ') : '';
  const dateIso = req.mission_date || req.scheduled_time;
  const dateShort = dateIso ? formatWallClockDateShort(dateIso) : '';
  const scheduleLabel = timeLabel
    ? (dateShort ? `${dateShort} · ${timeLabel}` : timeLabel)
    : (dateShort ? `${dateShort} · Horaire à définir` : 'À définir');

  const mob = req.mobility || {};
  const hasWheelchair = req.requires_wheelchair || mob.wheelchair;
  const hasVehicleWheelchair = mob.vehicle_wheelchair;
  const hasAssistance = req.requires_assistance || mob.needs_assistance;
  const assistanceType = (mob.assistance_type || '').trim();
  const hasStretcher = req.requires_stretcher || mob.stretcher;
  const hasOxygen = req.requires_oxygen || mob.oxygen;
  const hasNeedsSection = hasWheelchair || hasVehicleWheelchair || hasAssistance
    || hasStretcher || hasOxygen || req.notes;

  const est = offer?.price_estimate || null;
  const estAmount = est ? Number(est.amount) : NaN;
  const hasEstimate = est && !Number.isNaN(estAmount) && estAmount > 0;
  const billingIntent = req.billing_intent || 'patient';
  const estLabel = institutionOfferEstimateLabel(est, billingIntent);

  const passengerTitle = identity.passengerLabel || 'Demande institution';
  const passengerBirthDate = req.patient?.dob
    || req.patient?.birth_date
    || req.patient_date_of_birth
    || null;
  const canRespond = canRespondToInstitutionOffer(offer);
  const offerActions = resolveInstitutionOfferActions(offer);
  const isExpired = isInstitutionOfferExpired(offer);
  const statusLabel = canRespond ? 'En attente' : isExpired ? 'Expiré' : 'Indisponible';
  const statusClass = canRespond
    ? styles.badgeStatusWarning
    : isExpired
      ? styles.badgeStatusExpired
      : styles.badgeStatusNeutral;
  const expiresMeta = offer?.expires_at ? formatInstantDateTime(offer.expires_at) : null;

  const planIsPrimary = offerActions.canPlan
    && !offerActions.canValidate
    && !offerActions.canAcceptNow;

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <div className={styles.panelTitleRow}>
          <div className={styles.panelTitleStack}>
            <span className={styles.panelTitle}>{passengerTitle}</span>
            {expiresMeta && (
              <span className={styles.panelMeta}>
                Exp: {expiresMeta.date} {expiresMeta.time}
              </span>
            )}
          </div>
          <span className={`${styles.statusBadge} ${statusClass}`}>{statusLabel}</span>
        </div>
        <button type="button" className={styles.closeBtn} onClick={onClose} title="Fermer" aria-label="Fermer">
          <FiX size={16} />
        </button>
      </div>

      <div className={styles.panelBody}>
        {canRespond && offerActions.canRespond ? (
          <div className={styles.offerActionsBlock}>
            {offerActions.hint ? (
              <p className={styles.offerActionHint}>{offerActions.hint}</p>
            ) : null}
            <div className={styles.offerActionsRow}>
              {offerActions.canValidate ? (
                <button
                  type="button"
                  onClick={() => onValidate?.(offer)}
                  className={`${styles.actionBtn} ${styles.actionBtnOffer} ${styles.btnPrimary}`}
                >
                  <FiCheckCircle size={14} aria-hidden />
                  {offerActions.validateLabel}
                </button>
              ) : null}
              {offerActions.canAcceptNow ? (
                <button
                  type="button"
                  onClick={() => onAcceptNow?.(offer)}
                  className={`${styles.actionBtn} ${styles.actionBtnOffer} ${styles.btnAcceptNow}`}
                >
                  <FiZap size={14} aria-hidden />
                  {offerActions.acceptNowLabel}
                </button>
              ) : null}
              {offerActions.canPlan ? (
                <button
                  type="button"
                  onClick={() => onPlan?.(offer)}
                  className={`${styles.actionBtn} ${styles.actionBtnOffer} ${
                    planIsPrimary ? styles.btnPrimary : styles.btnSecondary
                  }`}
                >
                  <FiClock size={14} aria-hidden />
                  {offerActions.planLabel}
                </button>
              ) : null}
              {offerActions.canReject ? (
                <button
                  type="button"
                  onClick={() => onReject?.(offer.id)}
                  className={`${styles.actionBtn} ${styles.actionBtnOffer} ${styles.actionBtnReject}`}
                  title="Refuser la demande"
                >
                  <FiXCircle size={14} aria-hidden />
                  {offerActions.rejectLabel}
                </button>
              ) : null}
            </div>
          </div>
        ) : (
          <div className={styles.offerExpiredNotice}>
            <span className={styles.noActionLabel}>
              {isExpired
                ? 'Offre expirée, vous ne pouvez plus répondre.'
                : 'Aucune action disponible.'}
            </span>
          </div>
        )}

        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={`${styles.sectionIcon} ${styles.sectionIconBrand}`}><FaInfoCircle /></div>
            <h3 className={styles.sectionTitle}>Informations</h3>
          </div>
          <div className={styles.summaryGrid}>
            <div className={styles.summaryItem}>
              <span className={styles.summaryLabel}>Passager</span>
              <span className={styles.summaryValue}>{identity.passengerLabel || '—'}</span>
            </div>
            {passengerBirthDate && (
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Date de naissance</span>
                <span className={styles.summaryValue}>
                  {new Date(passengerBirthDate).toLocaleDateString('fr-CH')}
                </span>
              </div>
            )}
            <div className={styles.summaryItem}>
              <span className={styles.summaryLabel}>Origine</span>
              <span className={styles.summaryValue}>{identity.source?.name || 'Institution'}</span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.summaryLabel}>Horaire</span>
              <span className={styles.summaryValue}>{scheduleLabel}</span>
            </div>
            {req.mission_type && (
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Type</span>
                <span className={styles.summaryValue}>{formatMissionType(req.mission_type)}</span>
              </div>
            )}
            <div className={styles.summaryItem}>
              <span className={styles.summaryLabel}>{estLabel}</span>
              <span className={styles.summaryValue}>
                {hasEstimate
                  ? `${estAmount.toFixed(2)} ${est.currency || 'CHF'}`
                  : 'À définir à l\u2019acceptation'}
              </span>
            </div>
          </div>
        </div>

        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={`${styles.sectionIcon} ${styles.sectionIconBrand}`}><FaRoute /></div>
            <h3 className={styles.sectionTitle}>Trajet</h3>
          </div>
          <div className={styles.routeTimeline}>
            {routePoints.map((point, index) => {
              const isFirst = index === 0;
              const isLast = index === routePoints.length - 1;
              const dotClass = isFirst
                ? styles.routeDotStart
                : isLast
                  ? styles.routeDotEnd
                  : styles.routeDotMid;
              const hasDetails = point.details
                && (point.details.establishment || point.details.service || point.details.doctor);
              return (
                <div
                  className={styles.routeTimelineStop}
                  key={`stop-${point.kind}-${index}-${point.address || ''}`}
                >
                  <div className={styles.routeMarker}>
                    <span className={`${styles.routeDot} ${dotClass}`} />
                    {!isLast && <span className={styles.routeConnector} />}
                  </div>
                  <div className={styles.routeStopBody}>
                    <div className={styles.routeStopLabel}>
                      {point.label}
                      {renderRouteStopTime(point)}
                    </div>
                    <div className={styles.routeStopAddress}>{point.address || '—'}</div>
                    {hasDetails && (
                      <div className={styles.routeStopDetails}>
                        {[point.details.establishment, point.details.service, point.details.doctor]
                          .filter(Boolean)
                          .join(' · ')}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className={styles[tripBadge.className]}>{tripBadge.label}</div>
        </div>

        {hasNeedsSection && (
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <div className={`${styles.sectionIcon} ${styles.sectionIconMuted}`}><FaNotesMedical /></div>
              <h3 className={styles.sectionTitle}>Besoins</h3>
            </div>
            <div className={styles.needsRow}>
              {hasWheelchair && (
                <span className={`${styles.needsChip} ${styles.needsChipActive}`}>
                  <FaWheelchair size={10} /> Fauteuil
                </span>
              )}
              {hasVehicleWheelchair && (
                <span className={`${styles.needsChip} ${styles.needsChipActive}`}>Prendre chaise</span>
              )}
              {hasAssistance && (
                <span className={`${styles.needsChip} ${styles.needsChipActive}`}>Assistance</span>
              )}
              {hasStretcher && (
                <span className={`${styles.needsChip} ${styles.needsChipActive}`}>Brancard</span>
              )}
              {hasOxygen && (
                <span className={`${styles.needsChip} ${styles.needsChipDanger}`}>O₂</span>
              )}
            </div>
            {hasAssistance && assistanceType && (
              <div className={styles.routeStopDetails} style={{ marginTop: 6 }}>
                Type d&apos;assistance : {assistanceType}
              </div>
            )}
            {req.notes && (
              <div className={styles.notesBlock}>{req.notes}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default InstitutionOfferDetailPanel;
