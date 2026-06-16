// src/pages/company/Dashboard/components/ProposeOfferTimeModal.jsx
import React, { useEffect, useRef, useState } from 'react';
import styles from './ReservationTable.module.css';
import {
  formatMissionScheduleDetail,
  getConfirmedScheduleParts,
  formatSchedulePartLabel,
  getNextConfirmedScheduleInfo,
} from '../../../../utils/formatLegTime';
import {
  formatWallClockDateTime,
  combineMissionDateTimeNaive,
  toDatetimeLocalGeneva,
} from '../../../../utils/missionTimeDisplay';
import {
  fetchRouteTravelMinutes,
  formatOutboundRouteLabel,
} from '../../../../utils/routeTravelEstimate';

/** Décale une valeur datetime-local (YYYY-MM-DDTHH:mm) de N minutes (heure murale). */
const shiftDatetimeLocalMinutes = (value, deltaMinutes) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(value || '');
  if (!m) return value;
  const dt = new Date(
    Number(m[1]),
    Number(m[2]) - 1,
    Number(m[3]),
    Number(m[4]),
    Number(m[5]),
  );
  dt.setMinutes(dt.getMinutes() + deltaMinutes);
  const pad = (n) => String(n).padStart(2, '0');
  return (
    `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}` +
    `T${pad(dt.getHours())}:${pad(dt.getMinutes())}`
  );
};

/**
 * Calcule la valeur par défaut du champ datetime-local à partir de l'horaire
 * demandé. Si l'horaire demandé est une heure d'arrivée (RDV), la prise en
 * charge proposée par défaut = RDV − temps de trajet estimé.
 */
const computeDefaultProposedTime = (req, scheduleInfo, travelMinutes) => {
  const sourceIso = scheduleInfo?.iso || req.next_confirmed_time || req.scheduled_time;
  if (!sourceIso || !Number.isFinite(travelMinutes)) return '';
  const base = toDatetimeLocalGeneva(sourceIso);
  if (!base) return '';
  const isArrival =
    scheduleInfo?.kind === 'destination' ||
    scheduleInfo?.kind === 'return' ||
    (!scheduleInfo && req.scheduled_time_type === 'arrival');
  return isArrival ? shiftDatetimeLocalMinutes(base, -travelMinutes) : base;
};

/**
 * Modale « Proposer un horaire » pour une offre institution.
 * Permet au transporteur de confirmer / proposer une heure de prise en charge
 * lorsque l'institution attend une confirmation d'horaire.
 */
const ProposeOfferTimeModal = ({ offer, onConfirm, onClose }) => {
  const req = offer?.transport_request || {};
  const scheduleDetail = formatMissionScheduleDetail(req);

  const [travelMinutes, setTravelMinutes] = useState(null);
  const [travelLoading, setTravelLoading] = useState(true);
  const [proposedTime, setProposedTime] = useState('');
  const userEditedTimeRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    userEditedTimeRef.current = false;
    setTravelLoading(true);
    setProposedTime('');

    const transportReq = offer?.transport_request || {};
    const nextScheduleInfo = getNextConfirmedScheduleInfo(transportReq);

    fetchRouteTravelMinutes(transportReq, offer?.id).then((minutes) => {
      if (cancelled) return;
      setTravelMinutes(minutes);
      setTravelLoading(false);
      if (minutes != null) {
        setProposedTime(
          computeDefaultProposedTime(transportReq, nextScheduleInfo, minutes),
        );
      }
    }).catch(() => {
      if (cancelled) return;
      setTravelMinutes(null);
      setTravelLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [offer?.id, offer?.transport_request]);

  const requestedParts = getConfirmedScheduleParts(req);
  const requestedRows = requestedParts.map(formatSchedulePartLabel).join(' · ');
  const missionDateIso = req.mission_date || scheduleDetail.missionDate;
  const missionDateLabel = missionDateIso
    ? formatWallClockDateTime(`${missionDateIso}T12:00:00`).date
    : '—';

  const travelLabel = travelLoading
    ? 'calcul…'
    : travelMinutes != null
      ? `~${travelMinutes} min`
      : 'non disponible';

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!proposedTime) return;
    const naiveIso = combineMissionDateTimeNaive(
      proposedTime.slice(0, 10),
      proposedTime.slice(11, 16),
    );
    if (!naiveIso) return;
    onConfirm(offer.id, naiveIso);
  };

  return (
    <div className={styles.proposeTimeOverlay} onClick={onClose}>
      <div className={styles.proposeTimeDialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ margin: '0 0 16px', fontSize: '16px' }}>
          Proposer un horaire de prise en charge
        </h3>

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
              {missionDateLabel !== '—' ? `${missionDateLabel}` : ''}
              {requestedRows
                ? `${missionDateLabel !== '—' ? ' · ' : ''}${requestedRows}`
                : (missionDateLabel === '—' ? ' —' : '')}
            </strong>
          </div>
          <div style={{ marginTop: 4, color: 'var(--text-tertiary)' }}>
            {formatOutboundRouteLabel(req)}
          </div>
          <div style={{ marginTop: 4, color: 'var(--text-tertiary)' }}>
            Trajet estimé (Google Maps) : <strong>{travelLabel}</strong>
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
            onChange={(e) => {
              userEditedTimeRef.current = true;
              setProposedTime(e.target.value);
            }}
            required
            disabled={travelLoading && !proposedTime}
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
              disabled={!proposedTime}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                background: 'var(--brand-primary)',
                color: 'white',
                cursor: proposedTime ? 'pointer' : 'not-allowed',
                opacity: proposedTime ? 1 : 0.6,
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

export default ProposeOfferTimeModal;
