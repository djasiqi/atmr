// src/pages/company/Dashboard/components/OverviewCards.jsx
// NIVEAU 1 — Situation temps réel : KPI opérationnels (préfère stats bootstrap exactes)
import React, { useMemo } from 'react';
import { FiClock, FiUsers, FiNavigation, FiAlertTriangle } from 'react-icons/fi';
import styles from './OverviewCards.module.css';

const toYMD = (raw) => {
  if (!raw) return null;
  if (typeof raw === 'string') {
    const m = raw.trim().match(/^(\d{4}-\d{2}-\d{2})/);
    if (m) return m[1];
  }
  try {
    const d = new Date(raw);
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  } catch {
    return null;
  }
};

const whenOf = (r) => r?.scheduled_time ?? r?.pickup_time ?? r?.date_time ?? r?.datetime ?? null;
const norm = (s) => String(s || '').toLowerCase();
const isActive = (s) =>
  ['en_route', 'in_progress', 'onboard', 'en_route_pickup', 'en_route_dropoff'].includes(norm(s));

const OverviewCards = ({
  reservations,
  pendingReservations,
  assignedReservations,
  driver,
  day,
  delayCount = 0,
  hasCriticalDelays = false,
  /** Stats exactes bootstrap (kpi) — priorité sur le recalcul liste. */
  kpiStats = null,
  delaysError = false,
  driversError = false,
  bookingsTruncated = false,
  bookingsLimit = null,
}) => {
  const dayList = useMemo(() => {
    const all = Array.isArray(reservations) ? reservations : [];
    if (!day) return all;
    return all.filter((r) => toYMD(whenOf(r)) === day);
  }, [reservations, day]);

  const inProgressCount = useMemo(() => {
    if (kpiStats && typeof kpiStats.in_service === 'number') return kpiStats.in_service;
    if (kpiStats && typeof kpiStats.inProgress === 'number') return kpiStats.inProgress;
    return dayList.filter((r) => isActive(r.status) && (r.driver_id || r.driver?.id)).length;
  }, [dayList, kpiStats]);

  const pendingDecision = useMemo(() => {
    if (kpiStats && typeof kpiStats.pending_decision === 'number') return kpiStats.pending_decision;
    if (Array.isArray(pendingReservations)) {
      const inDay = (r) => (!day ? true : toYMD(whenOf(r)) === day);
      return pendingReservations.filter(inDay).length;
    }
    return dayList.filter((r) => norm(r.status) === 'pending').length;
  }, [kpiStats, pendingReservations, dayList, day]);

  const unassignedCount = useMemo(() => {
    if (kpiStats && typeof kpiStats.unassigned === 'number') return kpiStats.unassigned;
    if (Array.isArray(assignedReservations)) {
      const inDay = (r) => (!day ? true : toYMD(whenOf(r)) === day);
      return assignedReservations.filter(inDay).length;
    }
    return dayList.filter((r) => {
      const s = norm(r.status);
      const unassigned = !r?.driver_id && !r?.driver?.id;
      return (s === 'accepted' || s === 'assigned') && unassigned;
    }).length;
  }, [kpiStats, assignedReservations, dayList, day]);

  const resolvedDelayCount = useMemo(() => {
    if (delaysError) return null;
    if (kpiStats && typeof kpiStats.delay_count === 'number') return kpiStats.delay_count;
    return delayCount;
  }, [delaysError, kpiStats, delayCount]);

  const criticalFromKpi =
    kpiStats && typeof kpiStats.critical_delay_count === 'number'
      ? kpiStats.critical_delay_count > 0
      : hasCriticalDelays;

  const availableDrivers = useMemo(
    () =>
      (Array.isArray(driver) ? driver : []).filter(
        (d) => d?.is_active && String(d?.status || '').toLowerCase() === 'available'
      ).length,
    [driver]
  );

  const totalDrivers = useMemo(
    () => (Array.isArray(driver) ? driver : []).filter((d) => d?.is_active).length,
    [driver]
  );

  const cards = [
    {
      id: 'inprogress',
      Icon: FiNavigation,
      label: 'En cours',
      value: inProgressCount,
      accent: inProgressCount > 0 ? 'brand' : 'default',
    },
    {
      id: 'delays',
      Icon: FiAlertTriangle,
      label: 'Urgences',
      value: delaysError ? '—' : resolvedDelayCount,
      accent: delaysError
        ? 'danger'
        : resolvedDelayCount === 0
          ? 'success'
          : criticalFromKpi
            ? 'danger'
            : 'warning',
      hint: delaysError ? 'Indisponible' : undefined,
    },
    {
      id: 'decide',
      Icon: FiClock,
      label: 'À décider',
      value: pendingDecision,
      accent: pendingDecision > 0 ? 'warning' : 'success',
    },
    {
      id: 'unassigned',
      Icon: FiClock,
      label: 'Sans chauffeur',
      value: unassignedCount,
      accent: unassignedCount > 0 ? 'warning' : 'success',
    },
    {
      id: 'drivers',
      Icon: FiUsers,
      label: 'Chauffeurs',
      value: driversError ? '—' : `${availableDrivers}/${totalDrivers}`,
      accent: driversError
        ? 'danger'
        : availableDrivers === 0 && totalDrivers > 0
          ? 'danger'
          : 'default',
      hint: driversError ? 'Indisponible' : undefined,
    },
  ];

  return (
    <div className={styles.kpiGrid} data-tour-id="kpi-grid">
      {bookingsTruncated ? (
        <div
          className={styles.truncationBanner}
          role="status"
          data-tour-id="bookings-truncated-banner"
        >
          Affichage limité
          {bookingsLimit != null ? ` à ${bookingsLimit} courses` : ''}
          {' — '}
          les KPI ci-dessus restent exacts.
        </div>
      ) : null}
      {cards.map((card) => {
        const accentClass = styles[`accent_${card.accent}`] || '';
        return (
          <div
            key={card.id}
            className={`${styles.kpiCard} ${accentClass}`}
            data-tour-id={`kpi-${card.id}`}
            title={card.hint}
          >
            <div className={styles.kpiIconContainer}>
              <card.Icon className={styles.kpiIcon} />
            </div>
            <div className={styles.kpiContent}>
              <span className={styles.kpiLabel}>{card.label}</span>
              <span className={styles.kpiValue}>{card.value}</span>
              {card.hint ? <span className={styles.kpiHint}>{card.hint}</span> : null}
            </div>
          </div>
        );
      })}
    </div>
  );
};

function driverKpiSignature(driver) {
  const list = Array.isArray(driver) ? driver : [];
  let active = 0;
  let available = 0;
  list.forEach((d) => {
    if (!d?.is_active) return;
    active += 1;
    if (String(d?.status || '').toLowerCase() === 'available') available += 1;
  });
  return `${active}:${available}`;
}

function areOverviewCardsPropsEqual(prev, next) {
  if (prev === next) return true;
  if (
    prev.reservations !== next.reservations
    || prev.day !== next.day
    || prev.delayCount !== next.delayCount
    || prev.hasCriticalDelays !== next.hasCriticalDelays
    || prev.pendingReservations !== next.pendingReservations
    || prev.assignedReservations !== next.assignedReservations
    || prev.kpiStats !== next.kpiStats
    || prev.delaysError !== next.delaysError
    || prev.driversError !== next.driversError
    || prev.bookingsTruncated !== next.bookingsTruncated
    || prev.bookingsLimit !== next.bookingsLimit
  ) {
    return false;
  }
  return driverKpiSignature(prev.driver) === driverKpiSignature(next.driver);
}

export default React.memo(OverviewCards, areOverviewCardsPropsEqual);
