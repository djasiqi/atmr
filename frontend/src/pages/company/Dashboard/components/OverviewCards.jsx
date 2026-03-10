// src/pages/company/Dashboard/components/OverviewCards.jsx
// NIVEAU 1 — Situation temps réel : 4 KPI opérationnels uniquement
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
  ['en_route', 'in_progress', 'onboard', 'en_route_pickup', 'en_route_dropoff', 'assigned'].includes(norm(s));

const OverviewCards = ({
  reservations,
  pendingReservations,
  assignedReservations,
  driver,
  day,
  delayCount = 0,
  hasCriticalDelays = false,
}) => {
  const dayList = useMemo(() => {
    const all = Array.isArray(reservations) ? reservations : [];
    if (!day) return all;
    return all.filter((r) => toYMD(whenOf(r)) === day);
  }, [reservations, day]);

  const inProgressCount = useMemo(
    () => dayList.filter((r) => isActive(r.status) && (r.driver_id || r.driver?.id)).length,
    [dayList]
  );

  const waitingCount = useMemo(() => {
    if (Array.isArray(pendingReservations) || Array.isArray(assignedReservations)) {
      const p = Array.isArray(pendingReservations) ? pendingReservations : [];
      const a = Array.isArray(assignedReservations) ? assignedReservations : [];
      const inDay = (r) => (!day ? true : toYMD(whenOf(r)) === day);
      return p.filter(inDay).length + a.filter(inDay).length;
    }
    return dayList.filter((r) => {
      const s = norm(r.status);
      const unassigned = !r?.driver_id && !r?.driver?.id;
      return (s === 'pending' || s === 'accepted') && unassigned;
    }).length;
  }, [dayList, pendingReservations, assignedReservations, day]);

  const availableDrivers = useMemo(
    () => (Array.isArray(driver) ? driver : []).filter((d) => d?.is_active && d?.is_available).length,
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
      label: 'Retards',
      value: delayCount,
      accent: delayCount === 0 ? 'success' : hasCriticalDelays ? 'danger' : 'warning',
    },
    {
      id: 'waiting',
      Icon: FiClock,
      label: 'À assigner',
      value: waitingCount,
      accent: waitingCount > 0 ? 'warning' : 'success',
    },
    {
      id: 'drivers',
      Icon: FiUsers,
      label: 'Chauffeurs',
      value: `${availableDrivers}/${totalDrivers}`,
      accent: availableDrivers === 0 && totalDrivers > 0 ? 'danger' : 'default',
    },
  ];

  return (
    <div className={styles.kpiGrid} data-tour-id="kpi-grid">
      {cards.map((card) => {
        const accentClass = styles[`accent_${card.accent}`] || '';
        return (
          <div
            key={card.id}
            className={`${styles.kpiCard} ${accentClass}`}
            data-tour-id={`kpi-${card.id}`}
          >
            <div className={styles.kpiIconContainer}>
              <card.Icon className={styles.kpiIcon} />
            </div>
            <div className={styles.kpiContent}>
              <span className={styles.kpiLabel}>{card.label}</span>
              <span className={styles.kpiValue}>{card.value}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default OverviewCards;
