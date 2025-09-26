// src/pages/company/Dashboard/components/OverviewCards.jsx
import React, { useMemo, useEffect } from "react";
import styles from "../CompanyDashboard.module.css";

const toYMD = (raw) => {
  if (!raw) return null;
  if (typeof raw === "string") {
    const m = raw.trim().match(/^(\d{4}-\d{2}-\d{2})/);
    if (m) return m[1];
  }
  try {
    const d = new Date(raw);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  } catch {
    return null;
  }
};

// meilleur champ date disponible
const whenOf = (r) =>
  r?.scheduled_time ?? r?.pickup_time ?? r?.date_time ?? r?.datetime ?? null;

const norm = (s) => String(s || "").toLowerCase();
const isCompleted = (s) => ["completed", "return_completed", "done", "finished"].includes(norm(s));

const amountOf = (r) =>
  Number(
    r?.amount ??
    r?.total_amount ??
    r?.total ??
    r?.price ??
    r?.fare ??
    0
  ) || 0;

const OverviewCards = ({
  reservations,
  pendingReservations,        // déjà filtrées dans CompanyDashboard
  assignedReservations,       // accepted && !driver_id
  driver,
  day,                        // optionnel 'YYYY-MM-DD'
}) => {
  // Filtre jour (si fourni) — calcule 'all' *dans* le useMemo pour satisfaire eslint react-hooks/exhaustive-deps
  const dayList = useMemo(() => {
    const all = Array.isArray(reservations) ? reservations : [];
    if (!day) return all;
    return all.filter((r) => toYMD(whenOf(r)) === day);
  }, [reservations, day]);

  // Attente = pending + accepted sans driver (si on nous les passe), sinon heuristique
  const waitingCount = useMemo(() => {
    if (Array.isArray(pendingReservations) || Array.isArray(assignedReservations)) {
      const p = Array.isArray(pendingReservations) ? pendingReservations : [];
      const a = Array.isArray(assignedReservations) ? assignedReservations : [];
      // Si day est donné, restreindre
      const inDay = (r) => (!day ? true : toYMD(whenOf(r)) === day);
      return p.filter(inDay).length + a.filter(inDay).length;
    }
    // Fallback heuristique
    return dayList.filter((r) => {
      const s = norm(r.status);
      const unassigned = !r?.driver_id && !r?.driver?.id;
      return (s === "pending" || s === "accepted") && unassigned;
    }).length;
  }, [dayList, pendingReservations, assignedReservations, day]);

  const completedCount = useMemo(
    () => dayList.filter((r) => isCompleted(r.status)).length,
    [dayList]
  );

  const revenue = useMemo(
    () =>
      dayList.reduce((acc, r) => (isCompleted(r.status) ? acc + amountOf(r) : acc), 0),
    [dayList]
  );

  const availableDriver = useMemo(
    () => (Array.isArray(driver) ? driver : []).filter((d) => d?.is_active && d?.is_available).length,
    [driver]
  );

  // Petit logger pour vérifier la distribution réelle des statuts
  useEffect(() => {
    try {
      const dist = dayList.reduce((m, r) => {
        const s = norm(r.status);
        m[s] = (m[s] || 0) + 1;
        return m;
      }, {});
      // Commenter si ça t’embête dans la console
      console.debug("[OverviewCards] status distribution:", dist);
    } catch {}
  }, [dayList]);

  return (
    <div className={styles.overviewCards}>
      <div className={styles.card}>
        <h3>Réservations en attente</h3>
        <p>{waitingCount}</p>
      </div>
      <div className={styles.card}>
        <h3>Courses réalisées</h3>
        <p>{completedCount}</p>
      </div>
      <div className={styles.card}>
        <h3>Revenu généré</h3>
        <p>{revenue} CHF</p>
      </div>
      <div className={styles.card}>
        <h3>Chauffeurs disponibles</h3>
        <p>{availableDriver}</p>
      </div>
    </div>
  );
};

export default OverviewCards;
