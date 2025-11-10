import React, { useEffect, useMemo, useState } from 'react';
import styles from './CompanyPlanning.module.css';
import {
  fetchShifts,
  createShift,
  updateShift,
  deleteShift,
} from '../../../services/driverPlanningService';

import Scheduler from './components/Scheduler';
import KPIs from './components/KPIs';
import Filters from './components/Filters';
import DriverLegend from './components/DriverLegend';
import { fetchCompanyDriver } from '../../../services/driverPlanningService';
import ShiftModal from './components/ShiftModal';
import UnavailabilityModal from './components/UnavailabilityModal';
import BreakModal from './components/BreakModal';
import TemplateEditor from './components/TemplateEditor';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';

const todayIso = () => new Date().toISOString().slice(0, 10);

function startOfWeek(d) {
  const dt = new Date(d);
  const day = dt.getDay();
  const diff = (day === 0 ? -6 : 1) - day; // Monday as start
  dt.setDate(dt.getDate() + diff);
  dt.setHours(0, 0, 0, 0);
  return dt;
}

function formatRange(range, view) {
  const from = new Date(range.from);
  const to = new Date(range.to);
  const fmt = (d) =>
    d.toLocaleDateString('fr-CH', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
    });
  if (view === 'day') return fmt(from);
  return `${fmt(from)} – ${fmt(to)}`;
}

function computeRange(anchor = new Date(), view = 'week') {
  const base = new Date(anchor);
  base.setHours(0, 0, 0, 0);
  if (view === 'day') {
    const from = new Date(base);
    const to = new Date(base);
    to.setHours(23, 59, 59, 999);
    return { from: from.toISOString(), to: to.toISOString() };
  }
  if (view === 'month') {
    const from = new Date(base.getFullYear(), base.getMonth(), 1, 0, 0, 0, 0);
    const to = new Date(base.getFullYear(), base.getMonth() + 1, 0, 23, 59, 59, 999);
    return { from: from.toISOString(), to: to.toISOString() };
  }
  // week
  const wStart = startOfWeek(base);
  const wEnd = new Date(wStart);
  wEnd.setDate(wStart.getDate() + 6);
  wEnd.setHours(23, 59, 59, 999);
  return { from: wStart.toISOString(), to: wEnd.toISOString() };
}

function shiftRangeBy(range, view, dir = 1) {
  const delta = view === 'day' ? 1 : view === 'month' ? 30 : 7;
  const anchor = new Date(range.from);
  anchor.setDate(anchor.getDate() + dir * delta);
  return computeRange(anchor, view);
}

function HolidayBar({ range }) {
  // Swiss public holidays (minimal demo; ideally load per canton)
  const holidays = [];
  const y = new Date(range.from).getFullYear();
  // New Year
  holidays.push({ date: `${y}-01-01`, name: 'Nouvel an' });
  // Swiss National Day
  holidays.push({ date: `${y}-08-01`, name: 'Fête nationale' });
  // Christmas
  holidays.push({ date: `${y}-12-25`, name: 'Noël' });
  const inRange = (iso) => {
    const t = new Date(iso).getTime();
    return t >= new Date(range.from).getTime() && t <= new Date(range.to).getTime();
  };
  const visible = holidays.filter((h) => inRange(`${h.date}T12:00:00`));
  if (visible.length === 0) return null;
  return (
    <div className={styles.holidayBar}>
      {visible.map((h) => (
        <span key={h.date} className={styles.holidayPill}>
          {new Date(`${h.date}T00:00:00`).toLocaleDateString('fr-CH', {
            day: '2-digit',
            month: 'short',
          })}{' '}
          — {h.name}
        </span>
      ))}
    </div>
  );
}

export default function CompanyPlanning() {
  const [range, setRange] = useState({
    from: `${todayIso()}T00:00:00`,
    to: `${todayIso()}T23:59:59`,
  });
  const [view, setView] = useState('week'); // day | week | month
  const [driverId, setDriverId] = useState(null);
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [items, setItems] = useState([]);
  const [filters, setFilters] = useState({});
  const [modals, setModals] = useState({
    shift: false,
    unav: false,
    brk: false,
    tpl: false,
  });

  const params = useMemo(() => ({ from: range.from, to: range.to, driverId }), [range, driverId]);

  useEffect(() => {
    fetchCompanyDriver()
      .then((list) => {
        setDrivers(Array.isArray(list) ? list : []);
      })
      .catch(() => setDrivers([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchShifts(params);
        if (!cancelled) setItems(Array.isArray(data?.items) ? data.items : []);
      } catch (e) {
        if (!cancelled) setError('Erreur de chargement des shifts');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [params]);

  function goToday(nextView = view) {
    const r = computeRange(new Date(), nextView);
    setRange(r);
    setView(nextView);
  }

  function shiftRange(dir, nextView = view) {
    const r = shiftRangeBy(range, nextView, dir);
    setRange(r);
  }

  const onCreate = async (payload) => {
    const created = await createShift(null, payload);
    setItems((prev) => [created, ...prev]);
  };

  const onUpdate = async (id, payload) => {
    const updated = await updateShift(null, id, payload);
    setItems((prev) => prev.map((s) => (s.id === id ? updated : s)));
  };

  const onDelete = async (id) => {
    await deleteShift(null, id);
    setItems((prev) => prev.filter((s) => s.id !== id));
  };

  return (
    <div className={styles.pageContainer}>
      <CompanyHeader />
      <div className={styles.layoutWrapper}>
        <CompanySidebar />
        <div className={styles.mainContent}>
          <div className={styles.container}>
            <div className={styles.toolbar}>
              <div className={styles.left}>
                <div className={styles.tabs}>
                  <button
                    className={`${styles.tab} ${view === 'day' ? styles.tabActive : ''}`}
                    onClick={() => setView('day')}
                  >
                    Jour
                  </button>
                  <button
                    className={`${styles.tab} ${view === 'week' ? styles.tabActive : ''}`}
                    onClick={() => setView('week')}
                  >
                    Semaine
                  </button>
                  <button
                    className={`${styles.tab} ${view === 'month' ? styles.tabActive : ''}`}
                    onClick={() => setView('month')}
                  >
                    Mois
                  </button>
                </div>
                <div className={styles.nav}>
                  <button className={styles.navBtn} onClick={() => shiftRange(-1, view)}>
                    ◀
                  </button>
                  <div className={styles.dateLabel}>{formatRange(range, view)}</div>
                  <button className={styles.navBtn} onClick={() => shiftRange(1, view)}>
                    ▶
                  </button>
                  <button className={styles.navBtn} onClick={() => goToday(view)}>
                    Aujourd’hui
                  </button>
                </div>
                <input
                  type="date"
                  value={range.from.slice(0, 10)}
                  onChange={(e) =>
                    setRange({
                      from: `${e.target.value}T00:00:00`,
                      to: `${e.target.value}T23:59:59`,
                    })
                  }
                />
                <div style={{ marginLeft: 8 }}>
                  <Filters value={filters} onChange={setFilters} />
                </div>
              </div>
              <div className={styles.right}>
                <div style={{ marginRight: 8 }}>
                  <KPIs items={items} />
                </div>
                {loading && <span className={styles.badge}>Chargement…</span>}
                {error && <span className={styles.error}>{error}</span>}
                <div style={{ marginLeft: 8, display: 'flex', gap: 6 }}>
                  <button onClick={() => setModals({ ...modals, shift: true })}>+ Shift</button>
                  <button onClick={() => setModals({ ...modals, unav: true })}>+ Indispo</button>
                  <button onClick={() => setModals({ ...modals, brk: true })}>+ Pause</button>
                  <button onClick={() => setModals({ ...modals, tpl: true })}>Modèle</button>
                </div>
              </div>
            </div>
            <HolidayBar range={range} />
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '260px 1fr',
                gap: 12,
              }}
            >
              <div>
                <div style={{ marginBottom: 8 }}>
                  <label style={{ fontSize: 12, color: '#6b7280' }}>Chauffeur</label>
                  <select
                    value={driverId || ''}
                    onChange={(e) => setDriverId(e.target.value ? Number(e.target.value) : null)}
                    style={{ width: '100%' }}
                  >
                    <option value="">Tous</option>
                    {drivers.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.full_name || d.username}
                      </option>
                    ))}
                  </select>
                </div>
                <DriverLegend drivers={drivers} />
              </div>
              <Scheduler
                items={items}
                onCreate={onCreate}
                onUpdate={onUpdate}
                onDelete={onDelete}
              />
            </div>
            <ShiftModal
              open={modals.shift}
              onClose={() => setModals({ ...modals, shift: false })}
              onSave={(payload) => {
                setModals({ ...modals, shift: false });
                onCreate(payload);
              }}
            />
            <UnavailabilityModal
              open={modals.unav}
              onClose={() => setModals({ ...modals, unav: false })}
              onSave={() => setModals({ ...modals, unav: false })}
            />
            <BreakModal
              open={modals.brk}
              onClose={() => setModals({ ...modals, brk: false })}
              onSave={() => setModals({ ...modals, brk: false })}
            />
            <TemplateEditor
              open={modals.tpl}
              onClose={() => setModals({ ...modals, tpl: false })}
              onSave={() => setModals({ ...modals, tpl: false })}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
