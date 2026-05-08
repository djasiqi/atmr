import React, { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import {
  FiArrowUp,
  FiArrowDown,
  FiPackage,
  FiCpu,
  FiSearch,
  FiX,
  FiChevronDown,
  FiRefreshCw,
  FiAlertTriangle,
  FiClock,
} from 'react-icons/fi';
import DispatchTable from '../../Dashboard/components/DispatchTable';
import DispatchTableSkeleton from '../../../../components/SkeletonLoaders/DispatchTableSkeleton';
import EmptyState from '../../../../components/EmptyState';
import { getDispatchRowDelayInfo } from '../../../../utils/dispatchDelayMapKey';

function ChipDropdown({ icon, value, options, onChange, styles }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onClickOut = (e) => { if (ref.current && !ref.current.contains(e.target)) close(); };
    const onEsc = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('mousedown', onClickOut);
    document.addEventListener('keydown', onEsc);
    return () => { document.removeEventListener('mousedown', onClickOut); document.removeEventListener('keydown', onEsc); };
  }, [open, close]);

  const selected = options.find((o) => String(o.value) === String(value));

  return (
    <div className={styles.chipDrop} ref={ref}>
      <button type="button" className={styles.chipBtn} onClick={() => setOpen((p) => !p)}>
        {icon}
        <span className={styles.chipText}>{selected?.label || '—'}</span>
        <FiChevronDown size={11} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && (
        <div className={styles.chipMenu}>
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${styles.chipOption} ${String(o.value) === String(value) ? styles.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value); close(); }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// V11: Cle unifiee
const getDispatchKey = (d) => d.booking_id ?? d.id;

// V2: Statuts actifs pour calcul retards coherent
const ACTIVE_STATUSES = ['accepted', 'assigned', 'en_route', 'in_progress'];

// V17: Hierarchie visuelle pour tri par priorite metier
const getRowPriority = (d, delayMap) => {
  const delay = getDispatchRowDelayInfo(delayMap, d)?.minutes ?? delayMap?.[getDispatchKey(d)]?.minutes ?? 0;
  const isUnassigned = !d.driver_id && !d.driver;
  if (delay > 15) return 'critical';
  if (delay > 5) return 'moderate';
  if (delay > 0) return 'light';
  if (isUnassigned) return 'unassigned';
  return 'normal';
};

const PRIORITY_ORDER = { critical: 0, moderate: 1, light: 2, unassigned: 3, normal: 4 };

const ManualModePanel = ({
  dispatches = [],
  delays = [],
  loading,
  error,
  sortBy,
  setSortBy,
  sortOrder,
  setSortOrder,
  selectedReservationForAssignment: _selectedReservationForAssignment,
  setSelectedReservationForAssignment,
  onSchedule,
  onDispatchNow,
  onDelete,
  currentDate: _currentDate,
  drivers = [],
  styles = {},
  currentCompanyId,
  // Nouvelles props
  onAssignDirect,
  onRowClick,
  onGoToSemiAuto,
  delayMap = {},
  onRefresh,
}) => {
  // Zone B: Etat filtres
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');

  const kpis = useMemo(() => {
    const unassigned = dispatches.filter((d) => !d.driver_id && !d.driver).length;
    const delayedActive = dispatches.filter(
      (d) =>
        ACTIVE_STATUSES.includes(d.status) &&
        (getDispatchRowDelayInfo(delayMap, d)?.minutes || delayMap[getDispatchKey(d)]?.minutes || 0) >
          0
    ).length;
    const inProgress = dispatches.filter(
      (d) => d.status === 'en_route' || d.status === 'in_progress'
    ).length;

    return { unassigned, delayedActive, inProgress };
  }, [dispatches, delayMap]);

  // Zone B: Filtrage local
  const filteredDispatches = useMemo(() => {
    let result = [...dispatches];

    // Filtre par pill
    if (activeFilter === 'unassigned') {
      result = result.filter((d) => !d.driver_id && !d.driver);
    } else if (activeFilter === 'delayed') {
      // V15: Meme regle que KPI - actifs + delay > 0
      result = result.filter(
        (d) =>
          ACTIVE_STATUSES.includes(d.status) &&
          (getDispatchRowDelayInfo(delayMap, d)?.minutes || delayMap[getDispatchKey(d)]?.minutes || 0) >
            0
      );
    }

    // Filtre par recherche texte
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (d) =>
          (d.client_name || d.client?.full_name || '').toLowerCase().includes(q) ||
          (d.pickup_location || '').toLowerCase().includes(q) ||
          (d.dropoff_location || '').toLowerCase().includes(q)
      );
    }

    return result;
  }, [dispatches, activeFilter, searchQuery, delayMap]);

  // Tri
  const sortedDispatches = useMemo(() => {
    return [...filteredDispatches].sort((a, b) => {
      let aValue, bValue;

      switch (sortBy) {
        case 'priority': {
          const pa = PRIORITY_ORDER[getRowPriority(a, delayMap)] ?? 4;
          const pb = PRIORITY_ORDER[getRowPriority(b, delayMap)] ?? 4;
          if (pa !== pb) return pa - pb;
          return new Date(a.scheduled_time || 0) - new Date(b.scheduled_time || 0);
        }
        case 'time':
          aValue = new Date(a.scheduled_time || 0);
          bValue = new Date(b.scheduled_time || 0);
          break;
        case 'client':
          aValue = (a.client_name || a.client?.full_name || '').toLowerCase();
          bValue = (b.client_name || b.client?.full_name || '').toLowerCase();
          break;
        case 'status':
          aValue = a.status || '';
          bValue = b.status || '';
          break;
        default:
          return 0;
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
  }, [filteredDispatches, sortBy, sortOrder, delayMap]);

  // Etat clean : aucune action requise
  const isCleanState = kpis.unassigned === 0 && kpis.delayedActive === 0;

  if (loading) {
    return <DispatchTableSkeleton rows={8} />;
  }

  if (error) {
    return <div className={styles.error}>Erreur: {error}</div>;
  }

  return (
    <>
      <div className={styles.demoSummary} data-tour-id="dispatch-demo-summary">
        <strong>Aujourd'hui :</strong>{' '}
        {sortedDispatches.length} transport{sortedDispatches.length > 1 ? 's' : ''} planifie
        {sortedDispatches.length > 1 ? 's' : ''} · {kpis.unassigned} a assigner · {kpis.inProgress}{' '}
        en cours · {drivers.length} chauffeur{drivers.length > 1 ? 's' : ''} disponible
        {drivers.length > 1 ? 's' : ''}
      </div>
      {/* Zone B: Command Bar */}
      <div className={styles.commandBar} data-tour-id="dispatch-command-bar">
        <div className={styles.searchWrap}>
          <FiSearch size={14} className={styles.searchIcon} />
          <input
            type="text"
            placeholder="Rechercher client, adresse..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className={styles.searchInput}
            data-tour-id="dispatch-search-input"
          />
          {searchQuery && (
            <button className={styles.clearBtn} onClick={() => setSearchQuery('')} type="button">
              <FiX size={11} />
            </button>
          )}
        </div>

        <ChipDropdown
          icon={sortOrder === 'asc' ? <FiArrowUp size={12} /> : <FiArrowDown size={12} />}
          value={sortBy}
          options={[
            { value: 'priority', label: 'Priorité' },
            { value: 'time', label: 'Heure' },
            { value: 'client', label: 'Client' },
            { value: 'status', label: 'Statut' },
          ]}
          onChange={setSortBy}
          styles={styles}
        />

        <button
          onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
          className={styles.sortOrderBtn}
          title={sortOrder === 'asc' ? 'Tri croissant' : 'Tri décroissant'}
        >
          {sortOrder === 'asc' ? <FiArrowUp size={14} /> : <FiArrowDown size={14} />}
        </button>

        <div className={styles.segmented} data-tour-id="dispatch-filters">
          <button
            type="button"
            className={`${styles.segBtn} ${activeFilter === 'all' ? styles.segBtnActive : ''}`}
            onClick={() => setActiveFilter('all')}
          >
            Tous
          </button>
          <button
            type="button"
            className={`${styles.segBtn} ${activeFilter === 'unassigned' ? styles.segBtnActive : ''}`}
            onClick={() => setActiveFilter('unassigned')}
          >
            A assigner
            {kpis.unassigned > 0 && <span className={styles.segCount}>{kpis.unassigned}</span>}
          </button>
          <button
            type="button"
            className={`${styles.segBtn} ${activeFilter === 'delayed' ? styles.segBtnActive : ''}`}
            onClick={() => setActiveFilter('delayed')}
          >
            Retards
            {kpis.delayedActive > 0 && <span className={styles.segCount}>{kpis.delayedActive}</span>}
          </button>
        </div>

        <div className={styles.barSpacer} />

        {/* Alert indicators */}
        <div className={styles.alertIndicators}>
          {kpis.unassigned > 0 && (
            <button
              type="button"
              className={`${styles.alertDot} ${activeFilter === 'unassigned' ? styles.alertDotActive : ''}`}
              onClick={() => setActiveFilter('unassigned')}
              title={`${kpis.unassigned} à assigner`}
            >
              <FiAlertTriangle size={11} />
              <span>{kpis.unassigned}</span>
            </button>
          )}
          {kpis.delayedActive > 0 && (
            <button
              type="button"
              className={`${styles.alertDot} ${styles.alertDotDelay} ${activeFilter === 'delayed' ? styles.alertDotActive : ''}`}
              onClick={() => setActiveFilter('delayed')}
              title={`${kpis.delayedActive} retard${kpis.delayedActive > 1 ? 's' : ''}`}
            >
              <FiClock size={11} />
              <span>{kpis.delayedActive}</span>
            </button>
          )}
        </div>

        <div className={styles.barMeta}>
          <span className={styles.barResultCount}>
            {sortedDispatches.length} course{sortedDispatches.length !== 1 ? 's' : ''}
          </span>
          <button type="button" className={styles.refreshBtn} title="Rafraîchir" onClick={onRefresh}>
            <FiRefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Zone C: KPIs inline */}
      <div className={styles.kpisRow} data-tour-id="dispatch-kpis-row">
        <span className={`${styles.kpiItem} ${kpis.unassigned > 0 ? styles.kpiWarning : ''}`}>
          A assigner : <strong>{kpis.unassigned}</strong>
        </span>
        <span className={`${styles.kpiItem} ${kpis.delayedActive > 0 ? styles.kpiDanger : ''}`}>
          Retards : <strong>{kpis.delayedActive}</strong>
        </span>
        <span className={styles.kpiItem}>
          En cours : <strong>{kpis.inProgress}</strong>
        </span>
        <span className={styles.kpiItem}>
          Chauffeurs : <strong>{drivers.length}</strong>
        </span>
      </div>

      {/* Mini barre de priorite / etat clean */}
      {isCleanState ? (
        <div className={styles.cleanState}>
          Toutes les courses sont assignees et dans les temps.
        </div>
      ) : (
        <div className={styles.priorityBar}>
          {kpis.unassigned > 0 && (
            <button
              className={styles.prioritySegment}
              onClick={() => setActiveFilter('unassigned')}
            >
              {kpis.unassigned} a assigner
            </button>
          )}
          {kpis.delayedActive > 0 && (
            <button
              className={`${styles.prioritySegment} ${styles.prioritySegmentDanger}`}
              onClick={() => setActiveFilter('delayed')}
            >
              {kpis.delayedActive} en retard
            </button>
          )}
          {kpis.inProgress > 0 && (
            <span className={styles.prioritySegment}>
              {kpis.inProgress} en cours
            </span>
          )}
        </div>
      )}

      {/* Mini-alerte conditionnelle retards */}
      {kpis.delayedActive > 0 && activeFilter !== 'delayed' && (
        <div className={styles.alertInline}>
          {kpis.delayedActive} course{kpis.delayedActive > 1 ? 's' : ''} en retard
          <button
            className={styles.alertInlineBtn}
            onClick={() => setActiveFilter('delayed')}
          >
            Voir
          </button>
        </div>
      )}

      {/* Zone D: Table */}
      {dispatches.length === 0 ? (
        <EmptyState
          icon={<FiPackage size={40} />}
          title="Aucune course pour cette date"
          message="Creez de nouvelles reservations pour commencer l'assignation manuelle."
        />
      ) : sortedDispatches.length === 0 ? (
        <EmptyState
          icon={<FiSearch size={40} />}
          title="Aucun resultat"
          message="Modifiez vos filtres ou votre recherche."
        />
      ) : (
        <DispatchTable
          dispatches={sortedDispatches}
          delays={delays}
          delayMap={delayMap}
          onRowClick={onRowClick}
          onAssign={
            setSelectedReservationForAssignment
              ? (reservationId) => {
                  if (typeof setSelectedReservationForAssignment === 'function') {
                    setSelectedReservationForAssignment(reservationId);
                  }
                }
              : undefined
          }
          onAssignDirect={onAssignDirect}
          onSchedule={onSchedule}
          onDispatchNow={onDispatchNow}
          onDelete={onDelete}
          hideEdit={true}
          hideDelete={true}
          currentCompanyId={currentCompanyId}
          activeDrivers={drivers}
        />
      )}

      {/* Hint MDI compact */}
      {onGoToSemiAuto && (
        <div className={styles.mdiHint}>
          <FiCpu size={12} />
          <span>
            Suggestions IA disponibles en mode Semi-Automatique.{' '}
            <button className={styles.mdiHintLink} onClick={onGoToSemiAuto}>
              Passer en Semi-Auto
            </button>
          </span>
        </div>
      )}
    </>
  );
};

export default ManualModePanel;
