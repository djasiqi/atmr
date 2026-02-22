import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FiSearch, FiX, FiCalendar, FiArrowDown, FiArrowUp, FiRefreshCw, FiClock, FiUserX, FiChevronDown } from 'react-icons/fi';
import { formatDelay } from '../../../../utils/formatDelay';
import styles from './ReservationFilters.module.css';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';

function ChipDropdown({ icon, value, options, onChange, activeWhen }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onClick = (e) => { if (ref.current && !ref.current.contains(e.target)) close(); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open, close]);

  const selected = options.find((o) => o.value === value);
  const isActive = activeWhen ? activeWhen(value) : value !== options[0]?.value;

  return (
    <div className={styles.chipDrop} ref={ref}>
      <button
        type="button"
        className={`${styles.chipBtn} ${isActive ? styles.chipBtnActive : ''}`}
        onClick={() => setOpen((p) => !p)}
      >
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
              className={`${styles.chipOption} ${o.value === value ? styles.chipOptionActive : ''}`}
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

const PERIOD_OPTIONS = [
  { value: 'all', label: 'Toutes' },
  { value: 'today', label: "Aujourd'hui" },
  { value: 'week', label: 'Cette semaine' },
  { value: 'single', label: 'Une date' },
  { value: 'range', label: 'Periode' },
];

const ReservationFilters = ({
  selectedDay,
  setSelectedDay,
  searchTerm,
  setSearchTerm,
  sortOrder,
  setSortOrder,
  searchInputRef,
  viewMode,
  setViewMode,
  alertFilter,
  onClearAlertFilter,
  onRefresh,
  totalResults,
  alerts,
  onFilterByAlert,
}) => {
  const [periodMode, setPeriodMode] = React.useState('all');
  const [startDate, setStartDate] = React.useState('');
  const [endDate, setEndDate] = React.useState('');

  React.useEffect(() => {
    if (selectedDay === 'all') {
      setPeriodMode('all');
      setStartDate('');
      setEndDate('');
    } else if (selectedDay && selectedDay.includes(':')) {
      const [s, e] = selectedDay.split(':');
      setPeriodMode('range');
      setStartDate(s || '');
      setEndDate(e || '');
    } else if (selectedDay) {
      setPeriodMode('single');
    }
  }, [selectedDay]);

  // Ctrl+K shortcut to focus search
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInputRef?.current?.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [searchInputRef]);

  const handlePeriodChange = (mode) => {
    setPeriodMode(mode);

    if (mode === 'all') {
      setSelectedDay('all');
      setStartDate('');
      setEndDate('');
    } else if (mode === 'today') {
      const today = new Date().toISOString().split('T')[0];
      setSelectedDay(today);
    } else if (mode === 'week') {
      const now = new Date();
      const dayOfWeek = now.getDay();
      const diffToMonday = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
      const monday = new Date(now);
      monday.setDate(now.getDate() - diffToMonday);
      const sunday = new Date(monday);
      sunday.setDate(monday.getDate() + 6);
      const fmt = (d) => d.toISOString().split('T')[0];
      setSelectedDay(`${fmt(monday)}:${fmt(sunday)}`);
    } else if (mode === 'single') {
      setStartDate('');
      setEndDate('');
    } else if (mode === 'range') {
      setSelectedDay('');
    }
  };

  React.useEffect(() => {
    if (periodMode === 'range' && startDate && endDate) {
      setSelectedDay(`${startDate}:${endDate}`);
    }
  }, [periodMode, startDate, endDate, setSelectedDay]);

  const isDateRange = selectedDay && selectedDay.includes(':');
  const chipLabel = alertFilter === 'delays' ? 'Retards' : alertFilter === 'unassigned' ? 'Sans chauffeur' : null;

  // Compute alert counts
  const delayAlerts = (alerts || []).filter((a) => a.type === 'delay');
  const unassignedAlert = (alerts || []).find((a) => a.type === 'unassigned');
  const delayCount = delayAlerts.length;
  const unassignedCount = unassignedAlert?.count || 0;
  const hasAlerts = delayCount > 0 || unassignedCount > 0;

  const maxDelayMinutes = delayCount > 0
    ? Math.max(...delayAlerts.map((a) => {
        const scheduled = new Date(a.reservation?.scheduled_time);
        const now = new Date();
        return Math.floor((now - scheduled) / (1000 * 60));
      }).filter((v) => v > 0))
    : 0;
  const maxDelayFormatted = formatDelay(maxDelayMinutes);

  return (
    <div className={styles.commandBar}>
      {/* Search */}
      <div className={styles.searchWrap}>
        <FiSearch className={styles.searchIcon} size={14} />
        <input
          type="text"
          placeholder="Rechercher..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className={styles.searchInput}
          ref={searchInputRef}
        />
        {searchTerm && (
          <button
            type="button"
            onClick={() => setSearchTerm('')}
            className={styles.clearBtn}
            title="Effacer"
          >
            <FiX size={12} />
          </button>
        )}
      </div>

      {/* Period */}
      <ChipDropdown
        icon={<FiCalendar size={12} />}
        value={periodMode}
        options={PERIOD_OPTIONS}
        onChange={handlePeriodChange}
        activeWhen={(v) => v !== 'all'}
      />

      {periodMode === 'single' && (
        <InlineDatePicker
          value={selectedDay === 'all' || (selectedDay && selectedDay.includes(':')) ? '' : selectedDay}
          onChange={(v) => setSelectedDay(v)}
          placeholder="Date"
        />
      )}

      {periodMode === 'range' && (
        <div className={styles.dateRange}>
          <InlineDatePicker
            value={startDate}
            onChange={(v) => setStartDate(v)}
            placeholder="Début"
          />
          <span className={styles.dateRangeSep}>—</span>
          <InlineDatePicker
            value={endDate}
            onChange={(v) => setEndDate(v)}
            placeholder="Fin"
          />
        </div>
      )}

      {/* Sort */}
      <ChipDropdown
        icon={sortOrder === 'desc' ? <FiArrowDown size={12} /> : <FiArrowUp size={12} />}
        value={sortOrder}
        options={[
          { value: 'desc', label: 'Plus récent' },
          { value: 'asc', label: 'Plus ancien' },
        ]}
        onChange={setSortOrder}
      />

      {/* View toggle */}
      <div className={styles.segmented}>
        <button
          type="button"
          className={`${styles.segBtn} ${viewMode === 'table' ? styles.segBtnActive : ''}`}
          onClick={() => setViewMode('table')}
        >
          Tableau
        </button>
        <button
          type="button"
          className={`${styles.segBtn} ${viewMode === 'map' ? styles.segBtnActive : ''} ${isDateRange ? styles.segBtnDisabled : ''}`}
          onClick={() => {
            if (!isDateRange) setViewMode('map');
          }}
          disabled={!!isDateRange}
          title={isDateRange ? "Carte disponible pour une seule journee" : 'Vue carte'}
        >
          Carte
        </button>
      </div>

      {/* Active alert filter chip */}
      {chipLabel && (
        <div className={styles.chipFilter}>
          <span className={styles.chipLabel}>{chipLabel}</span>
          <button
            type="button"
            className={styles.chipClose}
            onClick={onClearAlertFilter}
            title="Retirer le filtre"
          >
            <FiX size={12} />
          </button>
        </div>
      )}

      {/* Spacer */}
      <div className={styles.barSpacer} />

      {/* Alert indicators (inline in bar) */}
      {hasAlerts && !alertFilter && (
        <div className={styles.alertIndicators}>
          {delayCount > 0 && (
            <button
              type="button"
              className={styles.alertDot}
              onClick={() => onFilterByAlert?.('delays')}
              title={`${delayCount} retard${delayCount > 1 ? 's' : ''}${maxDelayFormatted ? ` (max ${maxDelayFormatted})` : ''}`}
            >
              <FiClock size={11} />
              <span>{delayCount}</span>
            </button>
          )}
          {unassignedCount > 0 && (
            <button
              type="button"
              className={styles.alertDot}
              onClick={() => onFilterByAlert?.('unassigned')}
              title={`${unassignedCount} sans chauffeur`}
            >
              <FiUserX size={11} />
              <span>{unassignedCount}</span>
            </button>
          )}
        </div>
      )}

      {/* Result count + Refresh */}
      <div className={styles.barMeta}>
        {totalResults !== undefined && (
          <span className={styles.barResultCount}>
            {totalResults} resultat{totalResults !== 1 ? 's' : ''}
          </span>
        )}
        {onRefresh && (
          <button
            type="button"
            className={styles.refreshBtn}
            onClick={onRefresh}
            title="Rafraichir"
          >
            <FiRefreshCw size={14} />
          </button>
        )}
      </div>
    </div>
  );
};

export default ReservationFilters;
