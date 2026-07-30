// src/pages/company/Dashboard/components/ReservationFilterBar.jsx
import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { FiSearch, FiAlertTriangle, FiClock, FiX, FiRefreshCw, FiChevronDown, FiUser, FiUsers } from 'react-icons/fi';
import styles from './ReservationFilterBar.module.css';

export const TIME_RANGES = [
  { id: 'all', label: 'Toute la journée', start: null, end: null },
  { id: 'morning', label: '06h – 12h', start: 6, end: 12 },
  { id: 'afternoon', label: '12h – 18h', start: 12, end: 18 },
  { id: 'evening', label: '18h – 00h', start: 18, end: 24 },
];

function ChipDropdown({ icon, label, value, options, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onClickOutside = (e) => { if (ref.current && !ref.current.contains(e.target)) close(); };
    const onEsc = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('mousedown', onClickOutside);
    document.addEventListener('keydown', onEsc);
    return () => { document.removeEventListener('mousedown', onClickOutside); document.removeEventListener('keydown', onEsc); };
  }, [open, close]);

  const selected = options.find((o) => o.value === value);
  const isDefault = !value || value === options[0]?.value;

  return (
    <div className={styles.chipDrop} ref={ref}>
      <button
        type="button"
        className={`${styles.chipBtn} ${!isDefault ? styles.chipBtnActive : ''}`}
        onClick={() => setOpen((p) => !p)}
      >
        {icon}
        <span className={styles.chipText}>{selected?.label || label}</span>
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

const ReservationFilterBar = ({
  searchQuery,
  onSearchChange,
  urgenceMode,
  onToggleUrgence,
  delaysOnly,
  onToggleDelaysOnly,
  visibleCount,
  totalCount,
  activeDelayCount,
  drivers,
  selectedDriver,
  onDriverChange,
  institutions,
  selectedInstitution,
  onInstitutionChange,
  timeRange,
  onTimeRangeChange,
  onRefresh,
}) => {
  const activeDrivers = useMemo(
    () => (drivers || []).filter((d) => d.is_active),
    [drivers]
  );

  return (
    <div className={`${styles.commandBar} ${urgenceMode ? styles.commandBarUrgence : ''}`}>
      {/* Search */}
      <div className={styles.searchWrap}>
        <FiSearch size={14} className={styles.searchIcon} />
        <input
          type="text"
          className={styles.searchInput}
          placeholder="Rechercher..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
        {searchQuery && (
          <button className={styles.clearBtn} onClick={() => onSearchChange('')} type="button">
            <FiX size={11} />
          </button>
        )}
      </div>

      {/* Chips */}
      <ChipDropdown
        icon={<FiUser size={12} />}
        label="Chauffeur"
        value={selectedDriver || ''}
        options={[
          { value: '', label: 'Chauffeur' },
          ...activeDrivers.map((d) => ({ value: String(d.id), label: d.full_name || d.username })),
        ]}
        onChange={(v) => onDriverChange(v || null)}
      />

      {institutions && institutions.length > 0 && (
        <ChipDropdown
          icon={<FiUsers size={12} />}
          label="Institution"
          value={selectedInstitution || ''}
          options={[
            { value: '', label: 'Institution' },
            ...institutions.map((inst) => ({ value: inst, label: inst })),
          ]}
          onChange={(v) => onInstitutionChange(v || null)}
        />
      )}

      <ChipDropdown
        icon={<FiClock size={12} />}
        label="Horaire"
        value={timeRange || 'all'}
        options={TIME_RANGES.map((tr) => ({ value: tr.id, label: tr.label }))}
        onChange={onTimeRangeChange}
      />

      <div className={styles.barSpacer} />

      {/* Alert indicators */}
      <div className={styles.alertIndicators}>
        <button
          type="button"
          className={`${styles.alertDot} ${styles.alertDotTouch} ${urgenceMode ? styles.alertDotActive : ''}`}
          onClick={onToggleUrgence}
          aria-pressed={urgenceMode}
          aria-label={urgenceMode ? 'Désactiver le filtre Urgences' : 'Afficher les Urgences'}
          title="Urgences"
        >
          <FiAlertTriangle size={11} aria-hidden />
          <span>Urgences</span>
        </button>

        {activeDelayCount > 0 && (
          <button
            type="button"
            className={`${styles.alertDot} ${styles.alertDotTouch} ${styles.alertDotDelay} ${delaysOnly ? styles.alertDotActive : ''}`}
            onClick={onToggleDelaysOnly}
            aria-pressed={delaysOnly}
            aria-label={
              delaysOnly
                ? `Désactiver le filtre retards (${activeDelayCount})`
                : `Filtrer les ${activeDelayCount} retard${activeDelayCount > 1 ? 's' : ''}`
            }
            title={`${activeDelayCount} retard${activeDelayCount > 1 ? 's' : ''}`}
          >
            <FiClock size={11} aria-hidden />
            <span>{activeDelayCount}</span>
          </button>
        )}
      </div>

      {/* Meta */}
      <div className={styles.barMeta}>
        <span className={styles.barResultCount}>
          {visibleCount !== totalCount
            ? `${visibleCount} / ${totalCount}`
            : `${totalCount} résultat${totalCount !== 1 ? 's' : ''}`}
        </span>
        <button type="button" className={styles.refreshBtn} title="Rafraîchir" aria-label="Rafraîchir la liste" onClick={onRefresh}>
          <FiRefreshCw size={14} />
        </button>
      </div>
    </div>
  );
};

export default ReservationFilterBar;
