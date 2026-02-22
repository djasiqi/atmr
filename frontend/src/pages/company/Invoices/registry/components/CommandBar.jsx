import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { FiSearch, FiX, FiChevronDown, FiFilter, FiCalendar, FiSliders } from 'react-icons/fi';
import styles from './CommandBar.module.css';
import ntStyles from '../../../Settings/tabs/NotificationsTab.module.css';
import { fetchCompanyClients } from '../../../../../services/companyService';

function ChipDrop({ options, value, onChange, minWidth }) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef(null);
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 });

  useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const reposition = useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left, width: Math.max(r.width, minWidth || 120) });
  }, [minWidth]);

  useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const selected = options.find((o) => String(o.value) === String(value));

  return (
    <div className={styles.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${styles.chipBtn} ${value ? styles.chipBtnActive : ''}`}
        onClick={() => setOpen((p) => !p)}
      >
        <span className={styles.chipText}>{selected?.label || 'Sélectionner'}</span>
        <FiChevronDown size={10} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className={styles.chipMenu}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width, zIndex: 10000 }}
        >
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${styles.chipOption} ${String(o.value) === String(value) ? styles.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              {o.label}
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

const MONTH_LABELS = {
  1: 'Janvier',
  2: 'Fevrier',
  3: 'Mars',
  4: 'Avril',
  5: 'Mai',
  6: 'Juin',
  7: 'Juillet',
  8: 'Aout',
  9: 'Septembre',
  10: 'Octobre',
  11: 'Novembre',
  12: 'Decembre',
};

const STATUS_OPTIONS = [
  { value: '', label: 'Tous les statuts' },
  { value: 'draft', label: 'Brouillon' },
  { value: 'sent', label: 'Envoyee' },
  { value: 'partially_paid', label: 'Part. payee' },
  { value: 'paid', label: 'Payee' },
  { value: 'overdue', label: 'En retard' },
  { value: 'cancelled', label: 'Annulee' },
];

const STATUS_LABELS = STATUS_OPTIONS.reduce((acc, opt) => {
  if (opt.value) acc[opt.value] = opt.label;
  return acc;
}, {});

const CommandBar = ({ filters, defaultFilters, onFilterChange, companyId, searchInputRef }) => {
  const [clients, setClients] = useState([]);
  const [loadingClients, setLoadingClients] = useState(false);
  const [clientSearchOpen, setClientSearchOpen] = useState(false);
  const [clientSearchQuery, setClientSearchQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(-1);
  const clientDropdownRef = useRef(null);
  const clientInputRef = useRef(null);
  const listboxRef = useRef(null);

  useEffect(() => {
    const loadClients = async () => {
      if (!companyId) return;
      try {
        setLoadingClients(true);
        const clientsData = await fetchCompanyClients();
        setClients(clientsData);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Erreur chargement clients:', err);
      } finally {
        setLoadingClients(false);
      }
    };
    loadClients();
  }, [companyId]);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (clientDropdownRef.current && !clientDropdownRef.current.contains(e.target)) {
        setClientSearchOpen(false);
      }
    };
    if (clientSearchOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [clientSearchOpen]);

  const getClientDisplayName = useCallback((client) => {
    return (
      client.institution_name ||
      `${client.first_name || ''} ${client.last_name || ''}`.trim() ||
      client.username ||
      `Client ${client.id}`
    );
  }, []);

  const filteredClients = clients.filter((c) => {
    if (!clientSearchQuery) return true;
    const name = getClientDisplayName(c).toLowerCase();
    return name.includes(clientSearchQuery.toLowerCase());
  });

  const selectedClientName = filters.client_id
    ? getClientDisplayName(clients.find((c) => String(c.id) === String(filters.client_id)) || {})
    : '';

  const handleSelectClient = (clientId) => {
    onFilterChange({ client_id: clientId ? String(clientId) : '' });
    setClientSearchOpen(false);
    setClientSearchQuery('');
    setActiveIndex(-1);
  };

  const handleClientKeyDown = (e) => {
    if (!clientSearchOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        e.preventDefault();
        setClientSearchOpen(true);
      }
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((prev) => Math.min(prev + 1, filteredClients.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((prev) => Math.max(prev - 1, -1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex === -1 || activeIndex === 0) {
        handleSelectClient('');
      } else {
        const client = filteredClients[activeIndex - 1];
        if (client) handleSelectClient(client.id);
      }
    } else if (e.key === 'Escape') {
      setClientSearchOpen(false);
      setActiveIndex(-1);
    }
  };

  useEffect(() => {
    if (activeIndex >= 0 && listboxRef.current) {
      const items = listboxRef.current.querySelectorAll('[role="option"]');
      if (items[activeIndex]) {
        items[activeIndex].scrollIntoView({ block: 'nearest' });
      }
    }
  }, [activeIndex]);

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 5 }, (_, i) => currentYear - i);
  const months = [
    { value: '', label: 'Tous les mois' },
    ...Array.from({ length: 12 }, (_, i) => ({
      value: i + 1,
      label: MONTH_LABELS[i + 1],
    })),
  ];

  const hasActiveFilters = () => {
    return (
      filters.status !== defaultFilters.status ||
      filters.client_id !== defaultFilters.client_id ||
      filters.year !== defaultFilters.year ||
      filters.month !== defaultFilters.month ||
      (filters.q || '') !== (defaultFilters.q || '') ||
      filters.with_balance !== defaultFilters.with_balance ||
      filters.with_reminders !== defaultFilters.with_reminders
    );
  };

  const handleReset = () => {
    onFilterChange({
      status: defaultFilters.status,
      client_id: defaultFilters.client_id,
      year: defaultFilters.year,
      month: defaultFilters.month,
      q: defaultFilters.q,
      with_balance: defaultFilters.with_balance,
      with_reminders: defaultFilters.with_reminders,
      page: 1,
      per_page: defaultFilters.per_page,
    });
  };

  const activeChips = [];
  if (filters.status && filters.status !== defaultFilters.status) {
    activeChips.push({
      key: 'status',
      label: `Statut: ${STATUS_LABELS[filters.status] || filters.status}`,
      onRemove: () => onFilterChange({ status: '' }),
    });
  }
  if (filters.client_id && filters.client_id !== defaultFilters.client_id) {
    activeChips.push({
      key: 'client_id',
      label: `Client: ${selectedClientName || filters.client_id}`,
      onRemove: () => onFilterChange({ client_id: '' }),
    });
  }
  if (String(filters.year) !== String(defaultFilters.year)) {
    activeChips.push({
      key: 'year',
      label: `Annee: ${filters.year}`,
      onRemove: () => onFilterChange({ year: defaultFilters.year }),
    });
  }
  if (filters.month && filters.month !== defaultFilters.month) {
    activeChips.push({
      key: 'month',
      label: `Mois: ${MONTH_LABELS[filters.month] || filters.month}`,
      onRemove: () => onFilterChange({ month: '' }),
    });
  }
  if (filters.q && filters.q !== (defaultFilters.q || '')) {
    activeChips.push({
      key: 'q',
      label: `Recherche: "${filters.q}"`,
      onRemove: () => onFilterChange({ q: '' }),
    });
  }
  if (filters.with_balance && !defaultFilters.with_balance) {
    activeChips.push({
      key: 'with_balance',
      label: 'Solde > 0',
      onRemove: () => onFilterChange({ with_balance: false }),
    });
  }
  if (filters.with_reminders && !defaultFilters.with_reminders) {
    activeChips.push({
      key: 'with_reminders',
      label: 'Avec rappels',
      onRemove: () => onFilterChange({ with_reminders: false }),
    });
  }

  return (
    <div className={styles.commandBar}>
      {/* Row 1 — Search */}
      <div className={styles.searchRow}>
        <div className={styles.searchWrapper}>
          <FiSearch size={14} className={styles.searchIcon} />
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Rechercher par numero, client, email..."
            value={filters.q || ''}
            onChange={(e) => onFilterChange({ q: e.target.value })}
            ref={searchInputRef}
          />
          {filters.q && (
            <button
              className={styles.searchClear}
              onClick={() => onFilterChange({ q: '' })}
              aria-label="Effacer la recherche"
            >
              <FiX size={12} />
            </button>
          )}
        </div>
        {hasActiveFilters() && (
          <button className={styles.resetLink} onClick={handleReset}>
            Reinitialiser les filtres
          </button>
        )}
      </div>

      {/* Row 2 — Filters organized in groups */}
      <div className={styles.filtersRow}>
        {/* Group: Filtres principaux */}
        <div className={styles.filterGroup}>
          <span className={styles.filterGroupLabel}>
            <FiFilter size={10} />
            Filtres
          </span>
          <div className={styles.filterGroupControls}>
            <ChipDrop
              options={STATUS_OPTIONS}
              value={filters.status || ''}
              onChange={(v) => onFilterChange({ status: v })}
              minWidth={140}
            />

            <div className={styles.clientDropdown} ref={clientDropdownRef}>
              <button
                type="button"
                className={styles.clientDropdownTrigger}
                onClick={() => {
                  setClientSearchOpen(!clientSearchOpen);
                  if (!clientSearchOpen) {
                    setTimeout(() => clientInputRef.current?.focus(), 50);
                  }
                }}
                aria-haspopup="listbox"
                aria-expanded={clientSearchOpen}
              >
                <span className={filters.client_id ? styles.clientDropdownSelected : styles.clientDropdownPlaceholder}>
                  {filters.client_id ? selectedClientName : 'Tous les clients'}
                </span>
                {filters.client_id ? (
                  <FiX
                    size={12}
                    className={styles.clientClear}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSelectClient('');
                    }}
                  />
                ) : (
                  <FiChevronDown size={12} />
                )}
              </button>
              {clientSearchOpen && (
                <div className={styles.clientDropdownPanel}>
                  <input
                    ref={clientInputRef}
                    type="text"
                    className={styles.clientSearchInput}
                    placeholder="Filtrer client..."
                    value={clientSearchQuery}
                    onChange={(e) => {
                      setClientSearchQuery(e.target.value);
                      setActiveIndex(-1);
                    }}
                    onKeyDown={handleClientKeyDown}
                  />
                  <div className={styles.clientListbox} role="listbox" ref={listboxRef}>
                    <div
                      role="option"
                      aria-selected={!filters.client_id}
                      className={`${styles.clientOption} ${activeIndex === 0 ? styles.clientOptionActive : ''} ${!filters.client_id ? styles.clientOptionSelected : ''}`}
                      onClick={() => handleSelectClient('')}
                    >
                      Tous les clients
                    </div>
                    {loadingClients ? (
                      <div className={styles.clientLoading}>Chargement...</div>
                    ) : (
                      filteredClients.map((client, idx) => (
                        <div
                          key={client.id}
                          role="option"
                          aria-selected={String(client.id) === String(filters.client_id)}
                          className={`${styles.clientOption} ${activeIndex === idx + 1 ? styles.clientOptionActive : ''} ${String(client.id) === String(filters.client_id) ? styles.clientOptionSelected : ''}`}
                          onClick={() => handleSelectClient(client.id)}
                        >
                          {getClientDisplayName(client)}
                        </div>
                      ))
                    )}
                    {!loadingClients && filteredClients.length === 0 && clientSearchQuery && (
                      <div className={styles.clientEmpty}>Aucun resultat</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className={styles.separator} />

        {/* Group: Période */}
        <div className={styles.filterGroup}>
          <span className={styles.filterGroupLabel}>
            <FiCalendar size={10} />
            Periode
          </span>
          <div className={styles.filterGroupControls}>
            <ChipDrop
              options={years.map((y) => ({ value: y, label: String(y) }))}
              value={filters.year || ''}
              onChange={(v) => onFilterChange({ year: v })}
              minWidth={80}
            />
            <ChipDrop
              options={months}
              value={filters.month || ''}
              onChange={(v) => onFilterChange({ month: v })}
              minWidth={140}
            />
          </div>
        </div>

        <div className={styles.separator} />

        {/* Group: Options */}
        <div className={styles.filterGroup}>
          <span className={styles.filterGroupLabel}>
            <FiSliders size={10} />
            Options
          </span>
          <div className={styles.filterGroupControls}>
            <label className={`${ntStyles.notifRow} ${styles.miniToggleRow}`} htmlFor="toggle-with-balance">
              <div className={ntStyles.notifInfo}>
                <span className={ntStyles.notifLabel}>Solde &gt; 0</span>
              </div>
              <div className={ntStyles.miniToggle}>
                <input id="toggle-with-balance" type="checkbox" checked={filters.with_balance || false} onChange={(e) => onFilterChange({ with_balance: e.target.checked })} />
                <span className={ntStyles.miniSlider} />
              </div>
            </label>
            <label className={`${ntStyles.notifRow} ${styles.miniToggleRow}`} htmlFor="toggle-with-reminders">
              <div className={ntStyles.notifInfo}>
                <span className={ntStyles.notifLabel}>Avec rappels</span>
              </div>
              <div className={ntStyles.miniToggle}>
                <input id="toggle-with-reminders" type="checkbox" checked={filters.with_reminders || false} onChange={(e) => onFilterChange({ with_reminders: e.target.checked })} />
                <span className={ntStyles.miniSlider} />
              </div>
            </label>

            <div className={styles.perPageWrapper}>
              <ChipDrop
                options={[
                  { value: 10, label: '10 / page' },
                  { value: 25, label: '25 / page' },
                  { value: 50, label: '50 / page' },
                ]}
                value={filters.per_page || 20}
                onChange={(v) => onFilterChange({ per_page: parseInt(v, 10) })}
                minWidth={100}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Row 3 — Active filter chips */}
      {activeChips.length > 0 && (
        <div className={styles.chipsRow}>
          {activeChips.map((chip) => (
            <span key={chip.key} className={styles.chip}>
              {chip.label}
              <button
                className={styles.chipRemove}
                onClick={chip.onRemove}
                aria-label={`Retirer filtre ${chip.label}`}
              >
                <FiX size={10} />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

export default CommandBar;
