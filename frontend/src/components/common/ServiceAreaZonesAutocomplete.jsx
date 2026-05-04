import React, { useEffect, useMemo, useRef, useState } from 'react';
import acStyles from './AddressAutocomplete.module.css';
import { fetchServiceAreaZones } from '../../services/settingsService';

const TYPE_LABELS = {
  commune: 'Commune',
  canton: 'Canton',
  district: 'District',
};
const CANONICAL_TOKEN_REGEX = /^(commune|canton|district):[A-Za-z0-9_-]+$/;
/** Aligné sur OperationsTab / backend : GeoAdmin ou fallback Photon peuvent renvoyer canton_name:… */
const NAMED_TOKEN_REGEX = /^(commune_name|canton_name|district_name):.+$/;

const isPersistableZoneToken = (token) => {
  const t = String(token || '');
  return CANONICAL_TOKEN_REGEX.test(t) || NAMED_TOKEN_REGEX.test(t);
};
const normalizeKey = (item) =>
  `${String(item?.type || '').toLowerCase()}::${String(item?.name || '').trim().toLowerCase()}`;

export default function ServiceAreaZonesAutocomplete({
  onSelect,
  placeholder = 'Rechercher une commune, ville ou canton',
  minChars = 2,
  debounceMs = 250,
  maxResults = 20,
  inputId,
  disabled = false,
  allowFallbackResults = false,
}) {
  const [query, setQuery] = useState('');
  const [items, setItems] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [highlight, setHighlight] = useState(-1);
  const [meta, setMeta] = useState({});
  const listboxId = `${inputId || 'service-area-search'}-zones-listbox`;
  const wrapperRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    const onDocClick = (event) => {
      if (!wrapperRef.current) return;
      if (!wrapperRef.current.contains(event.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  useEffect(() => {
    const q = String(query || '').trim();
    if (timerRef.current) clearTimeout(timerRef.current);

    if (disabled || q.length < minChars) {
      setItems([]);
      setLoading(false);
      setOpen(false);
      setHighlight(-1);
      return;
    }

    timerRef.current = setTimeout(async () => {
      try {
        setLoading(true);
        const response = await fetchServiceAreaZones({
          q,
          types: 'commune,canton,district',
          limit: maxResults,
          includeMeta: true,
        });
        const rawItems = Array.isArray(response?.items) ? response.items : [];
        const degraded = Boolean(response?.meta?.degraded);
        const persistableItems = rawItems.filter((item) =>
          isPersistableZoneToken(String(item?.token || ''))
        );
        let filteredItems = persistableItems;
        if (allowFallbackResults && degraded) {
          filteredItems = persistableItems.length > 0 ? persistableItems : rawItems;
        }
        const grouped = new Map();
        filteredItems.forEach((item) => {
          const key = normalizeKey(item);
          if (!grouped.has(key)) grouped.set(key, []);
          grouped.get(key).push(item);
        });
        const nextItems = [];
        grouped.forEach((group) => {
          const preferred = group.filter((item) => isPersistableZoneToken(String(item?.token || '')));
          if (preferred.length > 0) {
            nextItems.push(...preferred);
            return;
          }
          nextItems.push(group[0]);
        });
        setItems(nextItems);
        setMeta(response?.meta || {});
        setOpen(true);
        setHighlight(nextItems.length > 0 ? 0 : -1);
      } catch (_error) {
        setItems([]);
        setMeta({});
        setOpen(true);
        setHighlight(-1);
      } finally {
        setLoading(false);
      }
    }, debounceMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [query, minChars, debounceMs, maxResults, disabled, allowFallbackResults]);

  const options = useMemo(() => items || [], [items]);

  const selectItem = (item) => {
    setQuery('');
    setItems([]);
    setOpen(false);
    setHighlight(-1);
    onSelect?.(item);
  };

  const onInputChange = (event) => {
    if (disabled) return;
    const next = event.target.value;
    setQuery(next);
  };

  const onKeyDown = (event) => {
    if (!open || options.length === 0) return;
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setHighlight((current) => (current + 1) % options.length);
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHighlight((current) => (current - 1 + options.length) % options.length);
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      if (highlight >= 0 && highlight < options.length) {
        selectItem(options[highlight]);
      }
      return;
    }
    if (event.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <div ref={wrapperRef} className={acStyles.wrapper}>
      <input
        id={inputId}
        type="text"
        value={query}
        onChange={onInputChange}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        autoComplete="off"
        disabled={disabled}
        role="combobox"
        aria-autocomplete="list"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        className={acStyles.input}
      />
      {open && (
        <div id={listboxId} role="listbox" className={acStyles.dropdown}>
          {!loading && allowFallbackResults && meta?.degraded && (
            <div className={acStyles.notice}>Mode dégradé: résultats fallback.</div>
          )}
          {loading && <div className={acStyles.notice}>Recherche…</div>}
          {!loading && options.length === 0 && (
            <div className={acStyles.notice}>Aucune zone trouvée</div>
          )}
          {!loading &&
            options.map((item, index) => {
              const active = index === highlight;
              const typeLabel = TYPE_LABELS[item.type] || item.type || 'Zone';
              const secondary = item.canton_code
                ? `${typeLabel}, ${item.canton_code}`
                : typeLabel;
              return (
                <div
                  key={item.token || `${item.type}-${item.id}-${index}`}
                  role="option"
                  aria-selected={active}
                  onMouseDown={(event) => {
                    event.preventDefault();
                    selectItem(item);
                  }}
                  onMouseEnter={() => setHighlight(index)}
                  className={`${acStyles.optionItem} ${active ? acStyles.optionItemActive : ''}`}
                >
                  <div className={acStyles.optionLabel}>{item.name}</div>
                  <div className={acStyles.optionSecondary}>{secondary}</div>
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}
