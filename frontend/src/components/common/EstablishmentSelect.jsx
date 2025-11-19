// components/common/EstablishmentSelect.jsx
import React from 'react';
import { searchEstablishments } from '../../services/companyService';

/**
 * Autocomplete d'établissement médical (HUG, cliniques…)
 * - Débounce 250 ms
 * - Annule la requête précédente (AbortController) pour éviter les races
 * - Navigation clavier ↑ ↓ Enter Échap
 * - Clic en dehors pour fermer
 * - ARIA combobox propre
 * - États loading / error / empty
 * - Bouton Effacer (clear)
 *
 * Props:
 *  - value?: string (texte affiché initial/contrôlé)
 *  - onChange?: (text: string) => void
 *  - onPickEstablishment?: (item: {id,label,address,lat,lon,...} | null) => void
 *  - placeholder?: string
 *  - minChars?: number (par défaut 2)
 *  - limit?: number (par défaut 8)
 *  - disabled?: boolean
 */
export default function EstablishmentSelect({
  value,
  onChange,
  onPickEstablishment,
  placeholder = 'Choisir un établissement...',
  minChars = 2,
  limit = 8,
  disabled = false,
}) {
  const [q, setQ] = React.useState(value || '');
  const [items, setItems] = React.useState([]);
  const [open, setOpen] = React.useState(false);
  const [activeIndex, setActiveIndex] = React.useState(-1);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);

  const wrapRef = React.useRef(null);
  const abortRef = React.useRef(null);
  const listboxId = React.useId();

  // Sync externe → interne
  React.useEffect(() => {
    if (value !== undefined && value !== q) setQ(value || '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  // Fermer au clic en dehors
  React.useEffect(() => {
    function handleClickOutside(e) {
      if (!wrapRef.current) return;
      if (!wrapRef.current.contains(e.target)) {
        setOpen(false);
        setActiveIndex(-1);
      }
    }
    document.addEventListener('mousedown', handleClickOutside, true);
    return () => document.removeEventListener('mousedown', handleClickOutside, true);
  }, []);

  // Requête (debounced + abort)
  React.useEffect(() => {
    const search = (q || '').trim();

    // Reset si trop court
    if (search.length < minChars || disabled) {
      if (abortRef.current) abortRef.current.abort();
      setItems([]);
      setOpen(false);
      setActiveIndex(-1);
      setLoading(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const t = setTimeout(async () => {
      try {
        const arr = await searchEstablishments(search, limit, ctrl.signal);
        setItems(arr);
        setOpen(arr.length > 0);
        setActiveIndex(arr.length ? 0 : -1);
        setLoading(false);
        setError(null);
      } catch (err) {
        if (err?.name === 'AbortError') return; // requête précédente annulée : normal
        setItems([]);
        setOpen(false);
        setActiveIndex(-1);
        setLoading(false);
        setError('Impossible de charger les établissements');
      }
    }, 250);

    return () => {
      clearTimeout(t);
      if (abortRef.current) abortRef.current.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, minChars, limit, disabled]);

  const pick = (it) => {
    if (!it) {
      // clear
      setQ('');
      onChange?.('');
      onPickEstablishment?.(null);
      setItems([]);
      setOpen(false);
      setActiveIndex(-1);
      return;
    }
    const label = it?.label || it?.name || '';
    setQ(label);
    onChange?.(label);
    onPickEstablishment?.(it);
    setOpen(false);
    setActiveIndex(-1);
  };

  const onKeyDown = (e) => {
    if (disabled) return;
    // Si la liste n'est pas ouverte, Enter = auto-pick si 1 résultat
    if (!open) {
      if (e.key === 'Enter' && items.length === 1) {
        e.preventDefault();
        pick(items[0]);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (items.length) setActiveIndex((i) => (i + 1) % items.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (items.length) setActiveIndex((i) => (i - 1 + items.length) % items.length);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && activeIndex < items.length) pick(items[activeIndex]);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
      setActiveIndex(-1);
    }
  };

  return (
    <div
      ref={wrapRef}
      style={{ position: 'relative' }}
      role="combobox"
      aria-expanded={open}
      aria-controls={open ? listboxId : undefined}
      aria-haspopup="listbox"
      aria-disabled={disabled || undefined}
    >
      <div className="relative">
        <input
          value={q}
          disabled={disabled}
          onChange={(e) => {
            const next = e.target.value;
            setQ(next);
            onChange?.(next);
          }}
          onFocus={() => {
            if (!disabled && items.length > 0) setOpen(true);
          }}
          onKeyDown={onKeyDown}
          aria-autocomplete="list"
          aria-activedescendant={
            open && activeIndex >= 0 ? `${listboxId}-opt-${activeIndex}` : undefined
          }
          placeholder={placeholder}
          className={`w-full border rounded px-3 py-2 pr-9 ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`}
          autoComplete="off"
        />
      </div>

      {/* Panneaux d’état */}
      {!open && loading && (
        <div className="absolute z-50 mt-1 w-full rounded border bg-white shadow px-3 py-2 text-sm text-gray-600">
          Recherche en cours…
        </div>
      )}

      {!open && error && (
        <div className="absolute z-50 mt-1 w-full rounded border bg-white shadow px-3 py-2 text-sm text-red-600">
          {error}
        </div>
      )}

      {open && (
        <div
          id={listboxId}
          role="listbox"
          className="absolute z-50 mt-1 w-full max-h-72 overflow-auto rounded border bg-white shadow"
        >
          {items.length === 0 && !loading && !error ? (
            <div className="px-3 py-2 text-sm text-gray-600">Aucun résultat</div>
          ) : (
            items.map((it, idx) => {
              const id = `${listboxId}-opt-${idx}`;
              const isActive = idx === activeIndex;
              return (
                <div
                  key={it.id || `${it.label || it.name}-${idx}`}
                  id={id}
                  role="option"
                  aria-selected={isActive}
                  className={`px-3 py-2 cursor-pointer ${
                    isActive ? 'bg-gray-100' : 'hover:bg-gray-50'
                  }`}
                  onMouseDown={(e) => e.preventDefault()} // évite blur avant click
                  onMouseEnter={() => setActiveIndex(idx)}
                  onClick={() => pick(it)}
                >
                  <div className="font-medium">{it.label || it.name}</div>
                  {it.address && <div className="text-xs text-gray-600">{it.address}</div>}
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
