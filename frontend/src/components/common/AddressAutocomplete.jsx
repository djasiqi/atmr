// frontend/src/components/common/AddressAutocomplete.jsx
import React, { useEffect, useMemo, useRef, useState, useTransition, useCallback } from 'react';
import { createPortal } from 'react-dom';
import apiClient from '../../utils/apiClient';
import acStyles from './AddressAutocomplete.module.css';

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 min
const CACHE_MAX_SIZE = 50;
const DEFAULT_BIAS = { lat: 46.2044, lon: 6.1432 };

function buildCacheKey(query, bias) {
  const q = (query || '').toString().trim().toLowerCase();
  if (!q) return null;
  if (!bias || bias.lat == null || bias.lon == null) return q;
  const lat = Number(bias.lat).toFixed(2);
  const lon = Number(bias.lon).toFixed(2);
  return `${q}|${lat},${lon}`;
}

function createAutocompleteLRUCache() {
  const entries = new Map();
  const order = [];

  function get(key) {
    const entry = entries.get(key);
    if (!entry) return null;
    if (Date.now() - entry.at > CACHE_TTL_MS) {
      entries.delete(key);
      const i = order.indexOf(key);
      if (i >= 0) order.splice(i, 1);
      return null;
    }
    return entry.data;
  }

  function set(key, data) {
    if (entries.has(key)) {
      const i = order.indexOf(key);
      if (i >= 0) order.splice(i, 1);
    } else if (order.length >= CACHE_MAX_SIZE) {
      const oldest = order.shift();
      entries.delete(oldest);
    }
    entries.set(key, { data, at: Date.now() });
    order.push(key);
  }

  function clear() {
    entries.clear();
    order.length = 0;
  }

  return { get, set, clear };
}

const autocompleteCache = createAutocompleteLRUCache();

/** Vide le cache LRU autocomplete (logout / changement d'entreprise). */
export function clearAddressAutocompleteCache() {
  autocompleteCache.clear();
}

export default function AddressAutocomplete({
  name,
  value,
  onChange,
  onSelect,
  placeholder = 'Saisir une adresse…',
  minChars = 3,
  debounceMs = 350,
  bias, // { lat, lon } optionnel – par défaut centre Genève
  maxResults = 8,
  prependSuggestions = [],
  inputClassName,
  inputId,
  /** Sans bordure ni halo : à utiliser quand le parent porte le cadre (fieldGroup, etc.) */
  flushInput = false,
  ...restProps
}) {
  const logTag =
    name === 'dropoff_location' ? 'DROP_OFF' :
    name === 'pickup_location' ? 'PICKUP' : 'ADDRESS';
  const log = useCallback((phase, payload) => {
    if (process.env.NODE_ENV !== 'development') return;
    // Jamais de texte d'adresse / query brute dans les logs
    const safe =
      payload && typeof payload === 'object'
        ? {
            ...payload,
            query: payload.query != null ? `[len=${String(payload.query).length}]` : undefined,
          }
        : payload;
    if (safe === undefined) {
      console.log(`[${logTag}][${phase}]`);
      return;
    }
    console.log(`[${logTag}][${phase}]`, safe);
  }, [logTag]);

  const [query, setQuery] = useState(value || '');
  const [items, setItems] = useState([]);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [justSelected, setJustSelected] = useState(false); // Pour éviter la réouverture après sélection
  const [userIsTyping, setUserIsTyping] = useState(false); // Tracker si l'utilisateur tape activement
  const quickSuggestions = useMemo(
    () => (Array.isArray(prependSuggestions) ? prependSuggestions.filter(Boolean) : []),
    [prependSuggestions]
  );

  const [, startTransition] = useTransition();

  const abortRef = useRef(null);
  const wrapRef = useRef(null);
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);
  const rafIdRef = useRef(null);
  const pendingRef = useRef(false);
  const requestSeqRef = useRef(0);
  /** Empêche une réponse API tardive de rouvrir le menu après Escape / clic extérieur. */
  const suggestionsDismissedRef = useRef(false);

  // Biais géographique (Genève par défaut) – useMemo pour stabilité des deps useCallback
  const BIAS = useMemo(() => (bias != null ? bias : DEFAULT_BIAS), [bias]);

  // Sync externe -> interne
  useEffect(() => {
    const next = value ? String(value) : '';
    setQuery(next);
    log('PROP_SYNC', { valueLen: next.length });
  }, [value, log]);

  useEffect(() => {
    log('RENDER', {
      valueLen: value ? String(value).length : 0,
      queryLen: query ? String(query).length : 0,
      open,
      loading,
      suggestions: items.length,
      justSelected,
      userIsTyping,
    });
  }, [value, query, open, loading, items.length, justSelected, userIsTyping, log]);

  useEffect(() => () => {
    log('UNMOUNT');
  }, [log]);

  // Fermer la liste si on clique à l'extérieur (portail inclus via dropdownRef)
  useEffect(() => {
    function onDocClick(e) {
      const t = e.target;
      if (wrapRef.current?.contains(t) || dropdownRef.current?.contains(t)) return;
      suggestionsDismissedRef.current = true;
      setOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  // Debounce util
  const debounce = useMemo(() => {
    let t;
    return (fn, ms) => {
      clearTimeout(t);
      t = setTimeout(fn, ms);
    };
  }, []);

  // Helper pour détecter les erreurs d'annulation (fetch natif ou Axios)
  const isCanceledError = useCallback((error) => {
    if (error?.name === 'AbortError') return true;
    if (error?.name === 'CanceledError' || error?.code === 'ERR_CANCELED') return true;
    return false;
  }, []);

  // Fetch backend uniquement (public + favoris JWT) — pas de Photon navigateur
  const fetchSuggestions = useCallback(async (queryText, signal) => {
    const q = (queryText || '').toString().trim();

    const DEFAULT_VALUES = ['non spécifié', 'non specifie', 'n/a', 'na', ''];
    if (DEFAULT_VALUES.includes(q.toLowerCase())) {
      return [];
    }

    const cacheKey = buildCacheKey(q, BIAS);
    if (cacheKey) {
      const cached = autocompleteCache.get(cacheKey);
      if (cached != null) {
        return cached;
      }
    }

    const publicUrl = `geocode/autocomplete?q=${encodeURIComponent(q)}&lat=${encodeURIComponent(
      BIAS.lat
    )}&lon=${encodeURIComponent(BIAS.lon)}&limit=${encodeURIComponent(maxResults)}`;

    let publicResults = [];
    try {
      const res = await apiClient.get(publicUrl, { signal });
      if (res.status === 200 && Array.isArray(res.data)) {
        publicResults = res.data;
      }
    } catch (error) {
      if (isCanceledError(error)) {
        return [];
      }
      if (process.env.NODE_ENV === 'development') {
        console.error('[AddressAutocomplete] erreur backend public', error?.code || error?.message);
      }
    }

    let favoriteResults = [];
    try {
      const favUrl = `geocode/favorites/autocomplete?q=${encodeURIComponent(q)}&limit=6`;
      const favRes = await apiClient.get(favUrl, {
        signal,
        skipAuthRedirect: true,
        validateStatus: (status) => status === 200 || status === 401 || status === 403,
      });
      if (favRes.status === 200 && Array.isArray(favRes.data)) {
        favoriteResults = favRes.data;
      }
    } catch (error) {
      if (isCanceledError(error)) {
        return [];
      }
      // Anonyme / hors entreprise : silencieux
    }

    const seen = new Set();
    const merged = [];
    for (const item of [...favoriteResults, ...publicResults]) {
      const key =
        item?.place_id ||
        `${(item?.address || item?.label || '').trim().toLowerCase()}|${item?.lat}|${item?.lon}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(item);
      if (merged.length >= maxResults) break;
    }

    if (cacheKey) autocompleteCache.set(cacheKey, merged);
    return merged;
  }, [BIAS, maxResults, isCanceledError]);

  // Charger les suggestions (debounce + abort)
  useEffect(() => {
    // Ne pas charger si on vient de sélectionner une adresse
    if (justSelected) {
      return;
    }

    const queryToUse = query;
    if (!queryToUse || (typeof queryToUse === 'string' ? queryToUse.trim().length : 0) < minChars) {
      startTransition(() => {
      setItems([]);
      const focused = document.activeElement === inputRef.current;
      setOpen(focused && quickSuggestions.length > 0);
      setHighlight(quickSuggestions.length > 0 ? 0 : -1);
      setLoading(false);
      });
      return;
    }
    debounce(async () => {
      try {
        const ctl = new AbortController();
        abortRef.current = ctl;
        const requestSeq = ++requestSeqRef.current;
        startTransition(() => {
        setLoading(true);
        });

        const queryStr = String(queryToUse || '');
        log('FETCH_START', { queryLen: queryStr.length, requestSeq });
        const next = await fetchSuggestions(queryStr, ctl.signal);
        // Favoris JWT en tête + alias/providers
        let enriched = Array.isArray(next) ? next : [];
        if (requestSeq !== requestSeqRef.current) {
          log('FETCH_IGNORED_STALE', { queryLen: queryStr.length, requestSeq });
          return;
        }
        if (enriched.length > 0) {
          log('FETCH_SUCCESS', { queryLen: queryStr.length, count: enriched.length, requestSeq });
        } else {
          log('FETCH_EMPTY', { queryLen: queryStr.length, requestSeq });
        }

        // ✅ PERF: Utiliser startTransition pour les mises à jour non-urgentes
        startTransition(() => {
          setItems(enriched);
          const focused = document.activeElement === inputRef.current;
          const hasQuery = queryStr.trim().length >= minChars;
          const mayOpen =
            focused && !justSelected && hasQuery && !suggestionsDismissedRef.current;
          setOpen(mayOpen);
          setHighlight(enriched.length ? 0 : -1);
          setLoading(false);
        });
      } catch (error) {
        // ✅ PHASE 3: Ignorer les erreurs d'annulation (AbortError pour fetch natif, CanceledError pour Axios)
        if (isCanceledError(error)) {
          log('FETCH_ERROR', { canceled: true });
          return; // Ne pas mettre à jour l'état si la requête a été annulée
        }
        log('FETCH_ERROR', { canceled: false, message: error?.message || String(error) });
        startTransition(() => {
          setItems([]);
          const focused = document.activeElement === inputRef.current;
          const hasQuery = String(queryToUse || '').trim().length >= minChars;
          const mayOpen =
            focused && !justSelected && hasQuery && !suggestionsDismissedRef.current;
          setOpen(mayOpen);
          setLoading(false);
        });
      } finally {
        setLoading(false);
      }
    }, debounceMs);
    // ✅ PHASE 3: Cleanup — annuler les requêtes en cours au démontage / avant le prochain effet
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, [
    query,
    minChars,
    debounceMs,
    justSelected,
    debounce,
    fetchSuggestions,
    isCanceledError,
    quickSuggestions.length,
    log,
  ]);

  function handleInputChange(e) {
    const v = e.target.value;
    log('CHANGE', { value: v });
    setQuery(v);
    setJustSelected(false); // Réinitialiser le flag si l'utilisateur modifie
    setUserIsTyping(true); // L'utilisateur est en train de taper
    suggestionsDismissedRef.current = false;
    const focused = document.activeElement === inputRef.current;
    if ((v || '').trim().length >= minChars && focused) {
      setOpen(true);
    }
    onChange?.({ target: { name, value: v } });
    
    // Recalculer la position du dropdown si ouvert (au cas où l'input change de taille)
    if (open && inputRef.current) {
      requestAnimationFrame(() => {
        const rect = inputRef.current.getBoundingClientRect();
        setDropdownPosition({
          top: rect.bottom + 2,
          left: rect.left,
          width: rect.width,
        });
      });
    }
  }

  // Groupes : alias canoniques, Google Places, puis autres résultats
  const aliases = useMemo(() => items.filter((i) => i.source === 'alias'), [items]);
  const googlePlaces = useMemo(() => items.filter((i) => i.source === 'google_places'), [items]);
  const others = useMemo(
    () => items.filter((i) => i.source !== 'google_places' && i.source !== 'alias'),
    [items]
  );
  const visibleItems = useMemo(
    () => [...quickSuggestions, ...aliases, ...googlePlaces, ...others],
    [quickSuggestions, aliases, googlePlaces, others]
  );

  useEffect(() => {
    log('OPEN_STATE', { open });
  }, [open, log]);

  useEffect(() => {
    log('SUGGESTIONS', {
      quick: quickSuggestions.length,
      aliases: aliases.length,
      google: googlePlaces.length,
      others: others.length,
      total: visibleItems.length,
    });
  }, [quickSuggestions.length, aliases.length, googlePlaces.length, others.length, visibleItems.length, log]);

  async function chooseItem(it) {
    log('SELECT', { label: it?.label || it?.address || '' });
    // Utiliser directement le label qui est déjà bien formaté
    const fullAddress = it?.label || it?.address || '';

    setQuery(fullAddress);

    // Fermer le menu et vider les suggestions
    setOpen(false);
    setItems([]);
    setHighlight(-1);
    setJustSelected(true);
    setUserIsTyping(false);

    onChange?.({ target: { name, value: fullAddress } });

    // ✅ Si c'est une suggestion Google Places avec place_id, récupérer les coordonnées GPS
    // ⚠️ IMPORTANT : Ne pas appeler onSelect immédiatement pour éviter double mise à jour
    // On l'appellera après l'enrichissement avec les coordonnées GPS
    const shouldEnrich = it.source === 'google_places' && it.place_id && (!it.lat || !it.lon);
    
    if (shouldEnrich) {
      try {
        const response = await apiClient.get(
          `geocode/place-details?place_id=${encodeURIComponent(it.place_id)}`
        );

        if (response.status === 200 && response.data) {
          const details = response.data;

          // Extraire les composants d'adresse depuis address_components
          const addressComponents = details.address_components || [];
          const streetNumber = addressComponents.find((c) =>
            c.types?.includes('street_number')
          )?.long_name;
          const route = addressComponents.find((c) => c.types?.includes('route'))?.long_name;
          const city =
            addressComponents.find((c) => c.types?.includes('locality'))?.long_name ||
            addressComponents.find((c) => c.types?.includes('administrative_area_level_2'))
              ?.long_name ||
            it.city;
          const postcode =
            addressComponents.find((c) => c.types?.includes('postal_code'))?.long_name ||
            it.postcode;

          // Construire la rue au format suisse : "Rue Numéro" (pas "Numéro Rue")
          const streetAddress = [route, streetNumber].filter(Boolean).join(' ') || route || '';

          // Format requis : Rue Numéro, Code postal, Ville (pays ajouté côté facture si hors domicile entreprise)
          const addressParts = [streetAddress];
          if (postcode) addressParts.push(postcode);
          if (city) addressParts.push(city);
          const enrichedAddressStr = addressParts.filter(Boolean).join(', ');

          const isPoi = it.types?.some((t) =>
            ['establishment', 'point_of_interest'].includes(t)
          );
          const placeName = isPoi ? (it.main_text || it.label) : null;
          const placeNameIsDistinct = placeName && placeName.trim() !== streetAddress.trim();

          // Toujours utiliser l'adresse enrichie (avec code postal). Ne préfixer par le nom du lieu
          // que pour les POI (ex. HUG) quand il est distinct de la rue, pour éviter les doublons.
          let finalLabel = fullAddress || it.label || '';
          if (enrichedAddressStr) {
            if (placeNameIsDistinct && isPoi) {
              finalLabel = `${placeName}, ${enrichedAddressStr}`;
            } else {
              finalLabel = enrichedAddressStr;
            }
          }

          // Enrichir l'item avec les coordonnées GPS et les composants d'adresse
          // ✅ Nom du lieu (ex. Fondation Butini) pour remplir "Établissement de résidence"
          const enrichedItem = {
            ...it,
            lat: details.lat,
            lon: details.lon,
            address: streetAddress || it.address || fullAddress,
            street: route || it.street || '',
            street_number: streetNumber || it.street_number || '',
            city: city || it.city || '',
            postcode: postcode || it.postcode || '',
            label: finalLabel,
            name: details.name || it.name || null,
          };

          // ✅ Appeler onSelect avec l'item enrichi (qui préserve l'adresse originale)
          onSelect?.(enrichedItem);
          return;
        }
      } catch (error) {
        console.warn('⚠️ Erreur lors de la récupération des coordonnées GPS:', error);
        // En cas d'erreur, appeler onSelect avec l'item original
        onSelect?.(it);
      }
    } else {
      // Sinon, passer l'item tel quel (Photon, ou Google avec coordonnées déjà présentes)
      onSelect?.(it);
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Escape') {
      if (open) {
        e.preventDefault();
        suggestionsDismissedRef.current = true;
        setOpen(false);
      }
      return;
    }
    if (!open || visibleItems.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlight((h) => (h + 1) % visibleItems.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlight((h) => (h - 1 + visibleItems.length) % visibleItems.length);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlight >= 0 && highlight < visibleItems.length) {
        chooseItem(visibleItems[highlight]);
      }
    }
  }

  const listboxId = `${name || 'address'}-ac-listbox`;
  const activeId = highlight >= 0 ? `${name || 'address'}-ac-option-${highlight}` : undefined;

  // Calculer la position du dropdown en position fixed — 1 recalcul max par frame (RAF scheduler)
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0, width: 0 });

  const scheduleUpdate = useCallback(() => {
    if (pendingRef.current) return;
    pendingRef.current = true;
    if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = requestAnimationFrame(() => {
      pendingRef.current = false;
      rafIdRef.current = null;
      if (!inputRef.current) return;
      const rect = inputRef.current.getBoundingClientRect();
      setDropdownPosition({
        top: rect.bottom + 2,
        left: rect.left,
        width: rect.width,
      });
    });
  }, []);

  const isScrollable = useCallback((el, style) => {
    const vals = ['auto', 'scroll', 'overlay'];
    return (
      el.scrollHeight > el.clientHeight ||
      vals.includes(style.overflow) ||
      vals.includes(style.overflowY) ||
      vals.includes(style.overflowX)
    );
  }, []);

  useEffect(() => {
    if (open && inputRef.current) {
      scheduleUpdate();
      const rafId1 = requestAnimationFrame(() => {
        requestAnimationFrame(scheduleUpdate);
      });
      const timeoutId = setTimeout(scheduleUpdate, 50);

      window.addEventListener('scroll', scheduleUpdate, true);
      window.addEventListener('resize', scheduleUpdate);

      let resizeObserver = null;
      if (window.ResizeObserver && inputRef.current) {
        resizeObserver = new ResizeObserver(scheduleUpdate);
        resizeObserver.observe(inputRef.current);
      }

      const vv = typeof window !== 'undefined' && window.visualViewport;
      if (vv) {
        vv.addEventListener('resize', scheduleUpdate);
        vv.addEventListener('scroll', scheduleUpdate);
      }

      const scrollContainers = [];
      let parent = inputRef.current.parentElement;
      while (parent && parent !== document.body) {
        const style = window.getComputedStyle(parent);
        if (isScrollable(parent, style)) {
          parent.addEventListener('scroll', scheduleUpdate, true);
          scrollContainers.push(parent);
        }
        parent = parent.parentElement;
      }

      return () => {
        cancelAnimationFrame(rafId1);
        clearTimeout(timeoutId);
        if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
        window.removeEventListener('scroll', scheduleUpdate, true);
        window.removeEventListener('resize', scheduleUpdate);
        if (vv) {
          vv.removeEventListener('resize', scheduleUpdate);
          vv.removeEventListener('scroll', scheduleUpdate);
        }
        if (resizeObserver) resizeObserver.disconnect();
        scrollContainers.forEach((c) => c.removeEventListener('scroll', scheduleUpdate, true));
      };
    } else if (!open) {
      setDropdownPosition({ top: 0, left: 0, width: 0 });
    }
  }, [open, items.length, scheduleUpdate, isScrollable]);

  return (
    <div ref={wrapRef} className={acStyles.wrapper}>
      <input
        ref={inputRef}
        id={inputId}
        type="text"
        name={name}
        value={query}
        onChange={handleInputChange}
        onFocus={() => {
          log('FOCUS', { value: query });
          const focused = document.activeElement === inputRef.current;
          if (focused && quickSuggestions.length > 0) {
            setOpen(true);
            setHighlight(0);
            return;
          }
          if (focused && items.length > 0) {
            setOpen(true);
            setHighlight((h) => (h >= 0 ? h : 0));
          }
        }}
        onBlur={() => {
          log('BLUR', { value: query });
          // Réinitialiser le mode typing quand on quitte le champ
          setUserIsTyping(false);
        }}
        placeholder={placeholder}
        autoComplete="off"
        role="combobox"
        aria-autocomplete="list"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        aria-activedescendant={open ? activeId : undefined}
        className={[flushInput ? acStyles.inputFlush : acStyles.input, inputClassName]
          .filter(Boolean)
          .join(' ')}
        {...restProps}
        onKeyDown={onKeyDown}
      />

      {open &&
        createPortal(
          <div
            ref={dropdownRef}
            id={listboxId}
            role="listbox"
            className={acStyles.dropdown}
            style={(() => {
              let top = dropdownPosition.top;
              let left = dropdownPosition.left;
              let width = dropdownPosition.width;
              if ((top <= 0 || left < 0 || width <= 0) && inputRef.current) {
                const rect = inputRef.current.getBoundingClientRect();
                top = rect.bottom + 4;
                left = rect.left;
                width = rect.width;
              }
              return {
                position: 'fixed',
                top: `${top}px`,
                left: `${left}px`,
                width: `${width}px`,
                zIndex: 'var(--z-modal-popover)',
              };
            })()}
          >
          {loading && <div className={acStyles.notice}>Recherche…</div>}

          {!loading && visibleItems.length === 0 && (
            <div className={acStyles.notice}>Aucun résultat</div>
          )}

          {!loading && visibleItems.length > 0 && (
            <>
              {quickSuggestions.length > 0 && (
                <>
                  <div className={acStyles.sectionHeaderGoogle}>Suggestions</div>
                  {quickSuggestions.map((it, idx) => {
                    const globalIndex = idx;
                    const active = globalIndex === highlight;
                    return (
                      <div
                        id={`${name || 'address'}-ac-option-${globalIndex}`}
                        key={`quick-${it.label || idx}`}
                        role="option"
                        aria-selected={active}
                        onMouseDown={(e) => { e.preventDefault(); chooseItem(it); }}
                        onMouseEnter={() => setHighlight(globalIndex)}
                        className={`${acStyles.optionItem} ${acStyles.optionItemGoogle} ${active ? acStyles.optionItemActive : ''}`}
                      >
                        <div className={acStyles.optionLabel}>{it.label || 'Suggestion'}</div>
                        {it.secondary_text && <div className={acStyles.optionSecondary}>{it.secondary_text}</div>}
                      </div>
                    );
                  })}
                </>
              )}

              {(aliases.length > 0 || googlePlaces.length > 0) && (
                <>
                  <div className={acStyles.sectionHeaderGoogle}>Suggestions</div>
                  {aliases.map((it, idx) => {
                    const globalIndex = quickSuggestions.length + idx;
                    const active = globalIndex === highlight;
                    return (
                      <div
                        id={`${name || 'address'}-ac-option-${globalIndex}`}
                        key={it.place_id || `alias-${it.label || idx}`}
                        role="option"
                        aria-selected={active}
                        onMouseDown={(e) => { e.preventDefault(); chooseItem(it); }}
                        onMouseEnter={() => setHighlight(globalIndex)}
                        className={`${acStyles.optionItem} ${acStyles.optionItemGoogle} ${active ? acStyles.optionItemActive : ''}`}
                      >
                        <div className={acStyles.optionLabel}>{it.main_text || it.label}</div>
                        {(it.secondary_text || it.address) && (
                          <div className={acStyles.optionSecondary}>{it.secondary_text || it.address}</div>
                        )}
                      </div>
                    );
                  })}
                  {googlePlaces.map((it, idx) => {
                    const globalIndex = quickSuggestions.length + aliases.length + idx;
                    const active = globalIndex === highlight;
                    return (
                      <div
                        id={`${name || 'address'}-ac-option-${globalIndex}`}
                        key={it.place_id || `google-${idx}`}
                        role="option"
                        aria-selected={active}
                        onMouseDown={(e) => { e.preventDefault(); chooseItem(it); }}
                        onMouseEnter={() => setHighlight(globalIndex)}
                        className={`${acStyles.optionItem} ${acStyles.optionItemGoogle} ${active ? acStyles.optionItemActive : ''}`}
                      >
                        <div className={acStyles.optionLabel}>{it.main_text || it.label}</div>
                        {it.secondary_text && <div className={acStyles.optionSecondary}>{it.secondary_text}</div>}
                      </div>
                    );
                  })}
                </>
              )}

              {others.length > 0 && (
                <>
                  <div className={acStyles.sectionHeader}>Autres résultats</div>
                  {others.map((it, idx) => {
                    const globalIndex = quickSuggestions.length + aliases.length + googlePlaces.length + idx;
                    const active = globalIndex === highlight;
                    const line =
                      [it.address, it.postcode, it.city, it.country].filter(Boolean).join(' · ') ||
                      it.label;
                    const key =
                      it.lat != null && it.lon != null
                        ? `${it.lat},${it.lon}-${idx}`
                        : `${it.label || it.address || 'addr'}-oth-${idx}`;
                    return (
                      <div
                        id={`${name || 'address'}-ac-option-${globalIndex}`}
                        key={key}
                        role="option"
                        aria-selected={active}
                        onMouseDown={(e) => { e.preventDefault(); chooseItem(it); }}
                        onMouseEnter={() => setHighlight(globalIndex)}
                        className={`${acStyles.optionItem} ${active ? acStyles.optionItemActive : ''}`}
                      >
                        <div className={acStyles.optionLabel}>{it.label || it.address}</div>
                        {line && <div className={acStyles.optionSecondary}>{line}</div>}
                      </div>
                    );
                  })}
                </>
              )}
            </>
          )}
          </div>,
          document.body
        )}
    </div>
  );
}
