// frontend/src/components/common/AddressAutocomplete.jsx
import React, { useEffect, useMemo, useRef, useState, useDeferredValue, useTransition, useCallback } from 'react';
import { createPortal } from 'react-dom';
import apiClient from '../../utils/apiClient';
import acStyles from './AddressAutocomplete.module.css';

export default function AddressAutocomplete({
  name,
  value,
  onChange,
  onSelect,
  placeholder = 'Saisir une adresse…',
  minChars = 2,
  debounceMs = 250,
  bias, // { lat, lon } optionnel – par défaut centre Genève
  maxResults = 8,
  inputClassName,
  inputId,
  ...restProps
}) {
  const [query, setQuery] = useState(value || '');
  const [items, setItems] = useState([]);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [justSelected, setJustSelected] = useState(false); // Pour éviter la réouverture après sélection
  const [userIsTyping, setUserIsTyping] = useState(false); // Tracker si l'utilisateur tape activement

  // ✅ PERF: useDeferredValue pour différer les recherches et améliorer l'INP
  const deferredQuery = useDeferredValue(query);
  const [, startTransition] = useTransition();

  const abortRef = useRef(null);
  const wrapRef = useRef(null);
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);
  const rafIdRef = useRef(null);
  const pendingRef = useRef(false);

  // Biais géographique (Genève par défaut)
  const BIAS = bias || { lat: 46.2044, lon: 6.1432 };

  // Base Photon (env front ou cloud public)
  const PHOTON_BASE = process.env.REACT_APP_PHOTON_URL || 'https://photon.komoot.io';

  // Sync externe -> interne
  useEffect(() => setQuery(value ? String(value) : ''), [value]);

  // Fermer la liste si on clique à l'extérieur
  useEffect(() => {
    function onDocClick(e) {
      if (!wrapRef.current) return;
      if (!wrapRef.current.contains(e.target)) setOpen(false);
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

  // Normalise les features Photon vers notre format
  function normalizePhoton(features) {
    return (features || []).map((f) => {
      const props = f.properties || {};
      const coords = f.geometry?.coordinates || []; // [lon, lat]
      const lon = Number(coords[0]);
      const lat = Number(coords[1]);

      // Construire l'adresse complète avec numéro et rue
      // Format : "Rue, Numéro" (avec virgule)
      const street = props.street || '';
      const housenumber = props.housenumber || '';
      const fullStreetAddress = street && housenumber ? `${street}, ${housenumber}` : street || '';

      const postcode = props.postcode || '';
      const city = props.city || props.locality || '';
      const placeName = props.name;

      // Construire le label : FORCER le format "Rue, Numéro, Code Postal, Ville" (SANS PAYS)
      // ✅ Le code postal doit TOUJOURS être inclus s'il est disponible
      // ✅ Le pays ne doit JAMAIS être inclus dans le label
      let label = '';

      if (placeName && fullStreetAddress) {
        // Lieu nommé avec adresse complète : "Nom, Rue, Numéro, CP, Ville"
        const addressParts = [fullStreetAddress];
        // ✅ Toujours inclure le code postal s'il est disponible
        if (postcode) addressParts.push(postcode);
        if (city) addressParts.push(city);
        // ❌ NE PAS inclure le pays
        const addressStr = addressParts.join(', ');
        label = `${placeName}, ${addressStr}`;
      } else if (placeName && street) {
        // Lieu nommé avec rue mais sans numéro : "Nom, Rue, CP, Ville"
        const addressParts = [street];
        // ✅ Toujours inclure le code postal s'il est disponible
        if (postcode) addressParts.push(postcode);
        if (city) addressParts.push(city);
        // ❌ NE PAS inclure le pays
        const addressStr = addressParts.join(', ');
        label = `${placeName}, ${addressStr}`;
      } else if (placeName) {
        // Lieu nommé sans adresse : juste le nom (fallback)
        label = placeName;
      } else if (fullStreetAddress && city) {
        // Format complet : "Rue, Numéro, CP, Ville"
        const parts = [fullStreetAddress];
        // ✅ Toujours inclure le code postal s'il est disponible
        if (postcode) parts.push(postcode);
        if (city) parts.push(city);
        // ❌ NE PAS inclure le pays
        label = parts.join(', ');
      } else if (fullStreetAddress && postcode) {
        // Rue avec numéro et code postal mais sans ville : "Rue, Numéro, CP"
        const parts = [fullStreetAddress, postcode];
        label = parts.join(', ');
      } else if (street && city) {
        // Rue sans numéro : "Rue, CP, Ville"
        const parts = [street];
        // ✅ Toujours inclure le code postal s'il est disponible
        if (postcode) parts.push(postcode);
        if (city) parts.push(city);
        // ❌ NE PAS inclure le pays
        label = parts.join(', ');
      } else if (street && postcode) {
        // Rue avec code postal mais sans ville : "Rue, CP"
        label = `${street}, ${postcode}`;
      } else if (city) {
        // Au moins la ville : inclure le code postal s'il est disponible
        label = postcode && city ? `${postcode} ${city}` : city;
      } else if (postcode) {
        // Seulement le code postal (cas rare)
        label = postcode;
      } else {
        // Dernier recours
        label = props.osm_value || 'Adresse';
      }

      return {
        source: 'photon',
        label,
        address: fullStreetAddress || street || null,
        postcode: postcode || null,
        city: city || null,
        country: props.country || null,
        lat,
        lon,
        raw: f,
        name: placeName || null,
      };
    });
  }

  // Helper pour détecter les erreurs d'annulation (fetch natif ou Axios)
  const isCanceledError = useCallback((error) => {
    // AbortError pour fetch natif
    if (error?.name === 'AbortError') return true;
    // CanceledError pour Axios
    if (error?.name === 'CanceledError' || error?.code === 'ERR_CANCELED') return true;
    return false;
  }, []);

  // Fetch proxy backend puis fallback Photon direct
  const fetchSuggestions = useCallback(async (queryText, signal) => {
    const q = (queryText || '').toString().trim();

    // ✅ Ignorer les valeurs par défaut qui ne sont pas de vraies adresses
    const DEFAULT_VALUES = ['non spécifié', 'non specifie', 'n/a', 'na', ''];
    if (DEFAULT_VALUES.includes(q.toLowerCase())) {
      console.log(`[AddressAutocomplete] ⚠️ Ignoré valeur par défaut: "${q}"`);
      return [];
    }

    // 1) Proxy backend — mélange alias/favoris + Photon si ton backend le fait
    try {
      // ✅ URL relative au baseURL (pas de / initial) pour que prod atteigne /api/v1/geocode/autocomplete
      const url = `geocode/autocomplete?q=${encodeURIComponent(q)}&lat=${encodeURIComponent(
        BIAS.lat
      )}&lon=${encodeURIComponent(BIAS.lon)}&limit=${encodeURIComponent(maxResults)}`;
      const res = await apiClient.get(url, { signal });
      if (res.status === 200) {
        const data = res.data || [];
        if (Array.isArray(data)) {
          if (data.length > 0) {
            console.log(`[AddressAutocomplete] ✅ Backend retourne ${data.length} résultats pour "${q}"`);
            return data;
          } else {
            // Ne pas logger d'avertissement pour les valeurs par défaut
            const DEFAULT_VALUES = ['non spécifié', 'non specifie', 'n/a', 'na'];
            if (!DEFAULT_VALUES.includes(q.toLowerCase())) {
              console.log(`[AddressAutocomplete] ⚠️ Backend retourne une liste vide pour "${q}"`);
            }
          }
      }
      } else {
        console.warn(`[AddressAutocomplete] ⚠️ Erreur backend (${res.status}) pour "${q}"`);
      }
    } catch (error) {
      // ✅ Ne pas logger les erreurs d'annulation (AbortError pour fetch natif, CanceledError pour Axios)
      if (isCanceledError(error)) {
        return [];
      }
      console.error(`[AddressAutocomplete] ❌ Erreur lors de l'appel backend:`, error);
    }

    // 2) Fallback Photon direct
    try {
      const url = new URL('/api', PHOTON_BASE);
      url.searchParams.set('q', q);
      url.searchParams.set('limit', String(maxResults));
      url.searchParams.set('lang', 'fr');
      url.searchParams.set('lat', String(BIAS.lat));
      url.searchParams.set('lon', String(BIAS.lon));

      const res = await fetch(url.toString(), { signal });
      if (!res.ok) throw new Error(`Photon error: ${res.status}`);
      const data = await res.json();
      const feats = Array.isArray(data?.features) ? data.features : [];
      const normalized = normalizePhoton(feats);
      if (normalized.length > 0) {
        console.log(`[AddressAutocomplete] ✅ Photon fallback retourne ${normalized.length} résultats pour "${q}"`);
      } else {
        console.log(`[AddressAutocomplete] ⚠️ Photon fallback ne trouve aucun résultat pour "${q}"`);
      }
      return normalized;
    } catch (error) {
      // ✅ Ne pas logger les erreurs d'annulation (AbortError pour fetch natif, CanceledError pour Axios)
      if (isCanceledError(error)) {
        return [];
      }
      console.error(`[AddressAutocomplete] ❌ Erreur Photon fallback:`, error);
      return [];
    }
  }, [BIAS.lat, BIAS.lon, PHOTON_BASE, maxResults, isCanceledError]);

  // Charger les suggestions (debounce + abort)
  useEffect(() => {
    // Ne pas charger si on vient de sélectionner une adresse
    if (justSelected) {
      return;
    }

    // ✅ PERF: Utiliser deferredQuery pour réduire le travail urgent
    const queryToUse = deferredQuery;
    if (!queryToUse || (typeof queryToUse === 'string' ? queryToUse.trim().length : 0) < minChars) {
      startTransition(() => {
      setItems([]);
      setOpen(false);
      setLoading(false);
      });
      return;
    }
    debounce(async () => {
      try {
        abortRef.current?.abort();
        const ctl = new AbortController();
        abortRef.current = ctl;
        startTransition(() => {
        setLoading(true);
        });

        const queryStr = String(queryToUse || '');
        const next = await fetchSuggestions(queryStr, ctl.signal);
        let enriched = Array.isArray(next) ? next : [];

        // Filet de sécu : si l'utilisateur tape "hug" et qu'aucun alias n'est présent,
        // on injecte l'adresse HUG en tête (évite de dépendre à 100% du backend).
        const qn = (queryToUse || '').toString().trim().toLowerCase();
        const hasAlias = enriched.some((it) => it.source === 'alias');
        const looksHUG = /\bhug\b|h[ôo]pit(?:al|aux).+gen[eè]ve|\bh[ôo]pital\s+cantonal\b/.test(qn);
        if (looksHUG && !hasAlias) {
          enriched.unshift({
            source: 'alias',
            label: 'Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
            address: 'Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
            lat: 46.19226,
            lon: 6.14262,
            category: 'hospital',
          });
        }

        // ✅ PERF: Utiliser startTransition pour les mises à jour non-urgentes
        startTransition(() => {
        setItems(enriched);
        // Ne rouvrir le menu que si l'utilisateur tape activement
        if (userIsTyping && !justSelected) {
          setOpen(true);
        }
        setHighlight(enriched.length ? 0 : -1);
          setLoading(false);
        });
      } catch (error) {
        // ✅ PHASE 3: Ignorer les erreurs d'annulation (AbortError pour fetch natif, CanceledError pour Axios)
        if (isCanceledError(error)) {
          return; // Ne pas mettre à jour l'état si la requête a été annulée
        }
        startTransition(() => {
        setItems([]);
        setOpen(false);
          setLoading(false);
        });
      } finally {
        setLoading(false);
      }
    }, debounceMs);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // ✅ PHASE 3: Cleanup function pour annuler les requêtes en cours lors du démontage
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, [query, minChars, debounceMs, BIAS.lat, BIAS.lon, PHOTON_BASE, maxResults, justSelected, debounce, deferredQuery, fetchSuggestions, userIsTyping, isCanceledError]);

  function handleInputChange(e) {
    const v = e.target.value;
    setQuery(v);
    setJustSelected(false); // Réinitialiser le flag si l'utilisateur modifie
    setUserIsTyping(true); // L'utilisateur est en train de taper
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

  // Groupes : alias/favoris en tête, puis Google Places, puis autres (Photon)
  const favorites = useMemo(
    () => items.filter((i) => i.source === 'favorite' || i.source === 'alias'),
    [items]
  );
  const googlePlaces = useMemo(() => items.filter((i) => i.source === 'google_places'), [items]);
  const others = useMemo(
    () =>
      items.filter(
        (i) => i.source !== 'favorite' && i.source !== 'alias' && i.source !== 'google_places'
      ),
    [items]
  );
  const visibleItems = useMemo(
    () => [...favorites, ...googlePlaces, ...others],
    [favorites, googlePlaces, others]
  );

  async function chooseItem(it) {
    // Utiliser directement le label qui est déjà bien formaté
    const fullAddress = it?.label || it?.address || '';

    setQuery(fullAddress);

    // Fermer le menu et vider les suggestions
    setOpen(false);
    setItems([]);
    setHighlight(-1);
    setJustSelected(true);
    setUserIsTyping(false);

    // Réinitialiser le flag après un court délai
    setTimeout(() => {
      setJustSelected(false);
    }, 300);

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
      // Sinon, passer l'item tel quel (Photon, alias, ou Google avec coordonnées déjà présentes)
      onSelect?.(it);
    }
  }

  function onKeyDown(e) {
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
    } else if (e.key === 'Escape') {
      setOpen(false);
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
        onKeyDown={onKeyDown}
        onFocus={() => {
          // Ne pas rouvrir automatiquement au focus
          // Le menu s'ouvrira seulement si l'utilisateur commence à taper
        }}
        onBlur={() => {
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
        className={inputClassName || acStyles.input}
        {...restProps}
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
                zIndex: 10000,
              };
            })()}
          >
          {loading && <div className={acStyles.notice}>Recherche…</div>}

          {!loading && visibleItems.length === 0 && (
            <div className={acStyles.notice}>Aucun résultat</div>
          )}

          {!loading && visibleItems.length > 0 && (
            <>
              {favorites.length > 0 && (
                <>
                  <div className={acStyles.sectionHeader}>Favoris & alias</div>
                  {favorites.map((it, idx) => {
                    const globalIndex = idx;
                    const active = globalIndex === highlight;
                    const line =
                      [it.address, it.postcode, it.city, it.country].filter(Boolean).join(' · ') ||
                      it.label;
                    const key =
                      it.lat != null && it.lon != null
                        ? `${it.lat},${it.lon}-${idx}`
                        : `${it.label || it.address || 'addr'}-fav-${idx}`;
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

              {googlePlaces.length > 0 && (
                <>
                  <div className={acStyles.sectionHeaderGoogle}>Suggestions</div>
                  {googlePlaces.map((it, idx) => {
                    const globalIndex = favorites.length + idx;
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
                    const globalIndex = favorites.length + googlePlaces.length + idx;
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
