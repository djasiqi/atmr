// frontend/src/components/common/AddressAutocomplete.jsx
import React, { useEffect, useMemo, useRef, useState, useDeferredValue, useTransition } from 'react';
// Using fetch with relative '/api' path to leverage CRA proxy in dev and avoid CORS

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
      const street = props.street || '';
      const housenumber = props.housenumber || '';
      const fullStreetAddress = street && housenumber ? `${street} ${housenumber}` : street || '';

      const postcode = props.postcode || '';
      const city = props.city || props.locality || '';

      // Construire le label : TOUJOURS inclure l'adresse complète avec numéro si disponible
      let label = '';
      const placeName = props.name;

      if (placeName && fullStreetAddress) {
        // Lieu nommé avec adresse complète : "Nom, Rue Numéro, CP, Ville"
        const addressParts = [fullStreetAddress];
        if (postcode) addressParts.push(postcode);
        if (city) addressParts.push(city);
        const addressStr = addressParts.join(', ');
        label = `${placeName}, ${addressStr}`;
      } else if (placeName && street) {
        // Lieu nommé avec rue mais sans numéro : "Nom, Rue, CP, Ville"
        const addressParts = [street];
        if (postcode) addressParts.push(postcode);
        if (city) addressParts.push(city);
        const addressStr = addressParts.join(', ');
        label = `${placeName}, ${addressStr}`;
      } else if (placeName) {
        // Lieu nommé sans adresse : juste le nom (fallback)
        label = placeName;
      } else if (fullStreetAddress && city) {
        // Format complet : "Rue Numéro, CP, Ville"
        label = postcode
          ? `${fullStreetAddress}, ${postcode}, ${city}`
          : `${fullStreetAddress}, ${city}`;
      } else if (fullStreetAddress) {
        // Au moins la rue avec numéro
        label = fullStreetAddress;
      } else if (street) {
        // Juste la rue sans numéro
        label =
          postcode && city
            ? `${street}, ${postcode}, ${city}`
            : city
              ? `${street}, ${city}`
              : street;
      } else if (city) {
        // Au moins la ville
        label = postcode ? `${postcode} ${city}` : city;
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
      };
    });
  }

  // Fetch proxy backend puis fallback Photon direct
  async function fetchSuggestions(queryText, signal) {
    const q = (queryText || '').toString().trim();

    // 1) Proxy backend — mélange alias/favoris + Photon si ton backend le fait
    try {
      // ✅ FIX: Utiliser l'URL complète de l'API en production (api.lirie.ch)
      // ou l'URL relative en développement
      const apiBaseUrl =
        process.env.REACT_APP_API_BASE_URL ||
        process.env.REACT_APP_API_URL ||
        (typeof window !== 'undefined' && window.location.hostname === 'localhost'
          ? '/api/v1'
          : 'https://api.lirie.ch/api/v1');
      
      const url = `${apiBaseUrl}/geocode/autocomplete?q=${encodeURIComponent(q)}&lat=${encodeURIComponent(
        BIAS.lat
      )}&lon=${encodeURIComponent(BIAS.lon)}&limit=${encodeURIComponent(maxResults)}`;
      const res = await fetch(url, { signal });
      if (res.ok) {
        const data = await res.json().catch(() => []);
        if (Array.isArray(data)) {
          if (data.length > 0) {
            console.log(`[AddressAutocomplete] ✅ Backend retourne ${data.length} résultats pour "${q}"`);
            return data;
          } else {
            console.log(`[AddressAutocomplete] ⚠️ Backend retourne une liste vide pour "${q}"`);
          }
      }
      } else {
        console.warn(`[AddressAutocomplete] ⚠️ Erreur backend (${res.status}) pour "${q}"`);
      }
    } catch (error) {
      console.error(`[AddressAutocomplete] ❌ Erreur lors de l'appel backend:`, error);
      // ignore -> fallback
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
      console.error(`[AddressAutocomplete] ❌ Erreur Photon fallback:`, error);
      return [];
    }
  }

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
      } catch {
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
  }, [query, minChars, debounceMs, BIAS.lat, BIAS.lon, PHOTON_BASE, maxResults, justSelected]);

  function handleInputChange(e) {
    const v = e.target.value;
    setQuery(v);
    setJustSelected(false); // Réinitialiser le flag si l'utilisateur modifie
    setUserIsTyping(true); // L'utilisateur est en train de taper
    onChange?.({ target: { name, value: v } });
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
        const response = await fetch(
          `/api/v1/geocode/place-details?place_id=${encodeURIComponent(it.place_id)}`
        );

        if (response.ok) {
          const details = await response.json();

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

          // Construire l'adresse complète (rue + numéro)
          const streetAddress = [streetNumber, route].filter(Boolean).join(' ') || route || '';
          
          // ✅ PRÉSERVER l'adresse originale sélectionnée par l'utilisateur
          // Ne remplacer que si l'adresse enrichie est clairement meilleure (a un numéro manquant dans l'original)
          const originalLabel = fullAddress || it.label || '';
          const originalHasNumber = /\d+/.test(originalLabel);
          const enrichedHasNumber = streetNumber && streetNumber.trim() !== '';
          
          // Construire le label : nom du lieu (si présent) + adresse complète avec numéro
          const placeName = it.main_text || (it.types?.some(t => 
            ['establishment', 'point_of_interest'].includes(t)
          ) ? it.label : null);
          
          // Déterminer quel label utiliser :
          // - Préférer l'original si l'enrichi n'apporte pas de numéro manquant
          // - Utiliser l'enrichi seulement si l'original n'a pas de numéro ET l'enrichi en a un
          let finalLabel = originalLabel;
          
          // Si l'original n'a pas de numéro mais l'enrichi en a un, construire un nouveau label
          if (!originalHasNumber && enrichedHasNumber && streetAddress) {
            if (placeName && streetAddress) {
              // Pour les établissements avec nom, inclure le nom + adresse avec numéro
              const addressParts = [streetAddress];
              if (postcode) addressParts.push(postcode);
              if (city) addressParts.push(city);
              const addressStr = addressParts.join(', ');
              finalLabel = `${placeName}, ${addressStr}`;
            } else if (streetAddress) {
              // Adresse avec numéro mais sans nom d'établissement
              const addressParts = [streetAddress];
              if (postcode) addressParts.push(postcode);
              if (city) addressParts.push(city);
              finalLabel = addressParts.join(', ');
            }
          }
          // Sinon, garder l'adresse originale (celle sélectionnée par l'utilisateur)

          // Enrichir l'item avec les coordonnées GPS et les composants d'adresse
          // ✅ IMPORTANT : Préserver l'adresse originale dans le label pour ne pas la modifier
          const enrichedItem = {
            ...it,
            lat: details.lat,
            lon: details.lon,
            address: streetAddress || it.address || fullAddress,
            street: route || it.street || '',
            street_number: streetNumber || it.street_number || '',
            city: city || it.city || '',
            postcode: postcode || it.postcode || '',
            // ✅ Préserver l'adresse originale sélectionnée par l'utilisateur
            label: finalLabel,
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

  return (
    <div ref={wrapRef} style={{ position: 'relative', width: '100%' }}>
      <input
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
        role="combobox" // ✅ combobox, plus textbox implicite
        aria-autocomplete="list"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        aria-activedescendant={open ? activeId : undefined}
        {...restProps}
        style={{
          width: '100%',
          border: '1px solid #e6e6e6',
          borderRadius: 8,
          padding: '10px 12px',
          outline: 'none',
        }}
      />

      {open && (
        <div
          id={listboxId}
          role="listbox"
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            zIndex: 1000,
            background: '#fff',
            border: '1px solid #e6e6e6',
            borderTop: 'none',
            borderRadius: '0 0 8px 8px',
            maxHeight: 280,
            overflowY: 'auto',
            boxShadow: '0 8px 24px rgba(0,0,0,0.08)',
          }}
        >
          {loading && <div style={{ padding: '10px 12px', color: '#6b7280' }}>Recherche…</div>}

          {!loading && visibleItems.length === 0 && (
            <div style={{ padding: '10px 12px', color: '#6b7280' }}>Aucun résultat</div>
          )}

          {!loading && visibleItems.length > 0 && (
            <>
              {favorites.length > 0 && (
                <>
                  <div
                    style={{
                      padding: '6px 12px',
                      fontSize: 11,
                      textTransform: 'uppercase',
                      color: '#6b7280',
                    }}
                  >
                    Favoris & alias
                  </div>
                  {favorites.map((it, idx) => {
                    const globalIndex = idx;
                    const active = globalIndex === highlight;
                    const line =
                      [it.address, it.postcode, it.city, it.country].filter(Boolean).join(' · ') ||
                      it.label;
                    // Clé unique : coordonnées + index pour éviter les doublons
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
                        onMouseDown={(e) => {
                          e.preventDefault();
                          chooseItem(it);
                        }}
                        onMouseEnter={() => setHighlight(globalIndex)}
                        style={{
                          padding: '10px 12px',
                          cursor: 'pointer',
                          background: active ? '#f5f7fb' : '#fff',
                        }}
                      >
                        <div style={{ fontWeight: 600, fontSize: 14 }}>
                          {it.label || it.address}
                        </div>
                        {line && (
                          <div
                            style={{
                              color: '#666',
                              fontSize: 12,
                              marginTop: 2,
                            }}
                          >
                            {line}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </>
              )}

              {googlePlaces.length > 0 && (
                <>
                  <div
                    style={{
                      padding: '6px 12px',
                      fontSize: 11,
                      textTransform: 'uppercase',
                      color: '#4285F4',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    🌍 Google Places
                  </div>
                  {googlePlaces.map((it, idx) => {
                    const globalIndex = favorites.length + idx;
                    const active = globalIndex === highlight;
                    return (
                      <div
                        id={`${name || 'address'}-ac-option-${globalIndex}`}
                        key={it.place_id || `google-${idx}`}
                        role="option"
                        aria-selected={active}
                        onMouseDown={(e) => {
                          e.preventDefault();
                          chooseItem(it);
                        }}
                        onMouseEnter={() => setHighlight(globalIndex)}
                        style={{
                          padding: '10px 12px',
                          cursor: 'pointer',
                          background: active ? '#f5f7fb' : '#fff',
                          borderLeft: '3px solid #4285F4',
                        }}
                      >
                        <div style={{ fontWeight: 600, fontSize: 14 }}>
                          {it.main_text || it.label}
                        </div>
                        {it.secondary_text && (
                          <div
                            style={{
                              color: '#666',
                              fontSize: 12,
                              marginTop: 2,
                            }}
                          >
                            {it.secondary_text}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </>
              )}

              {others.length > 0 && (
                <>
                  <div
                    style={{
                      padding: '6px 12px',
                      fontSize: 11,
                      textTransform: 'uppercase',
                      color: '#6b7280',
                    }}
                  >
                    Autres résultats
                  </div>
                  {others.map((it, idx) => {
                    const globalIndex = favorites.length + googlePlaces.length + idx;
                    const active = globalIndex === highlight;
                    const line =
                      [it.address, it.postcode, it.city, it.country].filter(Boolean).join(' · ') ||
                      it.label;
                    // Clé unique : coordonnées + index pour éviter les doublons
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
                        onMouseDown={(e) => {
                          e.preventDefault();
                          chooseItem(it);
                        }}
                        onMouseEnter={() => setHighlight(globalIndex)}
                        style={{
                          padding: '10px 12px',
                          cursor: 'pointer',
                          background: active ? '#f5f7fb' : '#fff',
                        }}
                      >
                        <div style={{ fontWeight: 600, fontSize: 14 }}>
                          {it.label || it.address}
                        </div>
                        {line && (
                          <div
                            style={{
                              color: '#666',
                              fontSize: 12,
                              marginTop: 2,
                            }}
                          >
                            {line}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
