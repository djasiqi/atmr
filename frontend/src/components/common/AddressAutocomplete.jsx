// frontend/src/components/common/AddressAutocomplete.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
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

      // Construire le label : TOUJOURS une adresse lisible
      let label = '';

      if (props.name) {
        // Si c'est un lieu nommé (ex: "HUG", "Aéroport")
        label = props.name;
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
      const url = `/api/v1/geocode/autocomplete?q=${encodeURIComponent(q)}&lat=${encodeURIComponent(
        BIAS.lat
      )}&lon=${encodeURIComponent(BIAS.lon)}&limit=${encodeURIComponent(maxResults)}`;
      const res = await fetch(url, { signal });
      if (res.ok) {
        const data = await res.json().catch(() => []);
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch {
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
      if (!res.ok) throw new Error('Photon error');
      const data = await res.json();
      const feats = Array.isArray(data?.features) ? data.features : [];
      return normalizePhoton(feats);
    } catch {
      return [];
    }
  }

  // Charger les suggestions (debounce + abort)
  useEffect(() => {
    // Ne pas charger si on vient de sélectionner une adresse
    if (justSelected) {
      return;
    }

    if (!query || (typeof query === 'string' ? query.trim().length : 0) < minChars) {
      setItems([]);
      setOpen(false);
      setLoading(false);
      return;
    }
    debounce(async () => {
      try {
        abortRef.current?.abort();
        const ctl = new AbortController();
        abortRef.current = ctl;
        setLoading(true);

        const queryStr = String(query || '');
        const next = await fetchSuggestions(queryStr, ctl.signal);
        let enriched = Array.isArray(next) ? next : [];

        // Filet de sécu : si l'utilisateur tape "hug" et qu'aucun alias n'est présent,
        // on injecte l'adresse HUG en tête (évite de dépendre à 100% du backend).
        const qn = (query || '').toString().trim().toLowerCase();
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

        setItems(enriched);
        // Ne rouvrir le menu que si l'utilisateur tape activement
        if (userIsTyping && !justSelected) {
          setOpen(true);
        }
        setHighlight(enriched.length ? 0 : -1);
      } catch {
        setItems([]);
        setOpen(false);
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
    if (it.source === 'google_places' && it.place_id && (!it.lat || !it.lon)) {
      try {
        const response = await fetch(
          `/api/v1/geocode/place-details?place_id=${encodeURIComponent(it.place_id)}`
        );

        if (response.ok) {
          const details = await response.json();

          // Enrichir l'item avec les coordonnées GPS
          const enrichedItem = {
            ...it,
            lat: details.lat,
            lon: details.lon,
            address: details.address || fullAddress,
            // Extraire les composants d'adresse si disponibles
            ...(details.address_components && {
              city:
                details.address_components.find((c) => c.types?.includes('locality'))?.long_name ||
                it.city,
              postcode:
                details.address_components.find((c) => c.types?.includes('postal_code'))
                  ?.long_name || it.postcode,
            }),
          };

          onSelect?.(enrichedItem);
          return;
        }
      } catch (error) {
        console.warn('⚠️ Erreur lors de la récupération des coordonnées GPS:', error);
      }
    }

    // Sinon, passer l'item tel quel (Photon, alias, ou Google avec coordonnées déjà présentes)
    onSelect?.(it);
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
