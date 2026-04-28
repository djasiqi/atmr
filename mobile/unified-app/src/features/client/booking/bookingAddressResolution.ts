import {
  AddressAutocompleteSuggestion,
  ClientProfile,
} from "../types";

/** Résultat d’alignement adresse côté serveur (aperçu / soumission de résa). */
export type ResolveFieldForSubmitResult =
  | { status: "ok" }
  | {
    status: "needs_pick_from_list";
    field: "pickup" | "dropoff";
    items: AddressAutocompleteSuggestion[];
  }
  | { status: "unresolved"; localizationOnly?: boolean };

/** L’API / certains fournisseurs envoient la longitude en `lon` ou `lng` — on normalise. */
export function effectiveSuggestionLon(
  s: Pick<AddressAutocompleteSuggestion, "lon" | "lng">
): number | undefined {
  const v = s.lon != null ? s.lon : s.lng;
  if (v == null) return undefined;
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

export function isGeocodedSuggestion(s: AddressAutocompleteSuggestion): boolean {
  return s.lat != null && effectiveSuggestionLon(s) != null;
}

export function uniqueGeocodedByPlaceOrCoord(
  items: AddressAutocompleteSuggestion[]
): AddressAutocompleteSuggestion[] {
  const m = new Map<string, AddressAutocompleteSuggestion>();
  for (const it of items) {
    if (!isGeocodedSuggestion(it)) continue;
    const la = it.lat;
    const lo = effectiveSuggestionLon(it);
    if (la == null || lo == null) continue;
    const k = String(it.place_id ?? "").trim() || `${la.toFixed(5)},${lo.toFixed(5)}`;
    if (!m.has(k)) m.set(k, it);
  }
  return [...m.values()];
}

/** Clé de correspondance : texte du champ et suggestions API. */
export function normAddrKey(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(
      /,?\s*(suisse|switzerland|schweiz|france|germany|deutschland|italia|italy)\s*$/i,
      ""
    )
    .trim();
}

const GEOCODE_ADDR_BOOK_CAP = 120;

export function recordGeocodedInAddressBook(
  book: Map<string, AddressAutocompleteSuggestion>,
  item: AddressAutocompleteSuggestion
) {
  if (!isGeocodedSuggestion(item)) return;
  const a = (item.address ?? item.label).trim();
  const label = (item.label ?? "").trim();
  for (const raw of [a, label].filter((x) => x.length > 0)) {
    const k = normAddrKey(raw);
    if (k.length >= 2) book.set(k, item);
  }
  if (a && label && a !== label) {
    const k = normAddrKey(`${a}|${label}`);
    if (k.length >= 2) book.set(k, item);
  }
  while (book.size > GEOCODE_ADDR_BOOK_CAP) {
    const first = book.keys().next().value;
    if (first == null) break;
    book.delete(first);
  }
}

/**
 * Si l’autocomplete n’a qu’un point géolocalisé (non ambigu) et que le texte saisi
 * correspond à cette entrée, on peut l’appliquer sans tap explicite.
 */
export function tryResolveSingleUnambiguousSuggestion(
  valueTrimmed: string,
  results: AddressAutocompleteSuggestion[]
): AddressAutocompleteSuggestion | null {
  if (valueTrimmed.length < 2 || results.length === 0) return null;
  const book = new Map<string, AddressAutocompleteSuggestion>();
  for (const it of results) {
    recordGeocodedInAddressBook(book, it);
  }
  const geo = uniqueGeocodedByPlaceOrCoord(results);
  if (geo.length !== 1) return null;
  const only = geo[0]!;
  if (!isGeocodedSuggestion(only)) return null;
  const found = findGeocodedInAddressBook(book, valueTrimmed);
  if (!found || !isGeocodedSuggestion(found)) return null;
  const id0 = String(only.place_id ?? "");
  const idF = String(found.place_id ?? "");
  const samePlace = Boolean(id0 && idF && id0 === idF);
  const la0 = only.lat;
  const lo0 = effectiveSuggestionLon(only);
  const laf = found.lat;
  const lof = effectiveSuggestionLon(found);
  const sameCoord =
    la0 != null &&
    laf != null &&
    lo0 != null &&
    lof != null &&
    Math.abs(la0 - laf) < 1e-5 &&
    Math.abs(lo0 - lof) < 1e-5;
  if (!samePlace && !sameCoord) return null;
  return only;
}

export function findGeocodedInAddressBook(
  book: Map<string, AddressAutocompleteSuggestion>,
  value: string
): AddressAutocompleteSuggestion | undefined {
  const v = value.trim();
  if (!v) return undefined;
  const nv = normAddrKey(v);
  if (nv.length < 2) return undefined;
  const direct = book.get(nv);
  if (direct && isGeocodedSuggestion(direct)) return direct;

  const vNoC = v.replace(
    /,?\s*(Suisse|Switzerland|Schweiz|SWITZERLAND|France|Germany)\s*$/i,
    ""
  );
  if (vNoC.trim() !== v) {
    const k2 = normAddrKey(vNoC.trim());
    if (k2.length >= 2) {
      const b = book.get(k2);
      if (b && isGeocodedSuggestion(b)) return b;
    }
  }
  for (const [key, it] of book) {
    if (!isGeocodedSuggestion(it)) continue;
    if (nv.length < 10 || key.length < 10) continue;
    if (nv === key) return it;
    if (nv.length >= 18 && (nv.startsWith(key) || key.startsWith(nv) || nv.includes(key) || key.includes(nv))) {
      return it;
    }
  }
  return undefined;
}

export function collectAlternativeAddressQueries(value: string): string[] {
  const out: string[] = [];
  const push = (s: string) => {
    const t = s.trim();
    if (t.length < 2) return;
    if (!out.includes(t)) out.push(t);
  };
  const v = value.trim();
  if (!v) return out;
  push(v);
  const noC = v.replace(
    /,?\s*(Suisse|Switzerland|Schweiz|SWITZERLAND|France|Germany|Italie|Italy)\s*$/i,
    ""
  ).trim();
  if (noC && noC !== v) push(noC);
  for (let s = v; s.length > 8; ) {
    const ci = s.lastIndexOf(",");
    if (ci <= 4) break;
    s = s.slice(0, ci).trim();
    if (s.length < 2) break;
    push(s);
  }
  return out;
}

export function getDomicileLatLon(
  profile: ClientProfile | null | undefined
): { lat: number; lon: number } | null {
  const d = profile?.domicile;
  if (!d) return null;
  const lat = d.lat;
  const lonD = d.lon != null ? d.lon : d.lng;
  if (typeof lat !== "number" || typeof lonD !== "number" || !Number.isFinite(lat) || !Number.isFinite(lonD)) {
    return null;
  }
  return { lat, lon: lonD };
}
