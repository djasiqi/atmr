import type { FleetMarkerVariant } from "./mapStatusTheme";

/** Canvas SVG avec marge transparente (évite le rognage react-native-maps). */
const VIEWBOX = 56;
const CX = VIEWBOX / 2;
const CY = VIEWBOX / 2;
const INNER_R = 15;

const FLEET_CAR_PATH_24 =
  "M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z";

const MARKER_SHADOW_FILTER = `<defs>
  <filter id="fleetPinShadow" x="-35%" y="-30%" width="170%" height="170%">
    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.12"/>
  </filter>
</defs>`;

function escapeSvgText(input: string): string {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function incidentTrianglePath(cx: number, cy: number, r: number): string {
  return `M ${cx} ${cy - r} L ${cx + r} ${cy + r * 0.9} L ${cx - r} ${cy + r * 0.9} Z`;
}

function alertBadgeSvg(cx: number, cy: number, innerR: number, fill: string): string {
  const br = 5.5;
  const bx = cx + innerR * 0.42;
  const by = cy - innerR * 0.42;
  return `
    <circle cx="${bx}" cy="${by}" r="${br}" fill="${fill}" opacity="0.95"/>
    <text x="${bx}" y="${by + 0.5}" text-anchor="middle" dominant-baseline="central"
      fill="#ffffff" font-size="7" font-weight="800"
      font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">!</text>
  `;
}

function fleetCarGlyphSvg(cx: number, cy: number): string {
  const ox = cx - 12;
  const oy = cy - 12;
  return `<g transform="translate(${ox} ${oy})"><path d="${FLEET_CAR_PATH_24}" fill="#ffffff"/></g>`;
}

function applyMarkerHostStyles(host: HTMLElement, sizePx: number, shadow: string, title?: string): void {
  host.setAttribute("role", "img");
  if (title) {
    host.setAttribute("aria-label", title);
    host.title = title;
  }
  host.style.cssText = [
    "display:block",
    "line-height:0",
    `width:${sizePx}px`,
    `height:${sizePx}px`,
    "pointer-events:auto",
    shadow,
  ].join(";");
}

function mountSvgMarkup(host: HTMLElement, markup: string, sizePx: number): HTMLElement {
  host.innerHTML = markup;
  const svg = host.querySelector("svg");
  if (svg) {
    svg.style.display = "block";
    svg.style.width = `${sizePx}px`;
    svg.style.height = `${sizePx}px`;
  }
  return host;
}

export type FleetMarkerIconOptions = {
  sizePx?: number;
  selected?: boolean;
  variant?: FleetMarkerVariant;
  pulse?: boolean;
};

export function resolveFleetMarkerSizePx(options?: FleetMarkerIconOptions): number {
  return options?.sizePx ?? VIEWBOX;
}

export function fleetMarkerCenterAnchor(_sizePx: number): { anchorLeft: string; anchorTop: string } {
  return { anchorLeft: "-50%", anchorTop: "-50%" };
}

/**
 * Marqueur chauffeur — disque couleur + voiture, glow discret, ombre SVG intégrée.
 * Tout le dessin tient dans le viewBox (pas de rognage Android/iOS).
 */
export function buildFleetVehicleMarkerSvgMarkup(
  fill: string,
  options?: FleetMarkerIconOptions
): string {
  const sizePx = resolveFleetMarkerSizePx(options);
  const variant = options?.variant ?? "vehicle";
  const selectedScale = options?.selected ? 1.12 : 1;
  const drawR = INNER_R * selectedScale;

  const pulseRing = options?.pulse
    ? `<circle cx="${CX}" cy="${CY}" r="${drawR + 3}" fill="none" stroke="${fill}" stroke-width="1.8" stroke-opacity="0.35"/>`
    : "";

  const glowRing = `<circle cx="${CX}" cy="${CY}" r="${drawR + 2}" fill="${fill}" fill-opacity="0.18"/>`;

  if (variant === "incident_triangle") {
    const triR = 14;
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${sizePx}" height="${sizePx}" viewBox="0 0 ${VIEWBOX} ${VIEWBOX}" aria-hidden="true">
      ${MARKER_SHADOW_FILTER}
      ${pulseRing}
      ${glowRing}
      <g filter="url(#fleetPinShadow)">
        <path d="${incidentTrianglePath(CX, CY, triR)}" fill="${fill}"/>
        <text x="${CX}" y="${CY + 4}" text-anchor="middle" dominant-baseline="central"
          fill="#ffffff" font-size="14" font-weight="800"
          font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">!</text>
      </g>
    </svg>`;
  }

  const badge = variant === "vehicle_alert" ? alertBadgeSvg(CX, CY, drawR, fill) : "";

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${sizePx}" height="${sizePx}" viewBox="0 0 ${VIEWBOX} ${VIEWBOX}" aria-hidden="true">
    ${MARKER_SHADOW_FILTER}
    ${pulseRing}
    ${glowRing}
    <g filter="url(#fleetPinShadow)">
      <circle cx="${CX}" cy="${CY}" r="${drawR}" fill="${fill}"/>
      ${fleetCarGlyphSvg(CX, CY)}
    </g>
    ${badge}
  </svg>`;
}

export function makeFleetVehicleMarkerDataUrl(
  fill: string,
  options?: FleetMarkerIconOptions
): string {
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(
    buildFleetVehicleMarkerSvgMarkup(fill, options)
  )}`;
}

type FleetMarkerPressTarget = {
  addListener?: (event: string, handler: () => void) => void;
};

export function bindFleetVehicleMarkerPress(
  content: HTMLElement,
  marker: FleetMarkerPressTarget,
  onPress: () => void
): void {
  const handler = () => onPress();
  content.style.cursor = "pointer";
  content.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    handler();
  });
  marker.addListener?.("click", handler);
  marker.addListener?.("gmp-click", handler);
}

export function createFleetVehicleMarkerElement(
  fill: string,
  options?: FleetMarkerIconOptions & { title?: string }
): HTMLElement | null {
  if (typeof document === "undefined") return null;
  const sizePx = resolveFleetMarkerSizePx(options);
  const root = document.createElement("div");
  applyMarkerHostStyles(
    root,
    sizePx,
    "filter:drop-shadow(0 2px 5px rgba(15,23,42,0.12))",
    options?.title
  );
  return mountSvgMarkup(root, buildFleetVehicleMarkerSvgMarkup(fill, options), sizePx);
}

/** Pastille « 2 », « 3 »… superposée sur l’icône PNG chauffeur. */
export function buildFleetClusterCountBadgeSvgMarkup(
  label: string,
  widthPx: number,
  heightPx: number
): string {
  const safeLabel = escapeSvgText(label);
  const w = widthPx;
  const h = heightPx;
  const cx = w / 2;
  const cy = h / 2;
  const fontSize = safeLabel.length >= 3 ? 9 : 11;
  const stroke = 1.5;
  const shape =
    safeLabel.length <= 1
      ? `<circle cx="${cx}" cy="${cy}" r="${h / 2 - stroke}" fill="#0F766E" stroke="#ffffff" stroke-width="${stroke}"/>`
      : `<rect x="${stroke / 2}" y="${stroke / 2}" width="${w - stroke}" height="${h - stroke}" rx="${h / 2}" fill="#0F766E" stroke="#ffffff" stroke-width="${stroke}"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">
    <defs>
      <filter id="fleetCountShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1" stdDeviation="1.2" flood-color="#0f172a" flood-opacity="0.35"/>
      </filter>
    </defs>
    <g filter="url(#fleetCountShadow)">${shape}</g>
    <text x="${cx}" y="${cy + 0.5}" text-anchor="middle" dominant-baseline="central" fill="#ffffff"
      font-size="${fontSize}" font-weight="800" letter-spacing="-0.02em"
      font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">${safeLabel}</text>
  </svg>`;
}

export function makeFleetClusterCountBadgeDataUrl(
  label: string,
  widthPx: number,
  heightPx: number
): string {
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(
    buildFleetClusterCountBadgeSvgMarkup(label, widthPx, heightPx)
  )}`;
}

export function buildFleetClusterMarkerSvgMarkup(count: number, sizePx = VIEWBOX): string {
  const label = escapeSvgText(String(Math.min(99, count)));
  const cx = sizePx / 2;
  const cy = sizePx / 2;
  const innerR = Math.max(12, Math.round(sizePx * 0.36));
  const fontSize = label.length >= 2 ? 11 : 13;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${sizePx}" height="${sizePx}" viewBox="0 0 ${sizePx} ${sizePx}" aria-hidden="true">
    ${MARKER_SHADOW_FILTER}
    <circle cx="${cx}" cy="${cy}" r="${innerR + 3}" fill="#00796B" fill-opacity="0.18"/>
    <g filter="url(#fleetPinShadow)">
      <circle cx="${cx}" cy="${cy}" r="${innerR}" fill="#00796B"/>
      <text x="${cx}" y="${cy + 0.5}" text-anchor="middle" dominant-baseline="central" fill="#ffffff"
        font-size="${fontSize}" font-weight="600"
        font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">${label}</text>
    </g>
  </svg>`;
}

export function makeFleetClusterMarkerDataUrl(count: number, sizePx = VIEWBOX): string {
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(
    buildFleetClusterMarkerSvgMarkup(count, sizePx)
  )}`;
}

export function createFleetClusterMarkerElement(
  count: number,
  sizePx = VIEWBOX,
  title?: string
): HTMLElement | null {
  if (typeof document === "undefined") return null;
  const root = document.createElement("div");
  applyMarkerHostStyles(
    root,
    sizePx,
    "filter:drop-shadow(0 6px 12px rgba(30,58,138,0.2))",
    title
  );
  return mountSvgMarkup(root, buildFleetClusterMarkerSvgMarkup(count, sizePx), sizePx);
}

/** Pastille mission (P/D) — data URL pour react-native-maps (`image`), sans pin Google natif. */
/** Pastille ETA carte (raster) — évite le pin Google sur les Marker à enfants View. */
export function makeFleetEtaBadgeMarkerDataUrl(label: string): {
  uri: string;
  width: number;
  height: number;
} {
  const trimmed = label.trim().slice(0, 14);
  const text = escapeSvgText(trimmed);
  const height = 26;
  const padX = 10;
  const charW = 6.1;
  const width = Math.max(58, Math.min(112, Math.round(trimmed.length * charW + padX * 2)));
  const rx = height / 2;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" aria-hidden="true">
    <defs>
      <filter id="etaShadow" x="-15%" y="-25%" width="130%" height="150%">
        <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.18"/>
      </filter>
    </defs>
    <rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="${rx}" fill="rgba(0,121,107,0.96)" stroke="rgba(255,255,255,0.66)" stroke-width="1" filter="url(#etaShadow)"/>
    <text x="${width / 2}" y="${height / 2 + 0.5}" text-anchor="middle" dominant-baseline="central"
      fill="#ffffff" font-size="11" font-weight="700" letter-spacing="0.15"
      font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">${text}</text>
  </svg>`;
  return {
    uri: `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`,
    width,
    height,
  };
}

export function makeMissionAnchorMarkerDataUrl(
  fill: string,
  options?: { stroke?: string; radiusPx?: number; selected?: boolean; halo?: boolean }
): string {
  const stroke = options?.stroke ?? "#ffffff";
  const radiusPx = options?.radiusPx ?? 6;
  const selected = options?.selected ?? false;
  const pad = selected ? 7 : 5;
  const sizePx = radiusPx * 2 + pad;
  const cx = sizePx / 2;
  const strokeW = selected ? 3 : 2;
  const haloRing =
    options?.halo !== false
      ? `<circle cx="${cx}" cy="${cx}" r="${radiusPx + 3}" fill="${fill}" fill-opacity="0.18"/>`
      : "";
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${sizePx}" height="${sizePx}" viewBox="0 0 ${sizePx} ${sizePx}" aria-hidden="true">
    ${haloRing}
    <circle cx="${cx}" cy="${cx}" r="${radiusPx + strokeW / 2}" fill="${stroke}" fill-opacity="0.95"/>
    <circle cx="${cx}" cy="${cx}" r="${radiusPx}" fill="${fill}"/>
  </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

export function createMissionAnchorElement(
  fill: string,
  radiusPx: number,
  options?: { stroke?: string; title?: string; selected?: boolean }
): HTMLElement | null {
  if (typeof document === "undefined") return null;
  const stroke = options?.stroke ?? "#ffffff";
  const size = radiusPx * 2 + (options?.selected ? 8 : 6);
  const node = document.createElement("div");
  node.style.cssText = [
    "display:flex",
    "align-items:center",
    "justify-content:center",
    `width:${size}px`,
    `height:${size}px`,
    options?.selected ? "filter:drop-shadow(0 4px 10px rgba(15,23,42,0.28))" : "",
  ].join(";");
  const dot = document.createElement("div");
  dot.style.cssText = [
    `width:${radiusPx * 2}px`,
    `height:${radiusPx * 2}px`,
    `border-radius:${radiusPx}px`,
    `background:${fill}`,
    `border:${options?.selected ? 3 : 2}px solid ${stroke}`,
    "box-sizing:border-box",
  ].join(";");
  if (options?.title) {
    node.title = options.title;
    node.setAttribute("aria-label", options.title);
  }
  node.appendChild(dot);
  return node;
}

export function createMissionDriverContextElement(options: {
  routeLine: string;
  metaLine: string;
  selected?: boolean;
}): HTMLElement | null {
  if (typeof document === "undefined") return null;
  const wrap = document.createElement("div");
  wrap.style.cssText = [
    "display:flex",
    "flex-direction:column",
    "gap:2px",
    "padding:5px 9px",
    "border-radius:12px",
    "background:rgba(255,255,255,0.96)",
    "border:1px solid",
    options.selected ? "rgba(0,121,107,0.55)" : "rgba(148,163,184,0.4)",
    "box-shadow:0 4px 14px rgba(15,23,42,0.12)",
    "max-width:220px",
    "pointer-events:none",
  ].join(";");
  const routeEl = document.createElement("div");
  routeEl.textContent = options.routeLine;
  routeEl.style.cssText =
    "font:700 11px system-ui,sans-serif;color:#0F172A;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
  const metaEl = document.createElement("div");
  metaEl.textContent = options.metaLine;
  metaEl.style.cssText =
    "font:600 10px system-ui,sans-serif;color:#64748B;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
  wrap.appendChild(routeEl);
  wrap.appendChild(metaEl);
  return wrap;
}
