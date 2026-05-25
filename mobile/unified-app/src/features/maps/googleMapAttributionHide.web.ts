import {
  LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS,
  LIRIE_GOOGLE_MAP_LOGO_CLIP_PX,
} from "./lirieMapChrome";

const HIDE_SELECTORS = [
  ".gm-style-cc",
  ".gmnoprint",
  ".gm-style-moc",
  "gmp-internal-google-attribution",
  "[class*='google-attribution']",
  "[class*='GoogleAttribution']",
  ".watermark",
  "a[href*='maps.google']",
  "a[href*='google.com/maps']",
  "a[title*='Google']",
  "a[aria-label*='Google']",
  "img[alt='Google']",
  "img[src*='google_logo']",
  "img[src*='google_white']",
  "img[src*='Google']",
];

function hideAttributionNodes(root: ParentNode): void {
  if (typeof document === "undefined") return;
  for (const selector of HIDE_SELECTORS) {
    try {
      root.querySelectorAll(selector).forEach((node) => {
        const el = node as HTMLElement;
        el.style.setProperty("display", "none", "important");
        el.style.setProperty("visibility", "hidden", "important");
        el.style.setProperty("opacity", "0", "important");
        el.style.setProperty("pointer-events", "none", "important");
      });
    } catch {
      // Sélecteur invalide dans un shadow root fermé — ignoré.
    }
  }
}

function buildHostScopedCss(hostSelector: string): string {
  const rules = HIDE_SELECTORS.map(
    (selector) => `${hostSelector} ${selector}`
  ).join(",\n");
  return `${rules} { display: none !important; visibility: hidden !important; opacity: 0 !important; pointer-events: none !important; }`;
}

/**
 * Masque logo / mentions Google dans un hôte carte JS (MutationObserver + CSS scopé).
 */
export function attachGoogleMapAttributionHide(host: HTMLElement): () => void {
  if (typeof document === "undefined") return () => {};

  const hostId =
    host.id ||
    `lirie-map-host-${Math.random().toString(36).slice(2, 9)}`;
  if (!host.id) host.id = hostId;
  host.classList.add("liri-google-map-host");

  const styleId = `lirie-map-attribution-hide-${hostId}`;
  if (!document.getElementById(styleId)) {
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `${LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS}\n${buildHostScopedCss(`#${hostId}`)}`;
    document.head.appendChild(style);
  }

  hideAttributionNodes(host);

  const observer = new MutationObserver(() => {
    hideAttributionNodes(host);
  });
  observer.observe(host, { childList: true, subtree: true });

  const interval = window.setInterval(() => hideAttributionNodes(host), 800);

  return () => {
    observer.disconnect();
    window.clearInterval(interval);
  };
}

export const LIRIE_WEB_MAP_BOTTOM_MASK_PX = LIRIE_GOOGLE_MAP_LOGO_CLIP_PX;
