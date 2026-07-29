import { ensurePdfUrlWorksInDev } from './pdfUrlFallback';

/** Recto uniquement (pas de recto-verso). Valeur par défaut pour l’impression facture. */
export const INVOICE_PRINT_SIDES = {
  SIMPLEX: 'simplex',
  DUPLEX_LONG_EDGE: 'duplex_long_edge',
  DUPLEX_SHORT_EDGE: 'duplex_short_edge',
};

const PRINT_ROOT_CLASS = 'atmr-invoice-print-root';
const PRINT_STYLE_ATTR = 'data-atmr-invoice-print-style';

/** Module pdf.js préchargé (évite le cold start au clic Imprimer). */
let pdfjsLoadPromise = null;

/**
 * Précharge pdf.js + configure le worker (à appeler à l’ouverture du modal).
 * @returns {Promise<typeof import('pdfjs-dist')>}
 */
export function preloadInvoicePdfPrint() {
  if (!pdfjsLoadPromise) {
    pdfjsLoadPromise = import('pdfjs-dist').then((pdfjs) => {
      pdfjs.GlobalWorkerOptions.workerSrc = `${
        process.env.PUBLIC_URL || ''
      }/pdf.worker.min.mjs`;
      return pdfjs;
    });
  }
  return pdfjsLoadPromise;
}

/**
 * CSS d’impression :
 * - `display:none` sur le reste du DOM (pas `visibility:hidden`) → pas de pages blanches
 *   réservées par la SPA ;
 * - `@page { margin: 0 }` → réduit / masque en-têtes et pieds Chrome (date, URL, titre).
 */
function ensurePrintStyles() {
  let style = document.querySelector(`style[${PRINT_STYLE_ATTR}]`);
  if (!style) {
    style = document.createElement('style');
    style.setAttribute(PRINT_STYLE_ATTR, '1');
    document.head.appendChild(style);
  }
  style.textContent = `
    .${PRINT_ROOT_CLASS} {
      position: fixed !important;
      left: -14000px !important;
      top: 0 !important;
      width: 210mm !important;
      margin: 0 !important;
      padding: 0 !important;
      opacity: 0 !important;
      pointer-events: none !important;
      z-index: -1 !important;
      background: #fff !important;
    }
    .${PRINT_ROOT_CLASS} canvas {
      display: block !important;
      width: 210mm !important;
      height: auto !important;
      max-width: 210mm !important;
    }
    @media print {
      @page {
        size: A4;
        margin: 0 !important;
      }
      html, body {
        margin: 0 !important;
        padding: 0 !important;
        background: #fff !important;
        height: auto !important;
        width: 100% !important;
        overflow: visible !important;
      }
      body > *:not(.${PRINT_ROOT_CLASS}) {
        display: none !important;
      }
      .${PRINT_ROOT_CLASS} {
        display: block !important;
        position: static !important;
        left: auto !important;
        top: auto !important;
        width: 210mm !important;
        max-width: 100% !important;
        margin: 0 auto !important;
        padding: 0 !important;
        opacity: 1 !important;
        z-index: auto !important;
        pointer-events: auto !important;
        background: #fff !important;
      }
      .${PRINT_ROOT_CLASS} canvas {
        display: block !important;
        width: 210mm !important;
        max-width: 100% !important;
        height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        page-break-after: always;
        break-after: page;
        page-break-inside: avoid;
        break-inside: avoid;
      }
      .${PRINT_ROOT_CLASS} canvas:last-child {
        page-break-after: auto !important;
        break-after: auto !important;
      }
    }
  `;
}

/**
 * @param {unknown} input
 * @returns {Uint8Array|null}
 */
function normalizePdfBytes(input) {
  if (input == null) return null;
  let view;
  if (input instanceof Uint8Array) {
    view = input;
  } else if (input instanceof ArrayBuffer) {
    view = new Uint8Array(input);
  } else if (ArrayBuffer.isView(input)) {
    view = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  } else {
    return null;
  }
  // buffer détaché (ex. après getDocument) → byteLength 0
  if (view.byteLength < 5) return null;
  const magic = String.fromCharCode(view[0], view[1], view[2], view[3], view[4]);
  if (!magic.startsWith('%PDF')) return null;
  // Copie défensive : pdf.js peut transférer / détacher le buffer d’origine.
  return view.slice();
}

/**
 * @param {Uint8Array} bytes
 * @returns {Promise<HTMLCanvasElement[]>}
 */
async function renderPdfPagesToCanvases(bytes) {
  const pdfjs = await preloadInvoicePdfPrint();
  const { getDocument, GlobalWorkerOptions } = pdfjs;
  // Toujours passer une copie : getDocument peut transférer le ArrayBuffer.
  const data = bytes.slice();

  let pdf;
  try {
    pdf = await getDocument({ data }).promise;
  } catch (workerErr) {
    console.warn('Worker PDF local indisponible, repli CDN:', workerErr);
    GlobalWorkerOptions.workerSrc =
      'https://unpkg.com/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs';
    pdf = await getDocument({ data: bytes.slice() }).promise;
  }

  if (!pdf?.numPages) {
    throw new Error('Le PDF ne contient aucune page.');
  }

  // Largeur A4 ~794 CSS px ; scale 1 = assez net pour l’impression, beaucoup plus rapide que 1.5×.
  const targetWidthPx = 794;

  const canvases = await Promise.all(
    Array.from({ length: pdf.numPages }, (_, i) => i + 1).map(async (pageNum) => {
      const page = await pdf.getPage(pageNum);
      const baseViewport = page.getViewport({ scale: 1 });
      const scale = targetWidthPx / baseViewport.width;
      const viewport = page.getViewport({ scale });
      const canvas = document.createElement('canvas');
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext('2d', { alpha: false });
      if (!ctx) {
        throw new Error(`Rendu page ${pageNum} impossible.`);
      }
      await page.render({ canvasContext: ctx, viewport }).promise;
      return canvas;
    })
  );

  return canvases;
}

/**
 * Imprime des octets PDF déjà en mémoire (chemin rapide).
 *
 * @param {Uint8Array|ArrayBuffer|ArrayBufferView} bytes
 * @returns {Promise<boolean>}
 */
export async function printPdfBytes(bytes) {
  const normalized = normalizePdfBytes(bytes);
  if (!normalized) {
    throw new Error(
      'Le fichier PDF est vide ou invalide. Réessayez ou utilisez « Télécharger ».'
    );
  }

  let root = null;
  const previousTitle = document.title;
  let titleRestored = false;

  const restoreTitle = () => {
    if (titleRestored) return;
    titleRestored = true;
    document.title = previousTitle;
  };

  const cleanupRoot = () => {
    try {
      root?.remove();
    } catch {
      /* noop */
    }
    root = null;
    restoreTitle();
  };

  try {
    ensurePrintStyles();
    const canvases = await renderPdfPagesToCanvases(normalized);

    root = document.createElement('div');
    root.className = PRINT_ROOT_CLASS;
    root.setAttribute('aria-hidden', 'true');
    for (const canvas of canvases) {
      root.appendChild(canvas);
    }
    document.body.appendChild(root);
    document.title = '\u00a0';

    await new Promise((r) => {
      window.requestAnimationFrame(() => r(undefined));
    });

    return await new Promise((resolve) => {
      let done = false;
      const finish = (ok) => {
        if (done) return;
        done = true;
        resolve(ok);
      };

      const onAfterPrint = () => {
        window.setTimeout(() => {
          cleanupRoot();
          finish(true);
        }, 100);
      };
      window.addEventListener('afterprint', onAfterPrint, { once: true });

      try {
        window.print();
      } catch {
        window.removeEventListener('afterprint', onAfterPrint);
        cleanupRoot();
        finish(false);
        return;
      }

      window.setTimeout(() => {
        if (!done) {
          finish(true);
          window.setTimeout(cleanupRoot, 120_000);
        }
      }, 1500);
    });
  } catch (err) {
    console.error('Impression PDF facture échouée:', err);
    cleanupRoot();
    if (err instanceof Error) throw err;
    throw new Error(
      "Impossible d'ouvrir le dialogue d'impression. Utilisez « Télécharger » puis imprimez le fichier."
    );
  }
}

/**
 * Rend chaque page du PDF en canvas (pdf.js), puis ouvre le dialogue Chrome
 * **sur la même page**.
 *
 * @param {string} url - URL blob: du PDF
 * @param {{ printSides?: string, documentTitle?: string }} [options]
 * @returns {Promise<boolean>}
 */
export async function printPdfFromUrlInHiddenFrame(url, options = {}) {
  const fixedUrl = ensurePdfUrlWorksInDev(url);
  if (!fixedUrl || typeof fixedUrl !== 'string') {
    return false;
  }

  let absoluteUrl = fixedUrl;
  try {
    absoluteUrl = fixedUrl.startsWith('/')
      ? `${window.location.origin}${fixedUrl}`
      : fixedUrl;
  } catch {
    return false;
  }

  void options.printSides;
  void options.documentTitle;

  const res = await fetch(absoluteUrl);
  if (!res.ok) {
    throw new Error(`Lecture du PDF impossible (HTTP ${res.status}).`);
  }
  const bytes = new Uint8Array(await res.arrayBuffer());
  return printPdfBytes(bytes);
}

/**
 * @param {HTMLIFrameElement | null | undefined} iframe
 * @returns {boolean}
 */
export function printPdfInEmbeddedIframe(iframe) {
  if (!iframe) return false;
  try {
    const w = iframe.contentWindow;
    if (!w || typeof w.print !== 'function') return false;
    w.focus();
    w.print();
    return true;
  } catch {
    return false;
  }
}

/** @deprecated Ne pas utiliser pour les factures. */
export function openBlankPrintWindow() {
  return null;
}

/** @deprecated Ne pas utiliser pour les factures. */
export function navigatePdfWindowAndPrint() {
  return false;
}

/** @deprecated Ne pas utiliser pour les factures. */
export function openPdfUrlWithPrintDialog() {
  return false;
}

/**
 * @param {string} url
 * @param {string} suggestedFilename
 * @returns {Promise<boolean>}
 */
export async function downloadPdfAsFile(url, suggestedFilename) {
  const fixedUrl = ensurePdfUrlWorksInDev(url);
  if (!fixedUrl || typeof fixedUrl !== 'string') {
    return false;
  }

  let absoluteUrl = fixedUrl;
  try {
    absoluteUrl = fixedUrl.startsWith('/')
      ? `${window.location.origin}${fixedUrl}`
      : fixedUrl;
  } catch {
    return false;
  }

  const name =
    typeof suggestedFilename === 'string' && suggestedFilename.trim()
      ? suggestedFilename.trim()
      : 'document.pdf';

  try {
    const res = await fetch(absoluteUrl, {
      method: 'GET',
      credentials: 'same-origin',
    });
    if (!res.ok) {
      return false;
    }
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    try {
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = name;
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * @returns {boolean}
 */
export function triggerPdfDownloadAnchorFallback(url, suggestedFilename) {
  const fixedUrl = ensurePdfUrlWorksInDev(url);
  if (!fixedUrl || typeof fixedUrl !== 'string') {
    return false;
  }
  const name =
    typeof suggestedFilename === 'string' && suggestedFilename.trim()
      ? suggestedFilename.trim()
      : 'document.pdf';
  try {
    const a = document.createElement('a');
    a.href = fixedUrl;
    a.download = name;
    a.rel = 'noopener noreferrer';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return true;
  } catch {
    return false;
  }
}
