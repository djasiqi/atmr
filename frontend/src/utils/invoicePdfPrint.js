import { ensurePdfUrlWorksInDev } from './pdfUrlFallback';

/** Recto uniquement (pas de recto-verso). Valeur par défaut pour l’impression facture. */
export const INVOICE_PRINT_SIDES = {
  SIMPLEX: 'simplex',
  DUPLEX_LONG_EDGE: 'duplex_long_edge',
  DUPLEX_SHORT_EDGE: 'duplex_short_edge',
};

function attachPrintListeners(printWindow, options = {}) {
  const printSides = options.printSides ?? INVOICE_PRINT_SIDES.SIMPLEX;
  void printSides;

  let done = false;
  const runPrint = () => {
    if (done) return;
    try {
      printWindow.focus();
      printWindow.print();
      done = true;
    } catch {
      /* plugin PDF / sandbox */
    }
  };

  try {
    printWindow.addEventListener(
      'load',
      () => {
        window.setTimeout(runPrint, 300);
      },
      { once: true }
    );
  } catch {
    /* vieux navigateurs */
  }

  window.setTimeout(runPrint, 600);
  window.setTimeout(() => {
    if (!done) runPrint();
  }, 1800);
}

/**
 * À appeler **de façon synchrone** dans le gestionnaire de clic (avant tout `await`),
 * pour éviter le blocage des pop-ups.
 * @returns {Window | null}
 */
export function openBlankPrintWindow() {
  return window.open('about:blank', '_blank', 'noopener,noreferrer');
}

/**
 * @param {Window} printWindow - Fenêtre créée par {@link openBlankPrintWindow}
 * @param {string} url - URL du PDF
 * @param {{ printSides?: string }} [options] — `printSides` défaut {@link INVOICE_PRINT_SIDES.SIMPLEX} (recto uniquement).
 */
export function navigatePdfWindowAndPrint(printWindow, url, options = {}) {
  const fixedUrl = ensurePdfUrlWorksInDev(url);
  if (!printWindow || !fixedUrl || typeof fixedUrl !== 'string') {
    return false;
  }

  printWindow.location.href = fixedUrl;
  attachPrintListeners(printWindow, options);
  return true;
}

/**
 * Ouvre le PDF dans un nouvel onglet puis lance l’impression (à n’utiliser que depuis un flux **sans** `await` avant l’ouverture).
 * Préférez {@link openBlankPrintWindow} + {@link navigatePdfWindowAndPrint} après des appels API.
 *
 * @param {string} url
 * @param {{ printSides?: string }} [options]
 */
export function openPdfUrlWithPrintDialog(url, options = {}) {
  const fixedUrl = ensurePdfUrlWorksInDev(url);
  if (!fixedUrl || typeof fixedUrl !== 'string') {
    return false;
  }
  const w = window.open(fixedUrl, '_blank', 'noopener,noreferrer');
  if (!w) return false;
  attachPrintListeners(w, options);
  return true;
}
