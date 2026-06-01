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

/**
 * Imprime le PDF déjà affiché dans un iframe : même effet que le bouton « Imprimer » du lecteur PDF
 * intégré au navigateur (ex. Chrome PDF viewer).
 *
 * @param {HTMLIFrameElement | null | undefined} iframe
 * @returns {boolean} true si {@link Window#print} a été appelé sur le document embarqué
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

/**
 * Imprime un PDF depuis son URL dans une iframe hors écran (pas d’onglet / fenêtre visible).
 * Utile après un `await` (régénération PDF) où `window.open` serait bloqué ou indésirable.
 *
 * La promesse ne se résout `true` que lorsque `print()` a **effectivement** été appelé sur le
 * document embarqué : l’appelant peut ainsi proposer un repli (nouvel onglet / téléchargement)
 * quand le visualiseur PDF n’a pas pu déclencher le dialogue d’impression.
 *
 * L’iframe est positionnée hors écran mais avec une **vraie taille** (et non 1px / opacity:0) :
 * certains navigateurs refusent d’imprimer un contenu effectivement invisible, et le plugin PDF
 * de Chromium a besoin d’être réellement rendu avant que `contentWindow.print()` soit utilisable.
 *
 * @param {string} url
 * @param {{ printSides?: string }} [options]
 * @returns {Promise<boolean>} `true` si l’impression a pu être déclenchée, sinon `false`
 */
export function printPdfFromUrlInHiddenFrame(url, options = {}) {
  const fixedUrl = ensurePdfUrlWorksInDev(url);
  if (!fixedUrl || typeof fixedUrl !== 'string') {
    return Promise.resolve(false);
  }

  let absoluteUrl = fixedUrl;
  try {
    absoluteUrl = fixedUrl.startsWith('/') ? `${window.location.origin}${fixedUrl}` : fixedUrl;
  } catch {
    return Promise.resolve(false);
  }

  void options.printSides;

  return new Promise((resolve) => {
    let iframe = null;
    try {
      iframe = document.createElement('iframe');
      iframe.setAttribute('title', 'Impression PDF');
      /**
       * Hors écran mais rendu (taille proche A4 à 96 dpi) : indispensable pour que le
       * visualiseur PDF intégré accepte de s’imprimer. On évite display:none / 1px / opacity:0.
       */
      Object.assign(iframe.style, {
        position: 'fixed',
        left: '-10000px',
        top: '0',
        width: '794px',
        height: '1123px',
        margin: '0',
        padding: '0',
        border: '0',
        pointerEvents: 'none',
        visibility: 'hidden',
      });

      document.body.appendChild(iframe);
    } catch {
      resolve(false);
      return;
    }

    const localIframe = iframe;
    let settled = false;

    const cleanup = () => {
      window.setTimeout(() => {
        try {
          localIframe.remove();
        } catch {
          /* noop */
        }
      }, 3000);
    };

    /** Résout une seule fois ; programme le retrait de l’iframe. */
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(ok);
    };

    const invokePrint = () => {
      if (settled) return;
      try {
        const w = localIframe.contentWindow;
        if (!w || typeof w.print !== 'function') return;
        w.focus();
        w.print();
        finish(true);
      } catch {
        /* viewer PDF sandbox / navigateur : on laissera le dernier timeout conclure à l’échec */
      }
    };

    localIframe.addEventListener(
      'load',
      () => {
        window.setTimeout(invokePrint, 450);
      },
      { once: true }
    );

    window.setTimeout(invokePrint, 900);
    /** Dernier essai : si rien n’a abouti, on conclut à l’échec pour permettre un repli. */
    window.setTimeout(() => {
      if (settled) return;
      invokePrint();
      if (!settled) finish(false);
    }, 2800);

    localIframe.src = absoluteUrl;
  });
}

/**
 * Télécharge le fichier PDF avec un nom suggéré — équivalent au bouton « Télécharger » du lecteur PDF
 * Chrome (enregistrement fichier / boîte « Enregistrer sous », sans s’appuyer uniquement sur l’attribut HTML download).
 *
 * @param {string} url - URL brute ou déjà passée par {@link ensurePdfUrlWorksInDev}
 * @param {string} suggestedFilename - ex. Facture_EM-2026-04-0002.pdf
 * @returns {Promise<boolean>} true si le déclenchement a réussi
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
 * Repli : lien programmatique (attribut download), si {@link downloadPdfAsFile} échoue.
 * @returns {boolean} true si un clic a été déclenché
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
