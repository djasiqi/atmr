/**
 * Normalise les URLs PDF pour l’affichage (iframe, impression, nouvel onglet).
 *
 * En développement (CRA) :
 * - Les URLs absolues vers le backend local (`http://127.0.0.1:5000/uploads/...`) sont
 *   réécrites en **chemins relatifs** `/uploads/...` pour passer par `setupProxy.js`.
 *   Sinon le navigateur tente une connexion directe au port 5000 → souvent
 *   « 127.0.0.1 n'autorise pas la connexion » / ERR_CONNECTION_REFUSED si seul le
 *   proxy webpack est utilisé ou si Docker n’expose pas ce port.
 *
 * Fallback historique : sur la page en `localhost`, remplacer `localhost` par `127.0.0.1`
 * dans l’URL (certains cas IPv6 / Docker Windows).
 *
 * @param {string} url - URL du PDF (absolue ou relative)
 * @returns {string} URL utilisable dans le navigateur
 */
export function ensurePdfUrlWorksInDev(url) {
  if (!url || typeof url !== 'string') return url;
  const trimmed = url.trim();
  if (typeof window === 'undefined') return trimmed;

  // Ne jamais réécrire les blob:/data: (sinon blob:http://localhost → 127.0.0.1 = PDF inaccessible).
  if (trimmed.startsWith('blob:') || trimmed.startsWith('data:')) {
    return trimmed;
  }

  /** Déjà relatif : laisser tel quel (déjà servi par le même origine / proxy). */
  if (trimmed.startsWith('/')) {
    return trimmed;
  }

  if (process.env.NODE_ENV === 'development') {
    try {
      const u = new URL(trimmed);
      const loopback = u.hostname === '127.0.0.1' || u.hostname === 'localhost';
      const port = u.port;
      /** Ports backend habituels du repo (setupProxy, docker-compose). */
      const looksLikeLocalApi =
        loopback && (port === '5000' || port === '5100');
      if (looksLikeLocalApi && u.pathname.startsWith('/uploads')) {
        return `${u.pathname}${u.search}`;
      }
    } catch {
      /* URL invalide : retomber sur les règles ci-dessous */
    }
  }

  if (
    window.location.hostname === 'localhost' &&
    trimmed.includes('localhost')
  ) {
    return trimmed.replace(/localhost/g, '127.0.0.1');
  }
  return trimmed;
}

/**
 * True uniquement pour une ligne issue du catalogue `partner_invoices`.
 * Les IDs des deux catalogues ne sont pas commensurables.
 *
 * Le numéro `PARTNER-…` sert de filet si le flag liste est absent.
 *
 * @param {{ is_partner_invoice?: boolean, invoice_number?: string, kind?: string }|null|undefined} invoice
 * @returns {boolean}
 */
export function isPartnerInvoice(invoice) {
  if (!invoice || typeof invoice !== 'object') return false;
  if (
    invoice.is_partner_invoice === true ||
    invoice.kind === 'partner' ||
    invoice.invoice_type === 'partner'
  ) {
    return true;
  }
  const number = String(invoice.invoice_number || '').trim().toUpperCase();
  return number.startsWith('PARTNER-');
}

/**
 * Construit le chemin API (relatif à apiClient.baseURL `/api/v1`) pour le PDF
 * d'une facture (Lot 0 SEC-06). Les anciens liens /uploads/invoices/... ne sont
 * plus publics.
 *
 * Ne pas préfixer `/api/v1` ici : apiClient le ajoute déjà (sinon double préfixe
 * → `/api/v1/api/v1/...` → 404).
 *
 * @param {{ id: number, company_id?: number }} invoice
 * @returns {string|null}
 */
export function buildInvoicePdfApiUrl(invoice) {
  if (!invoice?.id || !invoice?.company_id) return null;
  return `/invoices/companies/${invoice.company_id}/invoices/${invoice.id}/pdf`;
}

/**
 * Construit le chemin API (relatif à apiClient.baseURL `/api/v1`) pour le PDF
 * d'un rappel.
 *
 * @param {{ id: number, company_id?: number }} invoice
 * @param {{ id: number }} reminder
 * @returns {string|null}
 */
export function buildReminderPdfApiUrl(invoice, reminder) {
  if (!invoice?.id || !invoice?.company_id || !reminder?.id) return null;
  return `/invoices/companies/${invoice.company_id}/invoices/${invoice.id}/reminders/${reminder.id}/pdf`;
}

/**
 * Construit le chemin API (relatif à apiClient.baseURL `/api/v1`) pour le PDF
 * d'une facture partenaire (Lot 0 SEC-06).
 *
 * @param {{ id: number, company_id?: number }} partnerInvoice
 * @returns {string|null}
 */
export function buildPartnerInvoicePdfApiUrl(partnerInvoice) {
  if (!partnerInvoice?.id || !partnerInvoice?.company_id) return null;
  return `/invoices/companies/${partnerInvoice.company_id}/partner-invoices/${partnerInvoice.id}/pdf`;
}

/**
 * Route le PDF vers le bon catalogue. Un `partner_invoices.id` ne doit jamais
 * produire `/invoices/{id}/pdf`.
 *
 * @param {{ id?: number, company_id?: number, is_partner_invoice?: boolean }|null|undefined} invoice
 * @param {number|string|null|undefined} [companyId]
 * @returns {string|null}
 */
export function resolveInvoicePdfApiUrl(invoice, companyId) {
  const id = invoice?.id;
  const cid = companyId || invoice?.company_id;
  if (!id || !cid) return null;
  const scoped = {
    id,
    company_id: cid,
    is_partner_invoice: invoice?.is_partner_invoice,
    invoice_number: invoice?.invoice_number,
    kind: invoice?.kind,
    invoice_type: invoice?.invoice_type,
  };
  if (isPartnerInvoice(scoped)) {
    return buildPartnerInvoicePdfApiUrl(scoped);
  }
  return buildInvoicePdfApiUrl(scoped);
}

/**
 * Ajoute un fragment « PDF Open » (#toolbar=0&navpanes=0) pour masquer la barre d’outils
 * du lecteur PDF **intégré à Chromium** (Chrome, Edge, etc.) dans un `<iframe>`.
 *
 * Ne supprime pas le contenu côté extension Adobe Acrobat (overlay sur le canvas) :
 * seul le chrome du viewer navigateur est concerné.
 *
 * @param {string} url - URL complète ou chemin (ex. `/uploads/…pdf?x=1`)
 * @returns {string}
 */
export function appendPdfEmbedChromiumViewerFragment(url) {
  if (!url || typeof url !== 'string') return url;
  const trimmed = url.trim();
  if (!trimmed) return trimmed;

  const fragmentParams = 'toolbar=0&navpanes=0';

  try {
    const base =
      typeof window !== 'undefined' && window.location?.origin
        ? window.location.origin
        : 'http://localhost';
    const u = new URL(trimmed, base);
    const existing = u.hash ? u.hash.replace(/^#/, '') : '';
    u.hash = existing ? `${existing}&${fragmentParams}` : fragmentParams;

    if (trimmed.startsWith('/')) {
      return `${u.pathname}${u.search}${u.hash}`;
    }
    return u.toString();
  } catch {
    if (trimmed.includes('#')) {
      const sep = trimmed.endsWith('#') || trimmed.endsWith('&') ? '' : '&';
      return `${trimmed}${sep}${fragmentParams}`;
    }
    return `${trimmed}#${fragmentParams}`;
  }
}
