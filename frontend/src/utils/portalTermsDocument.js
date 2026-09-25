/**
 * Affichage / conservation des documents contractuels PORTAL (CGU / CGV).
 */

import { downloadProtectedPdfAsFile } from './protectedPdf';

export function portalTermsDocumentLabel(documentType, { variant = 'update' } = {}) {
  if (documentType === 'transport_terms') {
    return variant === 'order'
      ? 'Conditions générales de transport'
      : 'Conditions de réservation et de transport';
  }
  return 'Conditions générales d’utilisation';
}

export function portalTermsDocumentFilename(doc = {}) {
  const type = String(doc.document_type || 'document');
  const version = String(doc.current_version || doc.terms_version || 'unknown').replace(
    /[^\w.-]+/g,
    '_'
  );
  const kind =
    type === 'terms_of_service'
      ? 'CGU_compte_client_prive'
      : type === 'transport_terms'
        ? 'CGV_reservation_transport'
        : String(type).replace(/[^\w.-]+/g, '_');
  return `LIRIE_${kind}_v${version}.pdf`;
}

export function portalTermsPdfApiPath(doc = {}) {
  const type = String(doc.document_type || '').trim();
  if (!type) return null;
  return `/clients/me/portal-terms/${encodeURIComponent(type)}/pdf`;
}

/** Télécharge le PDF officiel (logo LIRIE) via l’API authentifiée. */
export async function downloadPortalTermsDocument(doc) {
  const path = portalTermsPdfApiPath(doc);
  if (!path) return false;
  return downloadProtectedPdfAsFile(path, portalTermsDocumentFilename(doc));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function printPortalTermsDocument(doc, { label } = {}) {
  const body = String(doc?.canonical_body || '');
  if (!body || typeof window === 'undefined') return false;
  const title =
    label ||
    `${portalTermsDocumentLabel(doc.document_type)}${
      doc.current_version || doc.terms_version
        ? ` — version ${doc.current_version || doc.terms_version}`
        : ''
    }`;
  const printWindow = window.open('', '_blank', 'noopener,noreferrer');
  if (!printWindow) return false;
  const logoSrc = `${window.location.origin}/logo-lirie.png`;
  const html = `<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(title)}</title>
  <style>
    @page { margin: 16mm; }
    body {
      font-family: Georgia, "Times New Roman", serif;
      color: #0f172a;
      line-height: 1.55;
      margin: 0;
      padding: 8px 4px 24px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 0 0 14px;
      padding-bottom: 10px;
      border-bottom: 2px solid #0b5cab;
    }
    .brand img {
      height: 32px;
      width: auto;
      max-width: 120px;
      object-fit: contain;
      object-position: left center;
      display: block;
    }
    .brand-meta {
      font-family: Helvetica, Arial, sans-serif;
      font-size: 11px;
      color: #64748b;
    }
    h1 {
      font-size: 1.15rem;
      margin: 0 0 0.35rem;
      font-weight: 650;
    }
    .meta {
      font-family: Helvetica, Arial, sans-serif;
      font-size: 0.8rem;
      color: #64748b;
      margin: 0 0 1rem;
    }
    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: inherit;
      font-size: 0.95rem;
      margin: 0;
    }
    .footer {
      margin-top: 1.5rem;
      padding-top: 0.75rem;
      border-top: 1px solid #cbd5e1;
      font-family: Helvetica, Arial, sans-serif;
      font-size: 0.75rem;
      color: #64748b;
    }
  </style>
</head>
<body>
  <div class="brand">
    <img src="${escapeHtml(logoSrc)}" alt="LIRIE" />
    <div class="brand-meta">Document officiel LIRIE · www.lirie.ch</div>
  </div>
  <h1>${escapeHtml(title)}</h1>
  <p class="meta">Conservez ce document pour vos archives personnelles.</p>
  <pre>${escapeHtml(body)}</pre>
  <p class="footer">Généré depuis le compte client privé LIRIE.</p>
</body>
</html>`;
  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();
  printWindow.focus();
  window.setTimeout(() => {
    try {
      printWindow.print();
    } catch {
      /* ignore */
    }
  }, 50);
  return true;
}
