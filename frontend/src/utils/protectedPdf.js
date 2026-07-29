import apiClient from './apiClient';
import { ensurePdfUrlWorksInDev } from './pdfUrlFallback';

/**
 * @param {unknown} raw
 * @param {string} [filename]
 * @returns {Blob|File}
 */
function toPdfBlob(raw, filename) {
  const base =
    raw instanceof Blob
      ? raw.type
        ? raw
        : new Blob([raw], { type: 'application/pdf' })
      : new Blob([raw], { type: 'application/pdf' });
  const name =
    typeof filename === 'string' && filename.trim() ? filename.trim() : '';
  if (!name || typeof File === 'undefined') return base;
  try {
    return new File([base], name, {
      type: base.type || 'application/pdf',
    });
  } catch {
    return base;
  }
}

/**
 * Télécharge un PDF via une route API JWT et retourne les octets bruts.
 * Plus rapide pour l’impression (évite blob URL + re-fetch).
 *
 * @param {string} apiPath
 * @returns {Promise<Uint8Array|null>}
 */
export async function fetchProtectedPdfBytes(apiPath) {
  if (!apiPath || typeof apiPath !== 'string') return null;
  const response = await apiClient.get(apiPath, { responseType: 'blob' });
  const raw = response?.data;
  if (!raw) return null;
  const blob = toPdfBlob(raw);
  const type = String(blob.type || '').toLowerCase();
  if (type.includes('json') || type.includes('html') || type.includes('text/')) {
    return null;
  }
  const bytes = new Uint8Array(await blob.arrayBuffer());
  if (bytes.length < 5) return null;
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]);
  if (!magic.startsWith('%PDF')) return null;
  return bytes;
}

/**
 * Télécharge un PDF via une route API JWT (Lot 0 SEC-06) et retourne une object URL.
 * Les anciens liens publics `/uploads/invoices/...` renvoient 404 : utiliser ce helper.
 *
 * L'appelant doit révoquer l'URL avec `URL.revokeObjectURL` quand elle n'est plus utile
 * (sauf ouverture dans un nouvel onglet où le navigateur garde la référence).
 *
 * @param {string} apiPath - Chemin relatif à apiClient.baseURL (ex. `/invoices/companies/1/invoices/2/pdf`)
 * @param {{ filename?: string }} [options]
 * @returns {Promise<string|null>}
 */
export async function fetchProtectedPdfObjectUrl(apiPath, options = {}) {
  if (!apiPath || typeof apiPath !== 'string') return null;
  const bytes = await fetchProtectedPdfBytes(apiPath);
  if (!bytes) return null;
  const blob = toPdfBlob(new Blob([bytes], { type: 'application/pdf' }), options.filename);
  return URL.createObjectURL(blob);
}

/**
 * Ouvre un PDF protégé dans un nouvel onglet (blob URL via API).
 * Repli optionnel sur une URL publique (logos / fichiers encore servis).
 *
 * @param {string|null|undefined} apiPath
 * @param {string|null|undefined} [fallbackUrl]
 * @param {{ filename?: string }} [options]
 * @returns {Promise<boolean>}
 */
export async function openProtectedPdfInNewTab(apiPath, fallbackUrl, options = {}) {
  if (apiPath) {
    try {
      const blobUrl = await fetchProtectedPdfObjectUrl(apiPath, options);
      if (blobUrl) {
        window.open(blobUrl, '_blank', 'noopener,noreferrer');
        return true;
      }
    } catch (err) {
      console.error('Ouverture PDF protégé échouée:', err);
      return false;
    }
  }
  const target = fallbackUrl ? ensurePdfUrlWorksInDev(fallbackUrl) : null;
  if (!target) return false;
  window.open(target, '_blank', 'noopener,noreferrer');
  return true;
}

/**
 * Télécharge un PDF protégé (API JWT) avec un nom de fichier suggéré.
 *
 * @param {string} apiPath
 * @param {string} suggestedFilename
 * @returns {Promise<boolean>}
 */
export async function downloadProtectedPdfAsFile(apiPath, suggestedFilename) {
  if (!apiPath || typeof apiPath !== 'string') return false;
  const name =
    typeof suggestedFilename === 'string' && suggestedFilename.trim()
      ? suggestedFilename.trim()
      : 'document.pdf';
  try {
    const response = await apiClient.get(apiPath, { responseType: 'blob' });
    const raw = response?.data;
    if (!raw) return false;
    const blob = toPdfBlob(raw, name);
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
  } catch (err) {
    console.error('Téléchargement PDF protégé échoué:', err);
    return false;
  }
}
