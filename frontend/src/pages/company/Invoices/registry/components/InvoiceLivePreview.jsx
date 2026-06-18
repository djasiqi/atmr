import React, { useMemo } from 'react';
import { formatCurrencyCHF } from '../../../../../services/invoiceService';
import { formatPatientDisplayNameNomPrenom } from '../../../../../utils/patientDisplayName';
import styles from './InvoiceLivePreview.module.css';

const MONTHS_FR = [
  'janvier',
  'février',
  'mars',
  'avril',
  'mai',
  'juin',
  'juillet',
  'août',
  'septembre',
  'octobre',
  'novembre',
  'décembre',
];

function parseMeta(raw) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    try {
      const p = JSON.parse(raw);
      return typeof p === 'object' && p !== null ? p : null;
    } catch {
      return null;
    }
  }
  if (typeof raw === 'object') return raw;
  return null;
}

function formatIsoDateFr(iso) {
  if (!iso || typeof iso !== 'string') return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
  return `${String(d.getDate()).padStart(2, '0')}.${String(d.getMonth() + 1).padStart(2, '0')}.${d.getFullYear()}`;
}

/** Date `YYYY-MM-DD` ou chaîne ISO complète → affichage JJ.MM.AAAA */
function formatServiceDateFr(raw) {
  if (raw == null || raw === '') return null;
  const s = String(raw).trim();
  const dm = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (dm) return `${dm[3]}.${dm[2]}.${dm[1]}`;
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) {
    return `${String(d.getDate()).padStart(2, '0')}.${String(d.getMonth() + 1).padStart(2, '0')}.${d.getFullYear()}`;
  }
  return s.slice(0, 10);
}

/** Texte affichable pour React (évite « Objects are not valid as a React child »). */
function safeText(value, fallback = '—') {
  if (value == null || value === '') return fallback;
  if (typeof value === 'string') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (typeof value === 'boolean') return value ? 'oui' : 'non';
  return fallback;
}

/** Parité backend `normalize_transport_line_description` : RIDE = motifs Trajet uniquement ; livraison = Livraison uniquement. */
function normalizeTransportLineDescription(text, lineKind) {
  if (text == null || text === '') return text;
  let t = String(text)
    .replace(/&nbsp;/g, ' ')
    .replace(/&#160;/g, ' ')
    .replace(/\u00a0/g, ' ')
    .replace(/\u202f/g, ' ')
    .replace(/[\u200b\u200c\u200d\ufeff\u2060]/g, '');
  const trajetRe =
    /Trajet\s*[:：\uFF1A]\s*(?:Trajet\s*[:：\uFF1A]\s*)*Trajet(?:\s+|(?=[A-Za-zÀ-ÿ0-9])|$)/gi;
  const livDashRe =
    /Livraison\s*[-–—]\s*(?:Livraison\s*[-–—]\s*)*Livraison(?:\s+|(?=[A-Za-zÀ-ÿ0-9])|$)/gi;
  const livColonRe =
    /Livraison\s*[:：\uFF1A]\s*(?:Livraison\s*[:：\uFF1A]\s*)*Livraison(?:\s+|(?=[A-Za-zÀ-ÿ0-9])|$)/gi;
  let prev;
  do {
    prev = t;
    if (lineKind === 'RIDE') {
      t = t.replace(trajetRe, 'Trajet : ');
    } else if (lineKind === 'MATERIAL_DELIVERY') {
      t = t.replace(livDashRe, 'Livraison – ');
      t = t.replace(livColonRe, 'Livraison : ');
    }
  } while (prev !== t);
  return t.trim();
}

/** Parité PDF : préfixe livraison matériel (pas « Trajet : »). */
function lineDescriptionWithPrefix(line) {
  const kind = String(line?.type ?? line?.line_type ?? '')
    .trim()
    .toUpperCase();
  let raw = safeText(line.description);
  if (kind === 'RIDE' && raw && raw !== '—') {
    raw = normalizeTransportLineDescription(raw, 'RIDE');
  }
  if (kind === 'MATERIAL_DELIVERY' && raw && raw !== '—') {
    raw = normalizeTransportLineDescription(raw, 'MATERIAL_DELIVERY');
  }
  if (kind === 'MATERIAL_DELIVERY' && raw && raw !== '—') {
    if (/^livraison\s*:/i.test(String(raw).trim())) return raw;
    return `Livraison : ${raw}`;
  }
  return raw;
}

/** Quantité durée affichée sous la description (évite « 4.00 h » quand c’est un entier). */
function formatTimeBillingQty(qty) {
  if (!Number.isFinite(qty)) return '—';
  const r = Math.round(qty);
  if (Math.abs(qty - r) < 1e-6) return String(r);
  return String(qty);
}

/** Date pour colonne « Date » : trajet, livraison matériel ou ligne perso. (CUSTOM). Prestation au mois → MM.YYYY. */
function lineDetailDateLabel(line, invoice) {
  if (!line) return null;
  const meta0 = parseMeta(line?.line_meta);
  if (meta0?.global_discount_line || meta0?.per_line_discount_line) return null;
  const kind = String(line?.type ?? line?.line_type ?? '')
    .trim()
    .toUpperCase();
  if (
    kind !== 'RIDE' &&
    kind !== 'CUSTOM' &&
    kind !== 'MATERIAL_DELIVERY'
  ) {
    return null;
  }
  const meta = meta0;
  const dateRaw = meta?.service_date ?? meta?.service_date_iso;
  const hasDate = dateRaw != null && String(dateRaw).trim() !== '';

  if (hasDate) {
    const cp = meta?.custom_prestation;
    if (
      kind === 'CUSTOM' &&
      cp &&
      typeof cp === 'object' &&
      cp.mode === 'time' &&
      cp.time_unit === 'mois'
    ) {
      const dm = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(dateRaw).trim());
      if (dm) return `${dm[2]}.${dm[1]}`;
    }
    const dateEndRaw = meta?.service_date_end ?? meta?.service_date_iso_end;
    if (dateEndRaw != null && String(dateEndRaw).trim() !== '') {
      const startLbl = formatServiceDateFr(String(dateRaw));
      const endLbl = formatServiceDateFr(String(dateEndRaw));
      if (startLbl && endLbl && startLbl !== endLbl) {
        return `${startLbl} – ${endLbl}`;
      }
    }
    const formatted = formatServiceDateFr(String(dateRaw));
    if (formatted != null && formatted !== '') return formatted;
  }

  // Libellé « mois année » uniquement pour prestation CUSTOM au mode « mois » (pas pour trajet/livraison).
  const cpMo = meta?.custom_prestation;
  if (
    kind === 'CUSTOM' &&
    cpMo &&
    typeof cpMo === 'object' &&
    cpMo.mode === 'time' &&
    cpMo.time_unit === 'mois'
  ) {
    const y = invoice?.period_year;
    const mo = invoice?.period_month;
    if (Number.isFinite(Number(y)) && Number.isFinite(Number(mo))) {
      return billingPeriodLabel(invoice);
    }
  }
  return null;
}

/** Facture clinique S2 : nom du client transporté sous la description (la date est en colonne). */
function rideLinePatientSubline(line, invoice) {
  if (invoice?.billing_strategy !== 's2_clinic_monthly') return null;
  const m = line?.line_meta;
  const meta = m && typeof m === 'object' ? m : null;
  const name =
    meta?.patient_name != null && String(meta.patient_name).trim() !== ''
      ? formatPatientDisplayNameNomPrenom(String(meta.patient_name).trim())
      : null;
  return name ? `Client : ${name}` : null;
}

/**
 * Ancien nom de la logique « contexte trajet » (avant colonne Date dédiée).
 * Conservé pour éviter `ReferenceError` si un chunk React Fast Refresh / cache référence encore cet identifiant.
 * Le rendu actuel utilise `rideLineTransportDateLabel` + `rideLinePatientSubline`.
 */
function _rideLineContextSubline(line, invoice) {
  const date = lineDetailDateLabel(line, invoice);
  const patientLine = rideLinePatientSubline(line, invoice);
  if (invoice?.billing_strategy === 's2_clinic_monthly') {
    const parts = [];
    if (patientLine) parts.push(patientLine);
    if (date) parts.push(`Date du trajet : ${date}`);
    return parts.length ? parts.join(' · ') : null;
  }
  return date ? `Date du trajet : ${date}` : null;
}

function billingPeriodLabel(inv) {
  const y = inv?.period_year;
  const m = inv?.period_month;
  if (!Number.isFinite(Number(y)) || !Number.isFinite(Number(m))) return '—';
  const mi = Math.max(1, Math.min(12, Number(m))) - 1;
  return `${MONTHS_FR[mi]} ${y}`;
}

/** Libellé du bloc principal (partie gauche) selon la stratégie de facturation. */
function primaryPartyLabel(inv) {
  const bs = inv?.billing_strategy;
  /** Facturation mensuelle établissement (S2) — pas le même vocabulaire que le particulier S1. */
  if (bs === 's2_clinic_monthly') return 'Contact clinique';
  if (bs === 'partner_monthly') return 'Entreprise partenaire';
  return 'Client / Patient';
}

function clientDisplayName(inv) {
  const c = inv?.client;
  if (!c || typeof c !== 'object') return '—';
  if (c.patient_display_name) return String(c.patient_display_name);
  const fn = c.first_name || '';
  const ln = c.last_name || '';
  const n = `${fn} ${ln}`.trim();
  if (n) return n;
  if (c.institution_name) return String(c.institution_name);
  return c.username ? String(c.username) : '—';
}

function payerHint(inv) {
  const bp = inv?.billing_party;
  if (bp?.display_name) return String(bp.display_name);
  const btc = inv?.bill_to_client;
  if (btc) {
    const n = `${btc.first_name || ''} ${btc.last_name || ''}`.trim();
    if (n) return n;
  }
  const bco = inv?.billed_to_company;
  if (bco?.name) return String(bco.name);
  return null;
}

/** Ligne RIDE fusionnée avec une autre pour l’aperçu : masquer la ligne « retour » dupliquée. */
function linePreviewHiddenMergedRoundTrip(line) {
  const m = line?.line_meta;
  if (!m || typeof m !== 'object') return false;
  return m.preview_hide_merged_round_trip === true;
}

/**
 * [A/R] uniquement si deux segments sont fusionnés (payeur / ligne masquée),
 * ou si l’aperçu période a fusionné un vrai A/R (jambe + borne fin → deux créneaux).
 * Ne pas se fier à `is_round_trip_leg` seul (faux positifs aller simple).
 */
/** [A/R] uniquement si fusion métier explicite (partenaire réservation ou ligne masquée). */
function lineIsRoundTrip(line) {
  const m = line?.line_meta;
  if (!m || typeof m !== 'object') return false;
  if (m.period_preview_single_leg) return false;
  if (m.round_trip_merge_partner_reservation_id != null) return true;
  if (m.preview_hide_merged_round_trip === true) return true;
  if (m.billing_unit === 'round_trip') return true;
  const sec = m.round_trip_secondary_reservation_ids;
  if (Array.isArray(sec) && sec.length > 0) return true;
  return m.round_trip_secondary_reservation_id != null;
}

/** Montants HT / TVA / TTC pour une ligne principale A/R (somme des deux segments facturés). */
function mergedRoundTripAmounts(line, partner) {
  const ht0 = Number(line?.line_total);
  const vat0 = Number(line?.vat_amount);
  const ttc0 = Number(line?.total_with_vat);
  if (!partner) {
    return {
      ht: Number.isFinite(ht0) ? ht0 : NaN,
      vat: Number.isFinite(vat0) ? vat0 : NaN,
      ttc: Number.isFinite(ttc0) ? ttc0 : NaN,
    };
  }
  const ht1 = Number(partner.line_total);
  const vat1 = Number(partner.vat_amount);
  const ttc1 = Number(partner.total_with_vat);
  return {
    ht: (Number.isFinite(ht0) ? ht0 : 0) + (Number.isFinite(ht1) ? ht1 : 0),
    vat: (Number.isFinite(vat0) ? vat0 : 0) + (Number.isFinite(vat1) ? vat1 : 0),
    ttc: (Number.isFinite(ttc0) ? ttc0 : 0) + (Number.isFinite(ttc1) ? ttc1 : 0),
  };
}

function catalogNetHint(line) {
  const m = line?.line_meta;
  if (!m || typeof m !== 'object') return null;
  const raw = m.original_line_total;
  if (raw == null || raw === '') return null;
  const cat = parseFloat(String(raw).replace(',', '.'));
  const net = Number(line.line_total);
  if (!Number.isFinite(cat) || !Number.isFinite(net) || Math.abs(cat - net) < 0.005) return null;
  return { cat, net };
}

function customPrestationSubline(line) {
  const meta = parseMeta(line?.line_meta);
  const cp = meta?.custom_prestation;
  if (!cp || typeof cp !== 'object') return null;
  const mode = cp.mode;
  const qty = Number(line.qty);
  const unit = Number(line.unit_price);
  const tu = cp.time_unit;
  const ht = Number(line.line_total);
  if (mode === 'time' && Number.isFinite(qty) && Number.isFinite(unit)) {
    const unitLbl =
      tu === 'min' ? 'min' : tu === 'h' ? 'h' : tu === 'd' ? 'j' : tu === 'mois' ? 'mois' : 'h';
    const dur = formatTimeBillingQty(qty);
    return `${unit.toFixed(2)} × ${dur} ${unitLbl} → ${
      Number.isFinite(ht) ? ht.toFixed(2) : '—'
    } CHF HT`;
  }
  if (mode === 'quantity' && Number.isFinite(qty) && Number.isFinite(unit)) {
    return `${qty} × ${unit.toFixed(2)} CHF = ${Number.isFinite(ht) ? ht.toFixed(2) : '—'} CHF HT`;
  }
  return null;
}

/**
 * Ordre d’affichage : trajets (RIDE, livraison matériel) → autres lignes (custom, frais…) → lignes techniques remise %.
 */
function invoiceLinePreviewSortRank(line) {
  const lm = parseMeta(line?.line_meta);
  if (lm?.global_discount_line || lm?.per_line_discount_line) return 2;
  const t = String(line?.type ?? line?.line_type ?? '').trim().toLowerCase();
  if (t === 'ride' || t === 'material_delivery') return 0;
  return 1;
}

/** Clé ISO `YYYY-MM-DD` pour tri chronologique — lignes sans date en fin de liste (parité PDF S2). */
function lineServiceDateSortKey(line) {
  const meta = parseMeta(line?.line_meta);
  const raw = meta?.service_date ?? meta?.service_date_iso;
  if (raw == null || String(raw).trim() === '') return '9999-12-31';
  const s = String(raw).trim();
  const dm = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (dm) return `${dm[1]}-${dm[2]}-${dm[3]}`;
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) {
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }
  return '9999-12-31';
}

function linePatientSortKey(line) {
  const meta = parseMeta(line?.line_meta);
  return String(meta?.patient_name ?? '')
    .trim()
    .toLowerCase();
}

/**
 * Aperçu facture HTML pour brouillon — données alignées sur `Invoice.to_dict()` / PDF (lignes, totaux, méta).
 */
/**
 * @param {boolean} [companyVatApplicable] — Paramètre entreprise « TVA applicable » : si false,
 *   seules les colonnes Description et HT sont affichées (pas TVA ni TTC).
 */
export default function InvoiceLivePreview({
  invoice,
  className,
  companyVatApplicable = true,
}) {
  const meta = useMemo(() => parseMeta(invoice?.meta), [invoice?.meta]);
  const gd = meta?.global_discount;
  const vatMeta = meta?.vat;
  const vatApplicable =
    Boolean(vatMeta?.applicable) || Number(invoice?.vat_total_amount) > 0.005;
  /** Colonnes taxe (TVA + TTC) : uniquement si l’entreprise a la TVA activée en facturation. */
  const showTaxColumns = companyVatApplicable !== false;
  const showVatColumn = showTaxColumns && vatApplicable;
  /** Pas de colonne TTC redondante lorsque la TVA est nulle (aligné aperçu brouillon / remise globale). */
  const showTtcColumn = showTaxColumns && vatApplicable;

  const rideLinesByReservationId = useMemo(() => {
    const raw = Array.isArray(invoice?.lines)
      ? invoice.lines.filter((ln) => ln != null && typeof ln === 'object')
      : [];
    const m = new Map();
    for (const ln of raw) {
      const rid = ln?.reservation_id;
      if (rid != null && Number.isFinite(Number(rid))) {
        m.set(Number(rid), ln);
      }
    }
    return m;
  }, [invoice?.lines]);

  const lines = useMemo(() => {
    const raw = Array.isArray(invoice?.lines)
      ? invoice.lines.filter((ln) => ln != null && typeof ln === 'object')
      : [];
    return raw
      .map((ln, idx) => ({ ln, idx }))
      .sort((a, b) => {
        const ra = invoiceLinePreviewSortRank(a.ln);
        const rb = invoiceLinePreviewSortRank(b.ln);
        if (ra !== rb) return ra - rb;
        const da = lineServiceDateSortKey(a.ln);
        const db = lineServiceDateSortKey(b.ln);
        if (da !== db) return da.localeCompare(db);
        const pa = linePatientSortKey(a.ln);
        const pb = linePatientSortKey(b.ln);
        if (pa !== pb) return pa.localeCompare(pb);
        return a.idx - b.idx;
      })
      .map(({ ln }) => ln)
      .filter((ln) => !linePreviewHiddenMergedRoundTrip(ln));
  }, [invoice?.lines]);
  const showTransportDateColumn = lines.some(
    (ln) => lineDetailDateLabel(ln, invoice) != null
  );
  const showRoundTripLegend = lines.some((ln) => lineIsRoundTrip(ln));
  const payer = payerHint(invoice);

  const gdBreakdown = useMemo(() => {
    if (!gd || typeof gd !== 'object') return null;
    const gross = Number(gd.subtotal_before_ht);
    const disc = Number(gd.amount_ht);
    const pct = Number(gd.percent);
    if (!Number.isFinite(gross) || !Number.isFinite(disc) || disc <= 0.005) return null;
    return { gross, disc, pct };
  }, [gd]);

  /** Aligné PDF (`pdf.py`) : deux-points séparateur avant les montants du pied. */
  const totalLabel =
    invoice?.billing_strategy === 's2_clinic_monthly' ? 'TOTAL À FACTURER :' : 'TOTAL :';

  return (
    <div className={[styles.root, className].filter(Boolean).join(' ')}>
      <div className={styles.sheet} role="document" aria-label="Aperçu facture">
        <header className={styles.header}>
          <div className={styles.headerGrid}>
            <div>
              <div className={styles.labelMuted}>Facture</div>
              <div className={styles.invoiceNumber}>{invoice?.invoice_number || '—'}</div>
            </div>
            <div className={styles.headerRight}>
              <div>
                <span className={styles.labelMuted}>Émission</span>{' '}
                <span>{formatIsoDateFr(invoice?.issued_at)}</span>
              </div>
              <div>
                <span className={styles.labelMuted}>Échéance</span>{' '}
                <span>{formatIsoDateFr(invoice?.due_date)}</span>
              </div>
              <div>
                <span className={styles.labelMuted}>Période</span>{' '}
                <span>{billingPeriodLabel(invoice)}</span>
              </div>
            </div>
          </div>
          <div className={styles.parties}>
            <div>
              <div className={styles.labelMuted}>{primaryPartyLabel(invoice)}</div>
              <div className={styles.partyName}>{clientDisplayName(invoice)}</div>
            </div>
            {payer ? (
              <div>
                <div className={styles.labelMuted}>Payeur</div>
                <div className={styles.partyName}>{payer}</div>
              </div>
            ) : null}
          </div>
        </header>

        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Détail des prestations</h2>
          <table className={styles.table}>
            <thead>
              <tr>
                {showTransportDateColumn ? (
                  <th className={styles.colDate}>Date</th>
                ) : null}
                <th>Description</th>
                <th className={styles.colNum}>Montant</th>
                {showVatColumn ? <th className={styles.colNum}>TVA</th> : null}
                {showTtcColumn ? <th className={styles.colNum}>TTC</th> : null}
              </tr>
            </thead>
            <tbody>
              {lines.map((line) => {
                const cn = catalogNetHint(line);
                const sub = customPrestationSubline(line);
                const transportDate = lineDetailDateLabel(line, invoice);
                const patientSub = rideLinePatientSubline(line, invoice);
                const lineMeta = parseMeta(line?.line_meta);
                const partnerRid = lineMeta?.round_trip_merge_partner_reservation_id;
                const mergePartner =
                  partnerRid != null ? rideLinesByReservationId.get(Number(partnerRid)) : null;
                const isAr = lineIsRoundTrip(line);
                const partnerKind = mergePartner
                  ? String(mergePartner.type ?? mergePartner.line_type ?? '')
                      .trim()
                      .toUpperCase()
                  : '';
                let partnerDesc =
                  mergePartner &&
                  mergePartner.description != null &&
                  String(mergePartner.description).trim() !== ''
                    ? String(mergePartner.description).trim()
                    : null;
                if (partnerDesc && partnerKind === 'RIDE') {
                  partnerDesc = normalizeTransportLineDescription(
                    partnerDesc,
                    'RIDE',
                  );
                }
                if (partnerDesc && partnerKind === 'MATERIAL_DELIVERY') {
                  partnerDesc = normalizeTransportLineDescription(
                    partnerDesc,
                    'MATERIAL_DELIVERY',
                  );
                }
                const adjNote =
                  line.adjustment_note != null && line.adjustment_note !== ''
                    ? safeText(line.adjustment_note, '')
                    : '';
                const { ht, vat, ttc } = mergedRoundTripAmounts(line, mergePartner);
                return (
                  <tr key={line.id ?? `${line.description}-${line.line_total}`}>
                    {showTransportDateColumn ? (
                      <td className={styles.colDate}>{transportDate ?? '—'}</td>
                    ) : null}
                    <td>
                      {patientSub ? (
                        <div className={styles.lineClinicContext}>{patientSub}</div>
                      ) : null}
                      <div className={styles.lineDesc}>
                        {lineDescriptionWithPrefix(line)}
                        {isAr ? (
                          <span className={styles.lineArTag} title="Transport aller-retour">
                            {' '}
                            [A/R]
                          </span>
                        ) : null}
                      </div>
                      {partnerDesc ? (
                        <div className={styles.lineSub}>
                          Retour : {partnerDesc}
                        </div>
                      ) : null}
                      {sub ? <div className={styles.lineSub}>{sub}</div> : null}
                      {cn ? (
                        <div className={styles.lineSub}>
                          {cn.cat.toFixed(2)} → {cn.net.toFixed(2)} CHF HT
                        </div>
                      ) : null}
                      {adjNote !== '' ? <div className={styles.lineNote}>{adjNote}</div> : null}
                    </td>
                    <td className={styles.colNum}>{Number.isFinite(ht) ? ht.toFixed(2) : '—'}</td>
                    {showVatColumn ? (
                      <td className={styles.colNum}>{Number.isFinite(vat) ? vat.toFixed(2) : '—'}</td>
                    ) : null}
                    {showTtcColumn ? (
                      <td className={styles.colNum}>{Number.isFinite(ttc) ? ttc.toFixed(2) : '—'}</td>
                    ) : null}
                  </tr>
                );
              })}
            </tbody>
          </table>
          {showRoundTripLegend ? (
            <p className={styles.arLegend} role="note">
              [A/R] = transport aller-retour
            </p>
          ) : null}
        </section>

        {gd && typeof gd === 'object' ? (
          <div className={styles.globalDiscHint}>
            Réduction globale enregistrée ({Number(gd.percent) || '—'} %) — montants HT ci-dessus après
            application.
            {gd.note ? ` « ${String(gd.note).slice(0, 200)} »` : ''}
          </div>
        ) : null}

        <footer className={styles.totals}>
          {gdBreakdown ? (
            <>
              <div className={styles.totalRow}>
                <span>Sous-total HT (avant réduction globale)</span>
                <span>{formatCurrencyCHF(gdBreakdown.gross)}</span>
              </div>
              <div className={styles.totalRow}>
                <span>
                  Réduction globale ({Number.isFinite(gdBreakdown.pct) ? gdBreakdown.pct : '—'} %)
                </span>
                <span className={styles.totalDiscountAmt}>
                  − {formatCurrencyCHF(gdBreakdown.disc)}
                </span>
              </div>
            </>
          ) : null}
          <div className={styles.totalRow}>
            <span>{gdBreakdown ? 'Total HT après réduction globale' : 'Sous-total HT'}</span>
            <span>{formatCurrencyCHF(invoice?.subtotal_amount ?? 0)}</span>
          </div>
          {showTaxColumns && vatApplicable ? (
            <div className={styles.totalRow}>
              <span>{vatMeta?.label ? `${String(vatMeta.label)} :` : 'TVA :'}</span>
              <span>{formatCurrencyCHF(invoice?.vat_total_amount ?? 0)}</span>
            </div>
          ) : null}
          <div className={`${styles.totalRow} ${styles.totalGrand}`}>
            <span>{totalLabel}</span>
            <span>{formatCurrencyCHF(invoice?.total_amount ?? 0)}</span>
          </div>
        </footer>

        <p className={styles.previewDisclaimer}>
          Aperçu HTML — le PDF officiel est généré à la finalisation, à l&apos;envoi, au téléchargement ou via
          « Régénérer le PDF ».
        </p>
      </div>
    </div>
  );
}
