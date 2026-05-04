import React, { useMemo } from 'react';
import { formatCurrencyCHF } from '../../../../../services/invoiceService';
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

/** Date pour colonne « Date » : trajet (RIDE) ou ligne perso. (CUSTOM). Prestation au mois → MM.YYYY. */
function lineDetailDateLabel(line) {
  if (!line) return null;
  const kind = String(line?.type ?? line?.line_type ?? '')
    .trim()
    .toUpperCase();
  if (kind !== 'RIDE' && kind !== 'CUSTOM') return null;
  const m = line.line_meta;
  const meta = m && typeof m === 'object' && !Array.isArray(m) ? m : null;
  const dateRaw = meta?.service_date ?? meta?.service_date_iso;
  if (dateRaw == null || String(dateRaw).trim() === '') return null;
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
  const formatted = formatServiceDateFr(String(dateRaw));
  return formatted == null || formatted === '' ? null : formatted;
}

/** Facture clinique S2 : nom du patient sous la description (la date est en colonne). */
function rideLinePatientSubline(line, invoice) {
  if (invoice?.billing_strategy !== 's2_clinic_monthly') return null;
  const m = line?.line_meta;
  const meta = m && typeof m === 'object' ? m : null;
  const name =
    meta?.patient_name != null && String(meta.patient_name).trim() !== ''
      ? String(meta.patient_name).trim()
      : null;
  return name ? `Patient : ${name}` : null;
}

/**
 * Ancien nom de la logique « contexte trajet » (avant colonne Date dédiée).
 * Conservé pour éviter `ReferenceError` si un chunk React Fast Refresh / cache référence encore cet identifiant.
 * Le rendu actuel utilise `rideLineTransportDateLabel` + `rideLinePatientSubline`.
 */
function _rideLineContextSubline(line, invoice) {
  const date = lineDetailDateLabel(line);
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

/** Aligné PDF : trajets aller-retour (même logique que `transport_type` / consolidation A/R). */
function lineIsRoundTrip(line) {
  const m = line?.line_meta;
  if (!m || typeof m !== 'object') return false;
  if (m.is_round_trip_leg === true) return true;
  return String(m.transport_type || '').trim().toUpperCase() === 'A/R';
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
  const cp = line?.line_meta?.custom_prestation;
  if (!cp || typeof cp !== 'object') return null;
  const mode = cp.mode;
  const qty = Number(line.qty);
  const unit = Number(line.unit_price);
  const tu = cp.time_unit;
  const ht = Number(line.line_total);
  if (mode === 'time' && tu && Number.isFinite(qty) && Number.isFinite(unit)) {
    const unitLbl =
      tu === 'min' ? 'min' : tu === 'h' ? 'h' : tu === 'd' ? 'j' : tu === 'mois' ? 'mois' : tu;
    return `${qty} ${unitLbl} × ${unit.toFixed(2)} CHF/${unitLbl} = ${Number.isFinite(ht) ? ht.toFixed(2) : '—'} CHF HT`;
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
        return a.idx - b.idx;
      })
      .map(({ ln }) => ln)
      .filter((ln) => !linePreviewHiddenMergedRoundTrip(ln));
  }, [invoice?.lines]);
  const showTransportDateColumn = lines.some((ln) => lineDetailDateLabel(ln) != null);
  const showRoundTripLegend = lines.some((ln) => lineIsRoundTrip(ln));
  const payer = payerHint(invoice);
  const totalLabel =
    invoice?.billing_strategy === 's2_clinic_monthly' ? 'TOTAL À FACTURER' : 'TOTAL';

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
                <th className={styles.colNum}>HT</th>
                {showVatColumn ? <th className={styles.colNum}>TVA</th> : null}
                {showTtcColumn ? <th className={styles.colNum}>TTC</th> : null}
              </tr>
            </thead>
            <tbody>
              {lines.map((line) => {
                const cn = catalogNetHint(line);
                const sub = customPrestationSubline(line);
                const transportDate = lineDetailDateLabel(line);
                const patientSub = rideLinePatientSubline(line, invoice);
                const partnerRid = line?.line_meta?.round_trip_merge_partner_reservation_id;
                const mergePartner =
                  partnerRid != null ? rideLinesByReservationId.get(Number(partnerRid)) : null;
                const isAr = lineIsRoundTrip(line);
                const partnerDesc =
                  mergePartner &&
                  mergePartner.description != null &&
                  String(mergePartner.description).trim() !== ''
                    ? String(mergePartner.description).trim()
                    : null;
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
                        {safeText(line.description)}
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
            <p className={styles.arLegend}>[A/R] = transport aller-retour</p>
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
          <div className={styles.totalRow}>
            <span>Sous-total HT</span>
            <span>{formatCurrencyCHF(invoice?.subtotal_amount ?? 0)}</span>
          </div>
          {showTaxColumns && vatApplicable ? (
            <div className={styles.totalRow}>
              <span>{vatMeta?.label ? String(vatMeta.label) : 'TVA'}</span>
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
