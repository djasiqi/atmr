/**
 * Brouillon de préparation (mémoire uniquement).
 * Même forme que la facture éditée par InvoiceLivePreview, sans invoice_id ni numéro.
 */

export const PREPARATION_INVOICE_ID = 'preparation';

const round2 = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.round(n * 100) / 100;
};

const isCustomLine = (line) =>
  Boolean(line?.preparation_custom) ||
  String(line?.type || line?.line_type || '').toLowerCase() === 'custom';

export const isPreparationInvoice = (invoice) =>
  Boolean(invoice?.preparation) || invoice?.id === PREPARATION_INVOICE_ID;

const serviceDateIso = (raw) => {
  const match = /^(\d{4}-\d{2}-\d{2})/.exec(String(raw || '').trim());
  return match ? match[1] : null;
};

const pushId = (ids, value) => {
  if (value == null || value === '') return;
  const text = String(value);
  if (!ids.includes(text)) ids.push(text);
};

export const previewLineMatchIds = (row) => {
  const ids = [];
  if (!row || typeof row !== 'object') return ids;
  pushId(ids, row.booking_id);
  pushId(ids, row.preview_row_id);
  pushId(ids, row.primary_booking_id);
  pushId(ids, row.round_trip_partner_booking_id);
  if (Array.isArray(row.booking_ids)) row.booking_ids.forEach((id) => pushId(ids, id));
  return ids;
};

const legAmountsForRow = (row, matchIds) => {
  const bookingIds = Array.isArray(row.booking_ids) ? row.booking_ids : [];
  const primary = Number(row.round_trip_primary_amount_ht);
  const partner = Number(row.round_trip_partner_amount_ht);
  if (bookingIds.length > 1 && Number.isFinite(primary) && Number.isFinite(partner)) {
    return [
      { match_id: String(bookingIds[0]), amount: round2(primary) },
      {
        match_id: String(bookingIds[bookingIds.length - 1]),
        amount: round2(partner),
      },
    ];
  }
  if (matchIds.length === 1) {
    return [{ match_id: matchIds[0], amount: round2(row.amount_ht) }];
  }
  return [];
};

const scaleLegs = (legs, nextTotal) => {
  const base = legs.reduce((sum, leg) => sum + (Number(leg.amount) || 0), 0);
  const target = round2(nextTotal);
  if (!legs.length) return [];
  if (base <= 0) {
    return legs.map((leg, index) => ({
      ...leg,
      amount: index === 0 ? target : 0,
    }));
  }
  const scaled = legs.map((leg) => ({
    ...leg,
    amount: round2((Number(leg.amount) / base) * target),
  }));
  const drift = round2(target - scaled.reduce((sum, leg) => sum + leg.amount, 0));
  scaled[0] = { ...scaled[0], amount: round2(scaled[0].amount + drift) };
  return scaled;
};

const strategyFor = (payerType) => {
  if (payerType === 'clinic') return 's2_clinic_monthly';
  if (payerType === 'partner') return 'partner_monthly';
  return 's1_patient';
};

const clientFor = (payerType, recipientName) => {
  const name = String(recipientName || '').trim();
  if (payerType === 'patient') {
    return { patient_display_name: name || 'Patient' };
  }
  return {
    institution_name: name || (payerType === 'partner' ? 'Partenaire' : 'Institution'),
    is_institution: true,
  };
};

const isMergedRoundTripRow = (row) => {
  const ids = Array.isArray(row?.booking_ids) ? row.booking_ids.filter((id) => id != null) : [];
  if (ids.length > 1) return true;
  return (
    row?.round_trip_partner_booking_id != null &&
    String(row.round_trip_partner_booking_id) !== String(row.booking_id)
  );
};

const singleLegRow = (row, bookingId, amount, description, scheduledAt) => ({
  ...row,
  booking_id: bookingId,
  booking_ids: [bookingId],
  preview_row_id: bookingId,
  amount_ht: amount,
  description,
  scheduled_at: scheduledAt || row.scheduled_at,
  is_round_trip_leg: false,
  round_trip_partner_booking_id: null,
  round_trip_primary_amount_ht: null,
  round_trip_partner_amount_ht: null,
  round_trip_partner_description: '',
});

const lineFromPreviewRow = (row, index) => {
  const matchIds = previewLineMatchIds(row);
  const amount = round2(row.amount_ht);
  const legs = legAmountsForRow(row, matchIds);
  const date = serviceDateIso(row.scheduled_at);
  const patient =
    row.patient_name != null && String(row.patient_name).trim() !== ''
      ? String(row.patient_name).trim()
      : '';
  const source = String(row.source_type || 'ride').toLowerCase();
  const type = source === 'service' || source === 'material_delivery' ? 'material_delivery' : 'ride';
  return {
    id: `prep-line-${matchIds[0] || index}`,
    type,
    line_type: type,
    description: String(row.description || '').trim() || 'Prestation',
    line_total: amount,
    preparation_base: amount,
    adjustment_note: null,
    reservation_id: row.booking_id ?? null,
    match_ids: matchIds,
    leg_amounts: legs,
    qty: 1,
    unit_price: amount,
    pickup_label: row.pickup_label || null,
    client_name: patient || row.client_name || null,
    departure: row.departure || null,
    arrival: row.arrival || null,
    pickup_time_source: row.pickup_time_source || null,
    scheduled_pickup_at: row.scheduled_pickup_at || null,
    boarded_at: row.boarded_at || null,
    line_meta: {
      ...(date ? { service_date: date } : {}),
      ...(patient ? { patient_name: patient } : {}),
      ...(row.pickup_label ? { pickup_label: row.pickup_label } : {}),
      ...(row.pickup_time_source ? { pickup_time_source: row.pickup_time_source } : {}),
      ...(row.scheduled_pickup_at ? { scheduled_pickup_at: row.scheduled_pickup_at } : {}),
      ...(row.boarded_at ? { boarded_at: row.boarded_at } : {}),
    },
  };
};

const sumLines = (lines) => round2(lines.reduce((sum, line) => sum + (Number(line.line_total) || 0), 0));

const discountFactor = (invoice) => {
  const global = Number(invoice?.meta?.global_discount?.percent);
  if (Number.isFinite(global) && global > 0) return { kind: 'global', percent: global };
  return { kind: 'none', percent: 0 };
};

const recomputeTotals = (invoice) => {
  const factor = discountFactor(invoice);
  const lines = (invoice.lines || []).map((line) => {
    if (isCustomLine(line)) return line;
    const base = round2(line.preparation_base ?? line.line_total);
    const linePercent =
      factor.kind === 'global'
        ? factor.percent
        : Number(line.preparation_line_percent) || 0;
    const net = linePercent > 0 ? round2(base * (1 - linePercent / 100)) : base;
    const meta = { ...(line.line_meta || {}) };
    if (linePercent > 0) meta.original_line_total = base;
    else delete meta.original_line_total;
    return {
      ...line,
      preparation_base: base,
      line_total: net,
      line_meta: meta,
    };
  });
  const gross = round2(
    lines.reduce((sum, line) => {
      if (isCustomLine(line)) return sum + (Number(line.line_total) || 0);
      return sum + (Number(line.preparation_base) || 0);
    }, 0)
  );
  const subtotal = sumLines(lines);
  const meta = { ...(invoice.meta || {}) };
  if (factor.kind === 'global') {
    meta.global_discount = {
      percent: factor.percent,
      amount_ht: round2(Math.max(0, gross - subtotal)),
      subtotal_before_ht: gross,
      note: invoice.meta?.global_discount?.note || null,
    };
    delete meta.per_line_discounts;
  }
  const basisHt = Number(invoice?.preparation_vat_basis?.ht);
  const basisVat = Number(invoice?.preparation_vat_basis?.vat);
  const basisTotal = Number(invoice?.preparation_vat_basis?.total);
  const sameBasis = basisHt > 0 && Math.abs(subtotal - basisHt) < 0.001;
  const vatTotal =
    basisHt > 0 && basisVat > 0 && subtotal > 0
      ? sameBasis
        ? round2(basisVat)
        : round2(basisVat * (subtotal / basisHt))
      : 0;
  let remainingVat = vatTotal;
  const taxedLines = lines.map((line, index) => {
    const isLast = index === lines.length - 1;
    const share =
      vatTotal <= 0
        ? 0
        : isLast
          ? round2(remainingVat)
          : round2((vatTotal * (Number(line.line_total) || 0)) / subtotal);
    if (!isLast) remainingVat = round2(remainingVat - share);
    return {
      ...line,
      vat_amount: share,
      total_with_vat: round2((Number(line.line_total) || 0) + share),
    };
  });
  if (vatTotal > 0) {
    meta.vat = { ...(meta.vat || {}), applicable: true, label: meta.vat?.label || 'TVA' };
  } else {
    delete meta.vat;
  }
  return {
    ...invoice,
    lines: taxedLines,
    meta,
    subtotal_amount: subtotal,
    vat_total_amount: vatTotal,
    total_amount:
      sameBasis && basisTotal > 0 ? round2(basisTotal) : round2(subtotal + vatTotal),
  };
};

/**
 * @param {{
 *   payerType: 'patient'|'clinic'|'partner'|string,
 *   periodYear: number,
 *   periodMonth: number,
 *   recipientName?: string,
 *   previewLines?: object[],
 * }} input
 */
export function buildPreparationInvoice(input) {
  const payerType = input?.payerType || 'patient';
  const sourceRows = (Array.isArray(input?.previewLines) ? input.previewLines : []).filter(
    (row) => row && row.already_invoiced !== true
  );
  const expandedRows = [];
  sourceRows.forEach((row) => {
    if (!isMergedRoundTripRow(row)) {
      expandedRows.push(row);
      return;
    }
    const ids = Array.isArray(row.booking_ids) && row.booking_ids.length > 1
      ? row.booking_ids
      : [row.booking_id, row.round_trip_partner_booking_id];
    const outboundId = ids[0];
    const returnId = ids[ids.length - 1];
    const outboundAmount = Number.isFinite(Number(row.round_trip_primary_amount_ht))
      ? round2(row.round_trip_primary_amount_ht)
      : round2(row.amount_ht);
    const returnAmount = Number.isFinite(Number(row.round_trip_partner_amount_ht))
      ? round2(row.round_trip_partner_amount_ht)
      : round2(Number(row.amount_ht) - outboundAmount);
    expandedRows.push(
      singleLegRow(row, outboundId, outboundAmount, row.description, row.scheduled_at)
    );
    expandedRows.push(
      singleLegRow(
        row,
        returnId,
        returnAmount,
        row.round_trip_partner_description || row.description,
        row.round_trip_partner_scheduled_at || row.scheduled_at
      )
    );
  });
  const lines = expandedRows.map(lineFromPreviewRow);
  const baseline = [];
  lines.forEach((line) => line.match_ids.forEach((id) => pushId(baseline, id)));
  const basisHt = Number(input?.estimatedSubtotalHt);
  const basisVat = Number(input?.estimatedVatTotal);
  const invoice = {
    preparation: true,
    id: PREPARATION_INVOICE_ID,
    invoice_number: '',
    status: 'draft',
    period_year: Number(input?.periodYear) || null,
    period_month: Number(input?.periodMonth) || null,
    billing_strategy: strategyFor(payerType),
    catalog_type: payerType === 'partner' ? 'partner' : 'standard',
    is_partner_invoice: payerType === 'partner',
    kind: payerType === 'partner' ? 'partner' : 'standard',
    editor_header: payerType === 'partner',
    line_time_mode: 'none',
    client: clientFor(payerType, input?.recipientName),
    recipient_name: String(input?.recipientName || '').trim(),
    lines,
    baseline_match_ids: baseline,
    meta: {},
    preparation_seq: lines.length,
    ...(basisHt > 0 && basisVat > 0
      ? {
          preparation_vat_basis: {
            ht: basisHt,
            vat: basisVat,
            total: Number(input?.estimatedTotalWithVat) || 0,
          },
        }
      : {}),
  };
  return recomputeTotals(invoice);
}

const clone = (invoice) => recomputeTotals({
  ...invoice,
  lines: (invoice.lines || []).map((line) => ({
    ...line,
    line_meta: { ...(line.line_meta || {}) },
    match_ids: [...(line.match_ids || [])],
    leg_amounts: (line.leg_amounts || []).map((leg) => ({ ...leg })),
  })),
  meta: { ...(invoice.meta || {}) },
});

export function updatePreparationLine(invoice, lineId, body) {
  const next = clone(invoice);
  next.lines = next.lines.map((line) => {
    if (String(line.id) !== String(lineId)) return line;
    const edited = { ...line };
    if (body.description != null) edited.description = String(body.description);
    if (body.adjustment_note !== undefined) {
      edited.adjustment_note = body.adjustment_note;
    }
    if (body.line_total != null && Number.isFinite(Number(body.line_total))) {
      const typed = round2(body.line_total);
      const currentNet = round2(line.line_total);
      const base = round2(line.preparation_base ?? line.line_total);
      const percent = base > 0 ? round2((1 - currentNet / base) * 100) : 0;
      const nextBase =
        percent > 0 && percent < 100 ? round2(typed / (1 - percent / 100)) : typed;
      edited.preparation_base = nextBase;
      edited.leg_amounts = scaleLegs(line.leg_amounts || [], nextBase);
    }
    return edited;
  });
  return recomputeTotals(next);
}

export function removePreparationLine(invoice, lineId) {
  const next = clone(invoice);
  next.lines = next.lines.filter((line) => String(line.id) !== String(lineId));
  return recomputeTotals(next);
}

export function addPreparationCustomLine(invoice, body) {
  const next = clone(invoice);
  const seq = Number(next.preparation_seq || next.lines.length) + 1;
  const total = round2(body.line_total);
  const qty = Number(body.qty) > 0 ? Number(body.qty) : 1;
  const date = serviceDateIso(body.service_date_iso);
  const mode = body.custom_mode === 'quantity' ? 'quantity' : 'time';
  next.preparation_seq = seq;
  next.lines = [
    ...next.lines,
    {
      id: `prep-custom-${seq}`,
      type: 'custom',
      line_type: 'custom',
      preparation_custom: true,
      description: String(body.description || '').trim() || 'Ligne supplémentaire',
      line_total: total,
      preparation_base: total,
      qty,
      unit_price: round2(total / qty),
      adjustment_note: null,
      match_ids: [],
      leg_amounts: [],
      line_meta: {
        ...(date ? { service_date: date } : {}),
        custom_prestation: {
          mode,
          ...(mode === 'time' && body.time_unit ? { time_unit: body.time_unit } : {}),
        },
        ...(total < 0 ? { manual_discount: true } : {}),
      },
    },
  ];
  return recomputeTotals(next);
}

export function applyPreparationGlobalDiscount(invoice, payload) {
  const percent = Number(payload?.global_discount_percent);
  const next = clone(invoice);
  next.lines = next.lines.map((line) => {
    if (isCustomLine(line)) return line;
    return { ...line, preparation_line_percent: 0 };
  });
  next.meta = {
    ...(next.meta || {}),
    global_discount: {
      percent,
      note: payload?.global_discount_note || null,
    },
  };
  delete next.meta.per_line_discounts;
  return recomputeTotals(next);
}

export function applyPreparationPerLineDiscount(invoice, payload) {
  const next = clone(invoice);
  const byId = new Map(
    (payload?.line_discounts || []).map((row) => [String(row.line_id), Number(row.percent)])
  );
  next.lines = next.lines.map((line) => {
    if (isCustomLine(line)) return line;
    const percent = byId.get(String(line.id));
    return {
      ...line,
      preparation_line_percent: Number.isFinite(percent) ? percent : 0,
    };
  });
  next.meta = { ...(next.meta || {}) };
  delete next.meta.global_discount;
  next.meta.per_line_discounts = {
    lines: (payload?.line_discounts || []).map((row) => ({
      line_id: row.line_id,
      percent: row.percent,
    })),
  };
  return recomputeTotals(next);
}

export function removePreparationDiscount(invoice) {
  const next = clone(invoice);
  next.lines = next.lines.map((line) => ({ ...line, preparation_line_percent: 0 }));
  next.meta = { ...(next.meta || {}) };
  delete next.meta.global_discount;
  delete next.meta.per_line_discounts;
  return recomputeTotals(next);
}

export function updatePreparationHeader(invoice, fields) {
  const next = clone(invoice);
  const name = fields.recipient_name != null ? String(fields.recipient_name) : next.recipient_name;
  return recomputeTotals({
    ...next,
    ...fields,
    recipient_name: name,
    client: next.is_partner_invoice
      ? { institution_name: name, is_institution: true }
      : next.client,
    line_time_mode:
      fields.line_time_mode != null ? fields.line_time_mode : next.line_time_mode,
  });
}

export function createPreparationDraftAdapter(readInvoice) {
  const apply = (mutate) => ({ invoice: mutate(readInvoice()) });
  return {
    catalogType: 'preparation',
    pdfApiUrl() {
      return null;
    },
    async load() {
      return readInvoice();
    },
    async updateHeader(fields) {
      return apply((invoice) => updatePreparationHeader(invoice, fields));
    },
    async updateLine(lineId, body) {
      return apply((invoice) => updatePreparationLine(invoice, lineId, body));
    },
    async addCustomLine(body) {
      return apply((invoice) => addPreparationCustomLine(invoice, body));
    },
    async removeLine(lineId) {
      return apply((invoice) => removePreparationLine(invoice, lineId));
    },
    async applyGlobalDiscount(payload) {
      return apply((invoice) => applyPreparationGlobalDiscount(invoice, payload));
    },
    async applyPerLineDiscount(payload) {
      return apply((invoice) => applyPreparationPerLineDiscount(invoice, payload));
    },
    async removeDiscount() {
      return apply((invoice) => removePreparationDiscount(invoice));
    },
    async regeneratePdf() {
      return null;
    },
  };
}

export function invoiceLineMatchKeys(line) {
  const keys = [];
  if (!line || typeof line !== 'object') return keys;
  pushId(keys, line.reservation_id);
  pushId(keys, line.booking_id);
  pushId(keys, line.source_id);
  const meta = line.line_meta;
  if (meta && typeof meta === 'object') {
    pushId(keys, meta.booking_id);
    pushId(keys, meta.primary_booking_id);
    if (Array.isArray(meta.booking_ids)) meta.booking_ids.forEach((id) => pushId(keys, id));
  }
  return keys;
}

const amountsDiffer = (left, right) => Math.abs(round2(left) - round2(right)) > 0.009;

/**
 * Opérations à rejouer sur la facture persistée. Vide si le brouillon n'a pas divergé.
 */
export function planPreparationReplay(preparation, serverInvoice) {
  const steps = [];
  if (!isPreparationInvoice(preparation)) return { steps };
  const serverLines = Array.isArray(serverInvoice?.lines) ? serverInvoice.lines : [];
  const localLines = Array.isArray(preparation.lines) ? preparation.lines : [];
  const kept = new Set();
  localLines.forEach((line) => {
    if (isCustomLine(line)) return;
    (line.match_ids || []).forEach((id) => kept.add(String(id)));
  });
  const removed = (preparation.baseline_match_ids || []).filter((id) => !kept.has(String(id)));
  const removedServerIds = new Set();
  serverLines.forEach((server) => {
    if (server?.id == null) return;
    const keys = invoiceLineMatchKeys(server);
    if (keys.some((key) => removed.includes(key))) {
      removedServerIds.add(server.id);
      steps.push({ type: 'remove', lineId: server.id });
    }
  });

  localLines.forEach((local) => {
    if (isCustomLine(local)) {
      steps.push({
        type: 'add',
        body: {
          description: local.description,
          line_total: local.line_total,
          qty: local.qty || 1,
          custom_mode: local.line_meta?.custom_prestation?.mode,
          time_unit: local.line_meta?.custom_prestation?.time_unit,
          service_date_iso: local.line_meta?.service_date,
        },
      });
      return;
    }
    const legs = local.leg_amounts?.length
      ? local.leg_amounts
      : (local.match_ids || []).map((id) => ({
          match_id: id,
          amount: local.preparation_base ?? local.line_total,
        }));
    legs.forEach((leg, index) => {
      const server = serverLines.find((candidate) => {
        if (candidate?.id == null || removedServerIds.has(candidate.id)) return false;
        return invoiceLineMatchKeys(candidate).includes(String(leg.match_id));
      });
      if (!server) return;
      const serverAmount = Number(server.line_total ?? server.amount);
      const descChanged =
        index === 0 && String(local.description || '') !== String(server.description || '');
      const noteChanged =
        index === 0 &&
        String(local.adjustment_note || '') !== String(server.adjustment_note || '');
      const amountChanged = amountsDiffer(leg.amount, serverAmount);
      if (!descChanged && !noteChanged && !amountChanged) return;
      steps.push({
        type: 'update',
        lineId: server.id,
        body: {
          ...(descChanged ? { description: local.description } : {}),
          ...(noteChanged ? { adjustment_note: local.adjustment_note || null } : {}),
          ...(amountChanged ? { line_total: round2(leg.amount) } : {}),
        },
      });
    });
  });

  const global = preparation.meta?.global_discount;
  const serverPercent = Number(serverInvoice?.meta?.global_discount?.percent);
  if (Number(global?.percent) > 0 && serverPercent !== Number(global.percent)) {
    steps.push({
      type: 'discount',
      body: {
        global_discount_percent: Number(global.percent),
        global_discount_note: global.note || null,
      },
    });
  }
  if (
    preparation.is_partner_invoice &&
    String(preparation.line_time_mode || 'none') !==
      String(serverInvoice?.line_time_mode || 'none')
  ) {
    steps.push({
      type: 'header',
      body: { line_time_mode: preparation.line_time_mode || 'none' },
    });
  }
  const perLine = preparation.meta?.per_line_discounts?.lines;
  if (Array.isArray(perLine) && perLine.length && !(Number(global?.percent) > 0)) {
    const mapped = [];
    perLine.forEach((row) => {
      const local = localLines.find((line) => String(line.id) === String(row.line_id));
      if (!local) return;
      const server = serverLines.find((candidate) =>
        invoiceLineMatchKeys(candidate).some((key) => (local.match_ids || []).includes(key))
      );
      if (server?.id == null) return;
      mapped.push({ line_id: server.id, percent: row.percent });
    });
    if (mapped.length) {
      steps.push({ type: 'per_line_discount', body: { line_discounts: mapped } });
    }
  }
  return { steps };
}

const unwrapMutatedInvoice = (payload) => {
  if (!payload || typeof payload !== 'object') return null;
  if (payload.invoice && payload.invoice.id != null) return payload.invoice;
  if (payload.id != null && Array.isArray(payload.lines)) return payload;
  return null;
};

/** Rejoue le brouillon sur la facture déjà créée. Un second passage sans nouvel écart ne refait rien. */
export async function applyPreparationToPersistedInvoice(adapter, serverInvoice, preparation) {
  let current = serverInvoice;
  const plan = planPreparationReplay(preparation, current);
  for (const step of plan.steps) {
    let payload = null;
    if (step.type === 'remove') payload = await adapter.removeLine(step.lineId, {});
    else if (step.type === 'update') payload = await adapter.updateLine(step.lineId, step.body);
    else if (step.type === 'add') payload = await adapter.addCustomLine(step.body);
    else if (step.type === 'discount') payload = await adapter.applyGlobalDiscount(step.body);
    else if (step.type === 'per_line_discount') {
      payload = await adapter.applyPerLineDiscount(step.body);
    } else if (step.type === 'header') {
      payload = await adapter.updateHeader(step.body);
    }
    const unwrapped = unwrapMutatedInvoice(payload);
    if (unwrapped?.lines) current = unwrapped;
  }
  if (typeof adapter.load === 'function') {
    const reloaded = await adapter.load();
    if (reloaded?.id != null) return reloaded;
  }
  return current;
}

const toBookingInt = (value) => {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
};

/** IDs de réservation encore retenus dans la préparation éditée. */
export function keptBookingIdsFromPreparation(preparation) {
  const kept = [];
  const seen = new Set();
  (preparation?.lines || []).forEach((line) => {
    if (isCustomLine(line)) return;
    (line.match_ids || []).forEach((raw) => {
      const id = toBookingInt(raw);
      if (id == null || seen.has(id)) return;
      seen.add(id);
      kept.push(id);
    });
  });
  return kept;
}

/** IDs retirés de l'aperçu (baseline − gardés). */
export function excludedBookingIdsFromPreparation(preparation) {
  const kept = new Set(keptBookingIdsFromPreparation(preparation).map(String));
  const excluded = [];
  const seen = new Set();
  (preparation?.baseline_match_ids || []).forEach((raw) => {
    const id = toBookingInt(raw);
    if (id == null || kept.has(String(id)) || seen.has(id)) return;
    seen.add(id);
    excluded.push(id);
  });
  return excluded;
}

/** Empreinte SHA-256 alignée sur le backend (`selection_fingerprint`). */
export async function selectionFingerprintFromBookingIds(bookingIds) {
  const ordered = [...new Set(
    (Array.isArray(bookingIds) ? bookingIds : [])
      .map((value) => toBookingInt(value))
      .filter((value) => value != null)
  )].sort((a, b) => a - b);
  const payload = JSON.stringify(ordered);
  // Navigateur : SubtleCrypto ; Jest : branche fallback ci-dessous.
  const root =
    typeof window !== 'undefined'
      ? window
      : typeof global !== 'undefined'
        ? global
        : undefined;
  const cryptoSubtle = root?.crypto?.subtle;
  if (typeof cryptoSubtle?.digest === 'function') {
    const bytes = new TextEncoder().encode(payload);
    const digest = await cryptoSubtle.digest('SHA-256', bytes);
    return [...new Uint8Array(digest)]
      .map((byte) => byte.toString(16).padStart(2, '0'))
      .join('');
  }
  // Jest / environnements sans SubtleCrypto : empreinte déterministe minimale.
  let hash = 0;
  for (let i = 0; i < payload.length; i += 1) {
    hash = (hash * 31 + payload.charCodeAt(i)) >>> 0;
  }
  return `fallback-${hash.toString(16)}-${ordered.join('-')}`;
}

/** Texte de confirmation avant émission (montants issus de la préparation serveur/calculée). */
export function formatPreparationEmissionRecap(preparation) {
  const kept = keptBookingIdsFromPreparation(preparation);
  const ht = Number(preparation?.subtotal_amount || 0);
  const vat = Number(preparation?.vat_total_amount || 0);
  const ttc = Number(preparation?.total_amount || 0);
  const refs = kept.length ? kept.map((id) => `n° ${id}`).join(', ') : 'aucun';
  return [
    `${kept.length} transport${kept.length > 1 ? 's' : ''} : ${refs}`,
    `HT ${ht.toFixed(2)} CHF · TVA ${vat.toFixed(2)} CHF · TTC ${ttc.toFixed(2)} CHF`,
    'La sélection et les montants seront revalidés par le serveur à l’émission.',
  ].join('\n');
}
