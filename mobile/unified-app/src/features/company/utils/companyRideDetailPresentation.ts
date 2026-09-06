import dayjs from "dayjs";
import type { BookingIdentityView } from "./bookingIdentity";
import { getBookingSourceMeta } from "./bookingSourceLabels";

export type RideDetailInfoRow = {
  label: string;
  value: string;
  tone?: "danger";
  /** Champ inconnu du snapshot — skeleton, jamais « — » factice. */
  pending?: boolean;
};

export type RideTimelineItem = {
  event: string;
  date: string;
  type?: string;
};

export type RideDestinationDetails = {
  establishment: string | null;
  service: string | null;
  doctor: string | null;
  clinicalLine: string | null;
};

export type RideBillingSummary = {
  amountLabel: string;
  recipientLabel: string;
  invoiceStatusLabel: string | null;
  hasUnpaidInvoice: boolean;
  adjustedNote: string | null;
};

const BILLED_TO_LABELS: Record<string, string> = {
  patient: "Patient",
  clinic: "Clinique",
  insurance: "Assurance",
  other: "Autre",
};

function deepRead(
  root: Record<string, unknown> | null | undefined,
  keys: string[]
): unknown {
  if (!root) return null;
  const queue: unknown[] = [root];
  const seen = new Set<unknown>();
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const row = current as Record<string, unknown>;
    for (const key of keys) {
      if (key in row && row[key] != null && String(row[key]).trim() !== "") {
        return row[key];
      }
    }
    for (const value of Object.values(row)) {
      if (value && typeof value === "object") queue.push(value);
    }
  }
  return null;
}

function readString(root: Record<string, unknown> | null | undefined, keys: string[]): string | null {
  const raw = deepRead(root, keys);
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function readClinicalLabel(value: unknown): string | null {
  if (value == null) return null;
  const s = String(value).trim();
  if (!s || s === "Non spécifié") return null;
  return s;
}

export function formatRideCurrency(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${n.toFixed(2)} CHF`;
}

export function formatRideShortDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const parsed = dayjs(dateStr);
  if (!parsed.isValid()) return "—";
  return parsed.format("DD.MM.YYYY • HH:mm");
}

export function formatRideBirthDate(value: unknown): string | null {
  if (value == null) return null;
  const s = String(value).trim();
  if (!s) return null;
  const parsed = dayjs(s);
  if (!parsed.isValid()) return null;
  return parsed.format("DD.MM.YYYY");
}

function resolvePassengerGender(data: Record<string, unknown>): string | null {
  const identity = data.identity as { passenger?: { gender?: unknown } } | null | undefined;
  const passenger = data.passenger as { gender?: unknown } | null | undefined;
  const client = data.client as { gender?: unknown } | null | undefined;
  const raw =
    identity?.passenger?.gender ??
    passenger?.gender ??
    client?.gender ??
    data.client_gender ??
    data.passenger_gender;
  if (raw == null) return null;
  const g = String(raw).trim().toUpperCase();
  return g.length > 0 ? g : null;
}

export function formatPassengerDisplayName(
  identity: BookingIdentityView,
  data: Record<string, unknown>
): string {
  const name = identity.passengerLabel?.trim() || "Non spécifié";
  const gender = resolvePassengerGender(data);
  if (gender === "FEMME" || gender === "FEMALE" || gender === "F") {
    return name.startsWith("Mme ") || name.startsWith("Madame ") ? name : `Mme ${name}`;
  }
  if (gender === "HOMME" || gender === "MALE" || gender === "M") {
    return name.startsWith("M. ") || name.startsWith("Monsieur ") ? name : `M. ${name}`;
  }
  return name;
}

export function formatRideOriginLine(identity: BookingIdentityView, data: Record<string, unknown>): string | null {
  const source = identity.source;
  if (!source?.name?.trim()) return null;
  const sourceMeta = getBookingSourceMeta(source.type);
  const parts: string[] = [sourceMeta.label];
  const originChannel = readString(data, ["origin_channel"]);
  if (originChannel) parts.push(originChannel);
  parts.push(source.name.trim());
  const code = source.code?.trim();
  if (code) {
    return `${parts.join(" · ")} (${code})`;
  }
  return parts.join(" · ");
}

export function readRidePassengerBirthDate(data: Record<string, unknown>): string | null {
  const identity = data.identity as { passenger?: { birth_date?: unknown } } | null | undefined;
  const passenger = data.passenger as { birth_date?: unknown } | null | undefined;
  const client = data.client as { birth_date?: unknown } | null | undefined;
  const raw =
    identity?.passenger?.birth_date ??
    passenger?.birth_date ??
    client?.birth_date ??
    data.birth_date ??
    data.client_birth_date;
  return formatRideBirthDate(raw);
}

export function readRideAmount(data: Record<string, unknown>): number | null {
  const raw = deepRead(data, ["amount", "requested_amount", "total_amount"]);
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function readRideDestinationDetails(data: Record<string, unknown>): RideDestinationDetails {
  const legClinical = data.institution_leg as Record<string, unknown> | null | undefined;
  const establishment =
    readClinicalLabel(data.medical_facility) ??
    readClinicalLabel(legClinical?.establishment) ??
    null;
  const service =
    readClinicalLabel(data.hospital_service) ?? readClinicalLabel(legClinical?.service) ?? null;
  const doctor =
    readClinicalLabel(data.doctor_name) ?? readClinicalLabel(legClinical?.doctor) ?? null;
  const clinicalLine = [establishment, service, doctor].filter(Boolean).join(" · ") || null;
  return { establishment, service, doctor, clinicalLine };
}

export function buildRideTimeline(data: Record<string, unknown>, driverName?: string | null): RideTimelineItem[] {
  const events: RideTimelineItem[] = [];
  const driver = driverName?.trim() || null;
  const it = data.institution_timeline as Record<string, unknown> | null | undefined;

  if (it) {
    const instName = typeof it.institution_name === "string" ? it.institution_name : null;
    if (it.created_at) {
      const by =
        typeof it.created_by_name === "string" && it.created_by_name.trim()
          ? ` par ${it.created_by_name.trim()}`
          : "";
      const inst = instName ? ` (${instName})` : "";
      events.push({
        event: `Demande créée${by}${inst}`,
        date: String(it.created_at),
      });
    }
    if (it.sent_at) events.push({ event: "Demande envoyée", date: String(it.sent_at) });
    if (it.accepted_at) {
      const by =
        typeof it.accepted_by_company_name === "string" && it.accepted_by_company_name.trim()
          ? ` par ${it.accepted_by_company_name.trim()}`
          : "";
      events.push({ event: `Demande acceptée${by}`, date: String(it.accepted_at) });
    }
    if (it.converted_at) events.push({ event: "Réservation créée", date: String(it.converted_at) });
    if (it.cancelled_at) events.push({ event: "Demande annulée", date: String(it.cancelled_at) });
  } else if (data.created_at) {
    events.push({ event: "Réservation créée", date: String(data.created_at) });
  }

  if (data.accepted_at && !it?.accepted_at) {
    events.push({
      event: `Acceptée${driver ? ` par ${driver}` : ""}`,
      date: String(data.accepted_at),
    });
  }
  if (data.assigned_at) {
    events.push({ event: `Assignée${driver ? ` à ${driver}` : ""}`, date: String(data.assigned_at) });
  }

  const journey = Array.isArray(data.route_journey) ? data.route_journey : null;
  if (journey?.length) {
    journey.forEach((ev) => {
      if (ev && typeof ev === "object" && (ev as { date?: unknown }).date) {
        const row = ev as { event?: unknown; date: unknown; type?: unknown };
        events.push({
          event: String(row.event ?? "Événement"),
          date: String(row.date),
          type: typeof row.type === "string" ? row.type : undefined,
        });
      }
    });
  } else {
    if (data.picked_up_at || data.boarded_at) {
      events.push({
        event: `Prise en charge${driver ? ` par ${driver}` : ""}`,
        date: String(data.picked_up_at ?? data.boarded_at),
      });
    }
    if (data.completed_at) {
      events.push({ event: "Dépose / course terminée", date: String(data.completed_at) });
    }
  }

  if (data.started_at) events.push({ event: "Course démarrée", date: String(data.started_at) });

  const cancelledAt = data.cancelled_at ?? data.canceled_at;
  if (cancelledAt && !it?.cancelled_at) {
    const roleMap: Record<string, string> = {
      company: "Entreprise",
      driver: "Chauffeur",
      admin: "Admin",
      system: "Système",
    };
    const byRole =
      typeof data.cancelled_by_role === "string"
        ? roleMap[data.cancelled_by_role] ?? ""
        : "";
    const reason =
      readString(data, ["cancellation_display_label", "cancellation_reason_code"]) ?? "";
    let detail = "Annulée";
    if (byRole) detail += ` par ${byRole}`;
    if (reason) detail += ` — ${reason}`;
    if (data.is_cancellation_billable === true) detail += " (facturée)";
    else if (data.is_cancellation_billable === false) detail += " (non facturée)";
    events.push({ event: detail, date: String(cancelledAt), type: "cancel" });
  }

  return events
    .filter((item) => item.date && dayjs(item.date).isValid())
    .sort((a, b) => dayjs(b.date).valueOf() - dayjs(a.date).valueOf());
}

export function buildRideBillingSummary(
  data: Record<string, unknown>,
  linkedInvoice: Record<string, unknown> | null | undefined
): RideBillingSummary {
  const billing = data.billing as Record<string, unknown> | null | undefined;
  const metadata = data.metadata_json as Record<string, unknown> | null | undefined;
  const amount = readRideAmount(data);
  const billedToType = String(
    data.billed_to_type ?? billing?.billed_to_type ?? metadata?.billing_resolution_intent ?? "patient"
  ).toLowerCase();
  const recipientLabel = BILLED_TO_LABELS[billedToType] ?? billedToType;

  const originalAmount = Number(
    data.amount_original ?? data.original_amount ?? data.requested_amount ?? NaN
  );
  const adjustedDelta = Number.isFinite(originalAmount) ? (amount ?? 0) - originalAmount : null;
  const adjustedNote =
    Number.isFinite(originalAmount) &&
    adjustedDelta != null &&
    Math.abs(adjustedDelta) >= 0.01
      ? `Montant saisi : ${formatRideCurrency(originalAmount)} — Ajusté : ${
          adjustedDelta >= 0 ? "+" : "-"
        }${formatRideCurrency(Math.abs(adjustedDelta))}`
      : null;

  let invoiceStatusLabel: string | null = null;
  let hasUnpaidInvoice = false;
  if (linkedInvoice) {
    invoiceStatusLabel = String(linkedInvoice.status ?? linkedInvoice.payment_status ?? "").trim() || null;
    const status = invoiceStatusLabel?.toLowerCase() ?? "";
    hasUnpaidInvoice = status === "unpaid" || status === "pending" || status === "overdue";
  }

  return {
    amountLabel: formatRideCurrency(amount),
    recipientLabel,
    invoiceStatusLabel,
    hasUnpaidInvoice,
    adjustedNote,
  };
}

export function buildRideDetailInfoRows(
  data: Record<string, unknown>,
  identity: BookingIdentityView,
  options: {
    statusLabel: string;
    scheduledIso: string | null;
    driverDisplay: string;
    billingSummary: RideBillingSummary;
    /** Snapshot liste : ne pas afficher les absences comme des vides métier. */
    awaitingServer?: boolean;
  }
): RideDetailInfoRow[] {
  const rows: RideDetailInfoRow[] = [];
  rows.push({ label: "Passager", value: formatPassengerDisplayName(identity, data) });

  const origin = formatRideOriginLine(identity, data);
  if (origin) rows.push({ label: "Origine", value: origin });

  if (identity.requester?.name) {
    rows.push({ label: "Demandeur", value: identity.requester.name });
  }
  if (identity.ownership?.owner_company_name) {
    rows.push({ label: "Propriétaire", value: identity.ownership.owner_company_name });
  }
  if (identity.execution?.executing_company_name) {
    rows.push({ label: "Exécutant", value: identity.execution.executing_company_name });
  }
  if (identity.upstream?.name) {
    const upstreamCode = identity.upstream.code?.trim();
    rows.push({
      label: "Source amont",
      value: upstreamCode ? `${identity.upstream.name} (${upstreamCode})` : identity.upstream.name,
    });
  }

  const timeConfirmed = data.time_confirmed;
  const horaire =
    timeConfirmed === false
      ? "⏱️ À définir"
      : options.scheduledIso
        ? formatRideShortDate(options.scheduledIso)
        : "—";
  rows.push({ label: "Horaire", value: horaire });

  const amount = readRideAmount(data);
  if (amount != null) {
    rows.push({ label: "Montant", value: formatRideCurrency(amount) });
  }

  rows.push({ label: "Chauffeur", value: options.driverDisplay });
  rows.push({ label: "Statut", value: options.statusLabel });

  const phone =
    readString(data, ["phone"]) ??
    readString(data.client as Record<string, unknown>, ["contact_phone", "phone"]);
  if (phone) {
    rows.push({ label: "Téléphone", value: phone });
  } else if (options.awaitingServer) {
    rows.push({ label: "Téléphone", value: "", pending: true });
  }

  const birthDate = readRidePassengerBirthDate(data);
  if (birthDate) {
    rows.push({ label: "Date de naissance", value: birthDate });
  } else if (options.awaitingServer) {
    rows.push({ label: "Date de naissance", value: "", pending: true });
  }

  const externalRef =
    readString(data, ["external_reference"]) ??
    readString(data.passenger as Record<string, unknown>, ["external_reference"]) ??
    readString(metadataFrom(data), ["external_reference"]);
  if (externalRef) {
    rows.push({ label: "Réf. patient", value: externalRef });
  } else if (options.awaitingServer) {
    rows.push({ label: "Réf. patient", value: "", pending: true });
  }

  if (options.billingSummary.invoiceStatusLabel) {
    rows.push({ label: "Facture", value: options.billingSummary.invoiceStatusLabel });
  } else if (!options.awaitingServer) {
    rows.push({ label: "Facturation", value: "Aucune facture liée" });
  }

  if (options.billingSummary.hasUnpaidInvoice) {
    rows.push({ label: "Alerte", value: "Impayé", tone: "danger" });
  }

  return rows;
}

function metadataFrom(data: Record<string, unknown>): Record<string, unknown> | null {
  const meta = data.metadata_json;
  return meta && typeof meta === "object" ? (meta as Record<string, unknown>) : null;
}
