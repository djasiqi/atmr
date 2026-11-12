import type { DriverMission } from "../types";
import { getDriverStatusUx, resolveDriverStatusForUx } from "../statusDictionary";
import { getScheduledWhenDisplay } from "./missionDisplay";

const SWISS_TZ = "Europe/Zurich";

export const ABSENT = "Non renseigné";
export const ABSENT_TIME = "Non renseignée";
export const NO_SETTLEMENT = "Aucun dossier de règlement";
export const TARIFF_ABSENT =
  "Tarif contractuel absent. L'encaissement reste refusé.";
export const SETTLEMENT_READ_ERROR =
  "Lecture impossible. Aucune opération n'est envoyée.";
export const CASH_RECEIVED = "Réglé";
export const CASH_PREPARED = "À encaisser";
export const CASH_UNCERTAIN = "Vérification nécessaire";
export const INVOICE_PREPARED = "Facture préparée";
export const SIGNATURE_ABSENT = "Mandat à signer";
export const GPS_UNAVAILABLE = "Coordonnées GPS non disponibles";

const FORWARD_STEPS = ["ASSIGNED", "EN_ROUTE", "ARRIVED", "IN_PROGRESS", "COMPLETED"] as const;

const STEP_LABEL: Record<(typeof FORWARD_STEPS)[number], string> = {
  ASSIGNED: "Mission assignée",
  EN_ROUTE: "En route vers la prise en charge",
  ARRIVED: "Arrivée auprès du patient",
  IN_PROGRESS: "Départ avec le patient",
  COMPLETED: "Mission terminée",
};

export type ConsultationRow = {
  label: string;
  value: string;
};

export type ReachedStep = {
  label: string;
  at: string | null;
};

export type SettlementHistoryLine = {
  label: string;
  at: string | null;
};

function text(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function flag(value: unknown): boolean | null {
  if (value === true) return true;
  if (value === false) return false;
  return null;
}

function numberOrNull(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

/** Instant Europe/Zurich, ou null. Ne retombe jamais sur une autre heure. */
export function formatZurichInstant(value: unknown): string | null {
  const raw = text(value);
  if (!raw) return null;
  const date = new Date(raw);
  if (!Number.isFinite(date.getTime())) return null;
  const datePart = date.toLocaleDateString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
  const timePart = date.toLocaleTimeString("fr-CH", {
    timeZone: SWISS_TZ,
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${datePart} • ${timePart}`;
}

export function scheduledTimeRow(mission: DriverMission): ConsultationRow {
  const when = getScheduledWhenDisplay(mission);
  const confirmed = flag(mission.time_confirmed);
  if (!when) return { label: "Heure prévue", value: ABSENT_TIME };
  if (confirmed === false) return { label: "Heure prévue", value: `${when}, non confirmée` };
  return { label: "Heure prévue", value: when };
}

export function actualTimeRow(label: string, value: unknown): ConsultationRow {
  return { label, value: formatZurichInstant(value) ?? ABSENT_TIME };
}

export function tripTypeLabel(mission: DriverMission): string {
  const returning = flag(mission.is_return);
  const roundTrip = flag(mission.is_round_trip);
  if (returning === true) return "Retour";
  if (roundTrip === true) return "Aller-retour";
  if (returning === false) return "Aller";
  return ABSENT;
}

export function sequenceLabel(mission: DriverMission): string | null {
  const raw = mission.route_sequence_number;
  if (typeof raw !== "number" || !Number.isFinite(raw)) return null;
  return `Étape ${raw}`;
}

export function hasMissionCoordinates(mission: DriverMission): boolean {
  return (
    numberOrNull(mission.pickup_lat) != null &&
    numberOrNull(mission.pickup_lon) != null &&
    numberOrNull(mission.dropoff_lat) != null &&
    numberOrNull(mission.dropoff_lon) != null
  );
}

export function assistanceLines(mission: DriverMission): string[] {
  const lines: string[] = [];
  if (flag(mission.wheelchair_client_has) === true) lines.push("Client en chaise roulante");
  if (flag(mission.wheelchair_need) === true) lines.push("Prendre une chaise roulante");
  const notes = text(mission.notes_medical);
  if (notes) lines.push(notes);
  if (lines.length > 0) return lines;
  if (flag(mission.wheelchair_client_has) === false && flag(mission.wheelchair_need) === false) {
    return ["Aucune assistance signalée"];
  }
  return [ABSENT];
}

export function accessLines(mission: DriverMission): string[] {
  const pairs: [string, unknown][] = [
    ["Établissement", mission.medical_facility],
    ["Service", mission.hospital_service],
    ["Médecin", mission.doctor_name],
    ["Étage départ", mission.pickup_floor],
    ["Code départ", mission.pickup_door_code],
    ["Accès départ", mission.pickup_access_notes],
    ["Étage arrivée", mission.dropoff_floor],
    ["Code arrivée", mission.dropoff_door_code],
    ["Accès arrivée", mission.dropoff_access_notes],
  ];
  const lines = pairs.flatMap(([label, value]) => {
    const shown = text(value);
    return shown ? [`${label} : ${shown}`] : [];
  });
  return lines.length > 0 ? lines : ["Aucune consigne d'accès"];
}

/** Étapes déduites du statut courant. Les statuts terminaux imprévus ne reconstituent pas le chemin. */
export function reachedSteps(mission: DriverMission): ReachedStep[] {
  const key = resolveDriverStatusForUx(typeof mission.status === "string" ? mission.status : null);
  const index = FORWARD_STEPS.indexOf(key as (typeof FORWARD_STEPS)[number]);
  if (index < 0) {
    return [{ label: getDriverStatusUx(key).label, at: null }];
  }
  return FORWARD_STEPS.slice(0, index + 1).map((step) => ({
    label: STEP_LABEL[step],
    at: timeForStep(step, mission),
  }));
}

function timeForStep(step: (typeof FORWARD_STEPS)[number], mission: DriverMission): string | null {
  if (step === "ARRIVED") return formatZurichInstant(mission.arrived_at);
  if (step === "IN_PROGRESS") return formatZurichInstant(mission.boarded_at);
  if (step === "COMPLETED") return formatZurichInstant(mission.completed_at);
  return null;
}

const CASH_LABEL: Record<string, string> = {
  prepared: CASH_PREPARED,
  received: CASH_RECEIVED,
  uncertain: CASH_UNCERTAIN,
  returned: "Vérification nécessaire",
  administrative: "Vérification nécessaire",
};

export function cashStatusLabel(state: unknown): string {
  const key = text(state);
  if (!key) return ABSENT;
  return CASH_LABEL[key] ?? "Statut financier à vérifier";
}

export type CashRead = {
  amount_due_chf?: unknown;
  currency?: unknown;
  state?: unknown;
  debtor_label?: unknown;
  company_name?: unknown;
  production_activation?: unknown;
  receipt_number?: unknown;
  receipt_partial?: unknown;
  balance_chf?: unknown;
  recognized_net_chf?: unknown;
  remaining_due_chf?: unknown;
  operations?: unknown;
  history?: unknown;
};

export type PatientRead = {
  state?: unknown;
  debtor_label?: unknown;
  company_name?: unknown;
  invoice_issued?: unknown;
  real_signature_collected?: unknown;
  due_chf?: unknown;
  recognized_chf?: unknown;
  balance_chf?: unknown;
  real_payment?: unknown;
  preparation_label?: unknown;
  history?: unknown;
};

export type ReservationRead = {
  price_chf?: unknown;
  due_chf?: unknown;
  recognized_chf?: unknown;
  balance_chf?: unknown;
  debtor_label?: unknown;
  creditor_label?: unknown;
  currency?: unknown;
};

export type SettlementConsultation = {
  unavailable: boolean;
  empty: boolean;
  amount: string | null;
  price: string | null;
  due: string | null;
  balance: string | null;
  debtor: string;
  creditor: string;
  lines: string[];
  history: SettlementHistoryLine[];
};

function money(value: unknown, currency: string): string | null {
  const raw = text(value);
  if (!raw) return null;
  return raw.includes("CHF") ? raw : `${raw} ${currency}`;
}

function positiveMoney(value: unknown, currency: string): string | null {
  const shown = money(value, currency);
  if (!shown) return null;
  const numeric = Number(String(value).replace(/CHF/gi, "").trim());
  if (!Number.isFinite(numeric) || numeric <= 0) return null;
  return shown;
}

export function isTariffAbsence(
  errors: { failed: boolean; code: string | null }[],
): boolean {
  if (!errors.some((item) => item.failed)) return false;
  return errors.every((item) => !item.failed || item.code === "montant_absent");
}

function blankConsultation(lines: string[], unavailable: boolean): SettlementConsultation {
  return {
    unavailable,
    empty: !unavailable,
    amount: null,
    price: null,
    due: null,
    balance: null,
    debtor: ABSENT,
    creditor: ABSENT,
    lines,
    history: [],
  };
}

export function settlementConsultation(input: {
  cash: CashRead | null;
  patient: PatientRead | null;
  reservation?: ReservationRead | null;
  failed: boolean;
  tariffAbsent?: boolean;
  notices?: string[];
}): SettlementConsultation {
  if (input.failed) {
    return blankConsultation([SETTLEMENT_READ_ERROR], true);
  }
  if (input.tariffAbsent) {
    return blankConsultation([TARIFF_ABSENT], false);
  }
  const reservation = input.reservation ?? null;
  const currency =
    text(reservation?.currency) ?? text(input.cash?.currency) ?? "CHF";
  const price = positiveMoney(reservation?.price_chf, currency);
  const due =
    money(input.cash?.amount_due_chf, currency) ??
    money(reservation?.due_chf, currency) ??
    money(input.patient?.due_chf, currency);
  const recognized =
    money(input.cash?.recognized_net_chf, currency) ??
    money(reservation?.recognized_chf, currency) ??
    money(input.patient?.recognized_chf, currency);
  const balance =
    money(input.cash?.remaining_due_chf, currency) ??
    money(reservation?.balance_chf, currency) ??
    money(input.patient?.balance_chf, currency) ??
    money(input.cash?.balance_chf, currency);
  if (!input.cash && !input.patient && !price && !due) {
    return blankConsultation([NO_SETTLEMENT], false);
  }
  const lines: string[] = [];
  if (!input.cash && !input.patient) lines.push(NO_SETTLEMENT);
  if (input.cash) lines.push(cashStatusLabel(input.cash.state));
  if (input.patient && input.patient.invoice_issued !== true) lines.push(INVOICE_PREPARED);
  if (price) lines.push(`Prix de la réservation ${price}`);
  if (due) lines.push(`Montant dû ${due}`);
  const collected = text(input.cash?.recognized_net_chf);
  const remaining = text(input.cash?.remaining_due_chf);
  if (collected) lines.push(`${collected} CHF encaissés`);
  if (remaining) lines.push(`${remaining} CHF restants`);
  if (recognized) lines.push(`Montant reconnu ${recognized}`);
  if (balance) lines.push(`Solde restant ${balance}`);
  for (const operation of operationDisplays(input.cash?.operations)) {
    lines.push(operation);
  }
  for (const notice of input.notices ?? []) {
    if (notice.trim()) lines.push(notice.trim());
  }
  if (input.patient && input.patient.real_signature_collected !== true) lines.push(SIGNATURE_ABSENT);
  if (input.cash?.production_activation === true) {
    lines.push("Encaissement réel signalé par le serveur");
  }
  const receiptNumber = text(input.cash?.receipt_number);
  if (receiptNumber) {
    lines.push(
      input.cash?.receipt_partial === true
        ? `Quittance ${receiptNumber}, réception partielle`
        : `Quittance ${receiptNumber}`
    );
  }
  const amount = text(input.cash?.amount_due_chf);
  return {
    unavailable: false,
    empty: false,
    amount: price ?? (amount ? `${amount} ${currency}` : due),
    price,
    due,
    balance,
    debtor:
      text(input.cash?.debtor_label) ??
      text(input.patient?.debtor_label) ??
      text(reservation?.debtor_label) ??
      ABSENT,
    creditor:
      text(input.cash?.company_name) ??
      text(input.patient?.company_name) ??
      text(reservation?.creditor_label) ??
      ABSENT,
    lines,
    history: [
      ...historyLines(input.cash?.history, true),
      ...historyLines(input.patient?.history, false),
    ],
  };
}

function operationDisplays(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    const display = text(record.display);
    if (!display) return [];
    const parts = [display];
    const mode = text(record.payment_mode);
    if (mode === "especes") parts.push("Espèces");
    else if (mode) parts.push(mode);
    const reference = text(record.reference);
    if (reference) parts.push(`Réf. ${reference}`);
    else if (record.id != null) parts.push(`Réf. CASH-${String(record.id)}`);
    if (record.booking_id != null) parts.push(`Mission ${String(record.booking_id)}`);
    const state = cashStatusLabel(record.state);
    if (state !== ABSENT) parts.push(state);
    const at = formatZurichInstant(record.created_at);
    if (at) parts.push(at);
    return [parts.join(" · ")];
  });
}

function historyLines(value: unknown, cash: boolean): SettlementHistoryLine[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    const action = text(record.action);
    const label = cash
      ? cashHistoryLabel(action, record.to_state)
      : patientEventLabel(record.to_state);
    return [{ label, at: formatZurichInstant(record.at) }];
  });
}

function cashHistoryLabel(action: string | null, state: unknown): string {
  if (action === "compensate_cash") return "Correction";
  if (action === "declare_handover") return "Remise de caisse";
  if (action === "declare_cash") return cashStatusLabel(state);
  return cashStatusLabel(state);
}

function patientEventLabel(state: unknown): string {
  const key = text(state);
  if (key === "preparing") return INVOICE_PREPARED;
  if (key === "administrative") return "Dossier patient enregistré, facture non émise";
  return "Événement de facture patient";
}
