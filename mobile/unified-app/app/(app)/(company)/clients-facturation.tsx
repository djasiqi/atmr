import { useEffect, useMemo, useState } from "react";
import { Platform, Pressable, StyleSheet, TouchableOpacity, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useLocalSearchParams } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import {
  useCompanyClientReadonlyDetailQuery,
  useCompanyClientsReadonlyQuery,
  useCompanyInvoicesReadonlyQuery,
} from "../../../src/features/company/hooks";
import {
  AppCard,
  AppEmptyState,
  AppInput,
  AppSpinner,
  AppText,
  brandTextMuted,
  Screen,
  useResponsiveTokens,
} from "../../../src/design/responsive";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { createShadow } from "../../../src/styles/shadowStyles";

const cardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

const BORDER_SLATE = "rgba(148, 163, 184, 0.22)";

/** Aligné sur `DispatchRideListCard` : pas de fond / surbrillance « bouton » au survol sur le web. */
const DISPATCH_SUMMARY_TOUCHABLE_WEB =
  Platform.OS === "web"
    ? ({
        backgroundColor: "transparent",
        cursor: "pointer",
        userSelect: "none",
      } as const)
    : ({} as const);

const INVOICE_KPI_PRESSABLE_WEB =
  Platform.OS === "web" ? ({ cursor: "pointer" as const, userSelect: "none" as const } as const) : ({} as const);

/** Ombre alignée sur `DispatchRideListCard` (`cardSurfaceShadow`). */
const clientRowCardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

/** Même cible que les pastilles résumé dispatch (80×28). */
const CLIENT_REF_BADGE_W = 80;
const CLIENT_REF_BADGE_H = 28;

function extractClientId(row: Record<string, unknown>): number | null {
  const candidate = row.client_id ?? row.id;
  if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
  if (typeof candidate === "string" && candidate.trim().length > 0) {
    const parsed = Number.parseInt(candidate, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function isInstitutionClientRow(data: Record<string, unknown>): boolean {
  const v = data.is_institution;
  if (v === true || v === 1) return true;
  if (typeof v === "string" && ["1", "true", "yes"].includes(v.trim().toLowerCase())) return true;
  return false;
}

/** E-mail technique généré (non pertinent pour l’affichage utilisateur). */
function isInternalPlaceholderEmail(email: string): boolean {
  const lower = email.trim().toLowerCase();
  return lower.includes("@internal.atmr.local") || /\.internal\b/i.test(lower);
}

function formatDetailValue(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") {
    const t = value.trim();
    return t.length > 0 ? t : null;
  }
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return null;
}

/** Texte équivalent à « vide » pour la fiche client (non affiché). */
function isPlaceholderDisplay(text: string): boolean {
  const raw = text.trim();
  if (!raw) return true;
  const key = raw
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "");
  const blanks = new Set([
    "non specifie",
    "non renseigne",
    "unspecified",
    "n/a",
    "na",
    "-",
    "—",
    "–",
    "...",
    "aucun",
    "aucune",
    "nom non renseigne",
  ]);
  return blanks.has(key);
}

/** Personne rattachée au dossier (contact utilisateur), distincte du nom d’établissement. */
function extractClientPersonName(row: Record<string, unknown>): string | null {
  const first = row.first_name;
  const last = row.last_name;
  if (
    (typeof first === "string" && first.trim().length > 0) ||
    (typeof last === "string" && last.trim().length > 0)
  ) {
    const name = [typeof first === "string" ? first.trim() : "", typeof last === "string" ? last.trim() : ""]
      .filter(Boolean)
      .join(" ");
    if (name.length > 0) return name;
  }
  for (const key of ["full_name", "name", "display_name"] as const) {
    const s = formatDetailValue(row[key]);
    if (s && !isPlaceholderDisplay(s)) return s;
  }
  return null;
}

/**
 * Titre principal en liste : nom de l’établissement pour une institution,
 * sinon la personne (ou repli e-mail / téléphone comme avant).
 */
function extractClientLabel(row: Record<string, unknown>): string {
  if (isInstitutionClientRow(row)) {
    const inst = formatDetailValue(row.institution_name);
    if (inst) return inst;
  }
  const person = extractClientPersonName(row);
  if (person) return person;
  const candidate = row.label ?? row.email ?? row.phone ?? "Client";
  const coerced = formatDetailValue(candidate);
  if (coerced) return coerced;
  return typeof candidate === "string" ? candidate : "Client";
}

/**
 * Sous-titre sous l’en-tête de carte : téléphone ou e-mail utile uniquement.
 * (Le nom de la personne de contact est dans la fiche déployée, pas ici.)
 */
function extractClientListSubtitle(row: Record<string, unknown>): string | null {
  const phone = row.phone ?? row.mobile ?? row.phone_number;
  const phoneStr = typeof phone === "string" && phone.trim().length > 0 ? phone.trim() : null;
  const emailRaw =
    row.email ??
    row.contact_email ??
    (row.user && typeof row.user === "object"
      ? (row.user as Record<string, unknown>).email
      : undefined);
  const emailStr =
    typeof emailRaw === "string" && emailRaw.trim().length > 0 && !isInternalPlaceholderEmail(emailRaw)
      ? emailRaw.trim()
      : null;

  if (phoneStr) return phoneStr;
  if (emailStr) return emailStr;
  return null;
}

function clientInitialsFromLabel(label: string): string {
  const parts = label.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function detailPayloadId(data: Record<string, unknown> | null | undefined): number | null {
  if (!data) return null;
  const raw = data.id;
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (typeof raw === "string") {
    const n = Number.parseInt(raw, 10);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function formatDomicileLine(data: Record<string, unknown>): string | null {
  const dom = data.domicile;
  if (!dom || typeof dom !== "object") return null;
  const o = dom as Record<string, unknown>;
  const parts = [o.address, o.zip, o.city]
    .map((x) => (typeof x === "string" ? x.trim() : ""))
    .filter((s) => s.length > 0);
  return parts.length > 0 ? parts.join(", ") : null;
}

/** Affiche la facturation seulement si elle diffère du domicile (évite doublon visuel). */
function formatBillingDistinctFromDomicile(data: Record<string, unknown>): string | null {
  const domLine = formatDomicileLine(data);
  const billRaw = formatDetailValue(data.billing_address);
  const billStr = billRaw ? String(billRaw).trim() : "";
  if (!billStr) return null;
  if (!domLine) return billStr;
  const norm = (s: string) =>
    s
      .toLowerCase()
      .normalize("NFD")
      .replace(/\p{M}/gu, "")
      .replace(/\s+/g, " ")
      .trim();
  if (norm(billStr) === norm(domLine)) return null;
  return billStr;
}

function formatAccessDomLine(data: Record<string, unknown>): string | null {
  const acc = data.access;
  if (!acc || typeof acc !== "object") return null;
  const o = acc as Record<string, unknown>;
  const bits: string[] = [];
  const door = o.door_code;
  if (typeof door === "string" && door.trim()) bits.push(`Porte ${door.trim()}`);
  const floor = o.floor;
  if (typeof floor === "string" && floor.trim()) bits.push(`Étage ${floor.trim()}`);
  else if (typeof floor === "number" && Number.isFinite(floor)) bits.push(`Étage ${floor}`);
  const notes = o.notes;
  if (typeof notes === "string" && notes.trim()) bits.push(notes.trim());
  return bits.length > 0 ? bits.join(" · ") : null;
}

function formatGenderFr(data: Record<string, unknown>): string | null {
  const user = data.user;
  let raw: unknown = null;
  if (user && typeof user === "object") raw = (user as Record<string, unknown>).gender;
  if (raw == null) raw = data.gender;
  if (raw == null) return null;
  const s = String(raw).toLowerCase().trim();
  if (!s || s === "non spécifié" || s === "unspecified") return null;
  if (s === "male" || s === "m" || s === "homme") return "Homme";
  if (s === "female" || s === "f" || s === "femme") return "Femme";
  if (s === "other" || s === "autre") return "Autre";
  return String(raw);
}

/** Affiche une date de naissance en jj.mm.aaaa (ex. API `1988-10-04`). */
function formatBirthDateFrench(isoLike: string): string {
  const trimmed = isoLike.trim();
  const datePart = trimmed.includes("T") ? (trimmed.split("T")[0] ?? trimmed) : trimmed;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(datePart);
  if (!m) return trimmed;
  const [, y, mo, d] = m;
  return `${d}.${mo}.${y}`;
}

function formatBirthDate(data: Record<string, unknown>): string | null {
  const user = data.user;
  let raw: unknown = null;
  if (user && typeof user === "object") raw = (user as Record<string, unknown>).birth_date;
  if (raw == null) raw = data.birth_date;
  const s = formatDetailValue(raw);
  if (!s) return null;
  return formatBirthDateFrench(s);
}

function formatPreferentialRate(data: Record<string, unknown>): string | null {
  const v = data.preferential_rate;
  if (v == null) return null;
  if (typeof v === "number" && Number.isFinite(v)) return `${v.toFixed(2)} CHF`;
  if (typeof v === "string" && v.trim()) return `${v.trim()} CHF`;
  return null;
}

function formatClientPhoneDetail(data: Record<string, unknown>): string | null {
  const direct = formatDetailValue(data.phone ?? data.contact_phone);
  if (direct) return direct;
  const user = data.user;
  if (user && typeof user === "object") {
    return formatDetailValue((user as Record<string, unknown>).phone);
  }
  return null;
}

function formatDetailEmail(data: Record<string, unknown>): string | null {
  const c = formatDetailValue(data.contact_email);
  if (c) return c;
  const user = data.user;
  if (user && typeof user === "object") {
    return formatDetailValue((user as Record<string, unknown>).email);
  }
  return null;
}

/** E-mail affichable (masque les adresses techniques internes). */
function formatDetailEmailForDisplay(data: Record<string, unknown>): string | null {
  const raw = formatDetailEmail(data);
  if (!raw) return null;
  return isInternalPlaceholderEmail(raw) ? null : raw;
}

function formatClientTypeDetailValue(data: Record<string, unknown>): string {
  return isInstitutionClientRow(data) ? "Institution" : "Client privé";
}

function ClientExpandRouteRow({
  icon,
  lines,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  lines: { label: string; value: string }[];
}) {
  const visible = lines.filter(
    (x) => x.value.trim().length > 0 && !isPlaceholderDisplay(x.value)
  );
  if (!visible.length) return null;
  return (
    <View style={styles.clientExpandRow}>
      <View style={styles.clientExpandIcon}>
        <Ionicons name={icon} size={16} color={E.BRAND} />
      </View>
      <View style={styles.clientExpandTexts}>
        {visible.map((row) => (
          <View key={row.label} style={styles.clientExpandLineBlock}>
            <AppText variant="caption" style={styles.clientExpandLabel}>
              {row.label}
            </AppText>
            <AppText
              variant="caption"
              style={styles.clientExpandValue}
              numberOfLines={4}
              ellipsizeMode="tail"
              selectable={Platform.OS === "web"}
            >
              {row.value}
            </AppText>
          </View>
        ))}
      </View>
    </View>
  );
}
function extractInvoiceValue(row: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const value = row[key];
    if (value == null) continue;
    if (typeof value === "string" && value.trim().length > 0) return value;
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return "n/a";
}

const INVOICE_PLACEHOLDER = "—";

function extractInvoiceClientDisplay(row: Record<string, unknown>): string {
  const c = row.client;
  if (c && typeof c === "object") {
    const o = c as Record<string, unknown>;
    const inst = o.institution_name;
    if (typeof inst === "string" && inst.trim()) return inst.trim();
    const fn = typeof o.first_name === "string" ? o.first_name.trim() : "";
    const ln = typeof o.last_name === "string" ? o.last_name.trim() : "";
    const parts = [fn, ln].filter(Boolean);
    if (parts.length) return parts.join(" ");
  }
  return "";
}

function formatInvoiceAmount(row: Record<string, unknown>): string {
  const raw = row.total_amount ?? row.amount_total ?? row.amount;
  let n: number | null = null;
  if (typeof raw === "number" && Number.isFinite(raw)) n = raw;
  else if (typeof raw === "string" && raw.trim() && !Number.isNaN(Number(raw))) n = Number(raw);
  if (n === null) return INVOICE_PLACEHOLDER;
  const cur =
    typeof row.currency === "string" && row.currency.trim() ? row.currency.trim() : "CHF";
  try {
    return new Intl.NumberFormat("fr-CH", { style: "currency", currency: cur }).format(n);
  } catch {
    return `${n.toFixed(2)} ${cur}`;
  }
}

function formatInvoiceStatusFr(row: Record<string, unknown>): string {
  const raw = row.status ?? row.payment_status;
  const s = String(raw ?? "")
    .toLowerCase()
    .replace(/-/g, "_");
  const map: Record<string, string> = {
    paid: "Payée",
    partially_paid: "Partiellement payée",
    sent: "Envoyée",
    draft: "Brouillon",
    overdue: "En retard",
    cancelled: "Annulée",
    void: "Annulée",
    unpaid: "Impayée",
    pending: "En attente",
  };
  if (!s) return INVOICE_PLACEHOLDER;
  return (map[s] ?? String(raw ?? "").trim()) || INVOICE_PLACEHOLDER;
}

function formatInvoiceDateFr(iso: unknown): string {
  if (typeof iso !== "string" || !iso.trim()) return INVOICE_PLACEHOLDER;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return INVOICE_PLACEHOLDER;
  return d.toLocaleDateString("fr-CH", { day: "2-digit", month: "short", year: "numeric" });
}

function extractInvoiceBookingIds(row: Record<string, unknown>): number[] {
  const raw = row.booking_ids ?? row.bookingIds;
  if (Array.isArray(raw) && raw.length > 0) {
    const out: number[] = [];
    const seen = new Set<number>();
    for (const x of raw) {
      const n =
        typeof x === "number" && Number.isFinite(x)
          ? x
          : typeof x === "string" && x.trim() && !Number.isNaN(Number(x.trim()))
            ? Number(x.trim())
            : null;
      if (n == null || seen.has(n)) continue;
      seen.add(n);
      out.push(n);
    }
    if (out.length > 0) return out;
  }
  for (const key of ["booking_id", "mission_id", "reservation_id"]) {
    const v = row[key];
    if (typeof v === "number" && Number.isFinite(v)) return [v];
    if (typeof v === "string" && v.trim() && !Number.isNaN(Number(v.trim())))
      return [Number(v.trim())];
  }
  return [];
}

function extractInvoiceBookingLabel(row: Record<string, unknown>): string {
  const ids = extractInvoiceBookingIds(row);
  if (ids.length === 0) return INVOICE_PLACEHOLDER;
  if (ids.length === 1) return `Course n° ${ids[0]}`;
  return `Courses n° ${ids.join(", ")}`;
}

function invoiceTitleLabel(row: Record<string, unknown>): string {
  const num = extractInvoiceValue(row, "invoice_number", "number");
  if (num !== "n/a") return `Facture ${num}`;
  const idRaw = row.id ?? row.invoice_id;
  if (typeof idRaw === "number" && Number.isFinite(idRaw)) return `Facture ${idRaw}`;
  if (typeof idRaw === "string" && idRaw.trim()) return `Facture ${idRaw.trim()}`;
  return "Facture";
}

/** N° seul (ex. EM-2026-01-0014) pour la ligne « Facture · … » sous le nom client. */
function invoiceNumberDisplay(row: Record<string, unknown>): string {
  const num = extractInvoiceValue(row, "invoice_number", "number");
  if (num !== "n/a") return num;
  const idRaw = row.id ?? row.invoice_id;
  if (typeof idRaw === "number" && Number.isFinite(idRaw)) return String(idRaw);
  if (typeof idRaw === "string" && idRaw.trim()) return idRaw.trim();
  return INVOICE_PLACEHOLDER;
}

function extractInvoiceId(row: Record<string, unknown>): number | null {
  const idRaw = row.id ?? row.invoice_id;
  if (typeof idRaw === "number" && Number.isFinite(idRaw)) return idRaw;
  if (typeof idRaw === "string" && idRaw.trim()) {
    const n = Number(idRaw.trim());
    if (Number.isFinite(n)) return n;
  }
  return null;
}

/** Millisecondes pour tri : dernière facture émise en tête de liste. */
function invoiceIssuedAtSortMs(row: Record<string, unknown>): number {
  for (const key of ["issued_at", "created_at", "updated_at"] as const) {
    const v = row[key];
    if (typeof v === "string" && v.trim()) {
      const t = new Date(v).getTime();
      if (!Number.isNaN(t)) return t;
    }
  }
  return 0;
}

function compareInvoicesNewestFirst(a: Record<string, unknown>, b: Record<string, unknown>): number {
  const tb = invoiceIssuedAtSortMs(b);
  const ta = invoiceIssuedAtSortMs(a);
  if (tb !== ta) return tb - ta;
  const idb = extractInvoiceId(b);
  const ida = extractInvoiceId(a);
  if (idb != null && ida != null && idb !== ida) return idb - ida;
  if (idb != null && ida == null) return 1;
  if (ida != null && idb == null) return -1;
  return 0;
}

/** Clé stable pour l’état « déplié » (préfère l’id API, sinon le n° affiché). */
function invoiceRowStableKey(row: Record<string, unknown>, index: number): string {
  const id = extractInvoiceId(row);
  if (id != null) return `id:${id}`;
  const num = extractInvoiceValue(row, "invoice_number", "number");
  if (num !== "n/a") return `num:${num}`;
  return `idx:${index}`;
}

function invoiceInitialsFromInvoiceNumber(row: Record<string, unknown>): string {
  const num = extractInvoiceValue(row, "invoice_number", "number");
  if (num === "n/a") return "Fx";
  const letters = num.replace(/[^A-Za-zÀ-ÿ]/g, "").toUpperCase();
  if (letters.length >= 2) return letters.slice(0, 2);
  if (letters.length === 1) return `${letters}${letters}`;
  const digits = num.replace(/\D/g, "");
  if (digits.length >= 2) return digits.slice(-2);
  return "Fx";
}

/** Initiales client si connu, sinon dérivées du n° de facture. */
function invoiceSummaryInitials(row: Record<string, unknown>): string {
  const clientLbl = extractInvoiceClientDisplay(row);
  if (clientLbl.trim()) return clientInitialsFromLabel(clientLbl);
  return invoiceInitialsFromInvoiceNumber(row);
}

function invoiceStatusBadgeClasses(row: Record<string, unknown>): { shell: object[]; label: object[] } {
  const raw = String(row.status ?? row.payment_status ?? "")
    .toLowerCase()
    .replace(/-/g, "_");

  if (raw === "draft") {
    return {
      shell: [styles.dispatchRefBadgeShell, styles.invoiceRefBadgeShellDraft],
      label: [styles.dispatchRefBadgeLabel, styles.invoiceRefBadgeLabelDraft],
    };
  }
  if (isInvoicePastDue(row)) {
    return {
      shell: [styles.dispatchRefBadgeShell, styles.invoiceRefBadgeShellDebt],
      label: [styles.dispatchRefBadgeLabel, styles.invoiceRefBadgeLabelDebt],
    };
  }
  if (raw === "paid" || raw === "partially_paid") {
    return {
      shell: [styles.dispatchRefBadgeShell, styles.invoiceRefBadgeShellPaid],
      label: [styles.dispatchRefBadgeLabel, styles.invoiceRefBadgeLabelPaid],
    };
  }
  if (raw === "sent") {
    return {
      shell: [styles.dispatchRefBadgeShell, styles.invoiceRefBadgeShellSent],
      label: [styles.dispatchRefBadgeLabel, styles.invoiceRefBadgeLabelSent],
    };
  }
  return {
    shell: [styles.dispatchRefBadgeShell],
    label: [styles.dispatchRefBadgeLabel],
  };
}

function isInvoicePastDue(row: Record<string, unknown>): boolean {
  const raw = String(row.status ?? row.payment_status ?? "")
    .toLowerCase()
    .replace(/-/g, "_");
  if (raw === "cancelled" || raw === "void" || raw === "canceled") return false;
  if (raw === "draft") return false;
  if (raw === "paid") return false;

  if (raw === "overdue") return true;

  const balRaw = row.balance_due;
  let bal: number | null = null;
  if (typeof balRaw === "number" && Number.isFinite(balRaw)) bal = balRaw;
  else if (typeof balRaw === "string" && balRaw.trim() && !Number.isNaN(Number(balRaw)))
    bal = Number(balRaw);
  if (bal == null || bal <= 0.009) return false;

  const dueRaw = row.due_date;
  if (typeof dueRaw !== "string" || !dueRaw.trim()) return false;
  const due = new Date(dueRaw);
  if (Number.isNaN(due.getTime())) return false;

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const dueDay = new Date(due);
  dueDay.setHours(0, 0, 0, 0);

  return dueDay.getTime() < today.getTime();
}

function resolveUnpaid(row: Record<string, unknown>): boolean {
  const statusRaw = String(row.status ?? row.payment_status ?? "").toLowerCase();
  if (statusRaw === "unpaid" || statusRaw === "pending" || statusRaw === "overdue") return true;
  const bal = row.balance_due;
  if (typeof bal === "number" && bal > 0.009) return true;
  return false;
}

/** Factures annulées : masquées du registre mobile (lecture seule). */
function isInvoiceCancelled(row: Record<string, unknown>): boolean {
  const s = String(row.status ?? row.payment_status ?? "")
    .toLowerCase()
    .replace(/-/g, "_");
  return s === "cancelled" || s === "void" || s === "canceled";
}

type CompanyReadonlySection = "clients" | "invoices";
type InvoiceKpiQuickFilter = "all" | "unpaid" | "overdue";

/** Attente lorsque les deux espaces « clients » et « facturation » sont désactivés par configuration. */
function ClientsAndBillingDisabledVisual() {
  return (
    <View style={styles.disabledInner}>
      <View style={styles.disabledHeroBand}>
        <View style={styles.disabledDualIcons}>
          <View style={styles.disabledIconPairItem}>
            <View style={[styles.disabledMiniWell, styles.disabledMiniWellBrand]}>
              <Ionicons name="people-outline" size={24} color={E.BRAND} />
            </View>
            <AppText variant="caption" style={styles.disabledMiniLabel}>
              Clients
            </AppText>
          </View>
          <View style={styles.disabledIconPairPlus}>
            <Ionicons name="add" size={14} color={E.TEXT_MUTED} />
          </View>
          <View style={styles.disabledIconPairItem}>
            <View style={[styles.disabledMiniWell, styles.disabledMiniWellSky]}>
              <Ionicons name="receipt-outline" size={24} color="#0284c7" />
            </View>
            <AppText variant="caption" style={styles.disabledMiniLabel}>
              Facturation
            </AppText>
          </View>
        </View>
        <View style={styles.disabledHeroText}>
          <AppText variant="sectionTitle" style={styles.disabledTitle}>
            Clients et facturation
          </AppText>
          <AppText variant="caption" style={styles.disabledSubtitle}>
            Un seul espace pour parcourir vos dossiers clients et suivre vos factures en lecture seule,
            aligné sur le portail entreprise.
          </AppText>
          <View style={styles.disabledUnifiedChip}>
            <Ionicons name="layers-outline" size={14} color={E.BRAND_DARK} />
            <AppText variant="caption" style={styles.disabledUnifiedChipText}>
              Vue unifiée · deux onglets (Clients / Facturation)
            </AppText>
          </View>
        </View>
      </View>

      <View style={styles.disabledCard}>
        <View style={styles.disabledCardHeader}>
          <View style={styles.disabledLockWell}>
            <Ionicons name="lock-closed-outline" size={16} color={E.TEXT_SEC} />
          </View>
          <View style={{ flex: 1, minWidth: 0 }}>
            <AppText variant="body" style={styles.disabledCardTitle}>
              Bientôt disponible sur cet environnement
            </AppText>
            <AppText variant="caption" style={styles.disabledCardIntro}>
              Les listes, filtres et tableaux du web seront proposés ici dès activation des options
              ci-dessous.
            </AppText>
          </View>
        </View>

        <View style={styles.disabledPreviewGrid}>
          <View style={styles.disabledPreviewTile}>
            <View style={styles.disabledPreviewTileHead}>
              <Ionicons name="person-outline" size={18} color={E.BRAND} />
              <View style={styles.disabledPillSoft}>
                <AppText variant="caption" style={styles.disabledPillSoftText}>
                  À activer
                </AppText>
              </View>
            </View>
            <AppText variant="body" style={styles.disabledPreviewTileTitle}>
              Clients
            </AppText>
            <AppText variant="caption" style={styles.disabledPreviewTileHint}>
              Annuaire, institutions, recherche et fiche détail en lecture seule.
            </AppText>
            <AppText variant="caption" style={styles.disabledFlagKey} selectable={Platform.OS === "web"}>
              company_mobile_clients_readonly_enabled
            </AppText>
          </View>
          <View style={styles.disabledPreviewTile}>
            <View style={styles.disabledPreviewTileHead}>
              <Ionicons name="document-text-outline" size={18} color="#0284c7" />
              <View style={styles.disabledPillSoft}>
                <AppText variant="caption" style={styles.disabledPillSoftText}>
                  À activer
                </AppText>
              </View>
            </View>
            <AppText variant="body" style={styles.disabledPreviewTileTitle}>
              Facturation
            </AppText>
            <AppText variant="caption" style={styles.disabledPreviewTileHint}>
              Registre des factures, statuts, filtres par période et solde.
            </AppText>
            <AppText variant="caption" style={styles.disabledFlagKey} selectable={Platform.OS === "web"}>
              company_mobile_invoices_readonly_enabled
            </AppText>
          </View>
        </View>
      </View>

      <View style={styles.disabledInfoStrip}>
        <View style={styles.disabledInfoIconCircle}>
          <Ionicons name="information-circle-outline" size={22} color={E.BRAND_DARK} />
        </View>
        <View style={{ flex: 1, minWidth: 0, gap: 6 }}>
          <AppText variant="body" style={styles.disabledInfoLead}>
            Configuration
          </AppText>
          <AppText variant="caption" style={styles.disabledInfoText}>
            Ces écrans sont pilotés par les feature flags de l’application. Leur activation relève de votre
            configuration produit ou technique pour cet environnement (build, tenant, etc.).
          </AppText>
        </View>
      </View>
    </View>
  );
}

/** Premier chargement liste clients ; le reste via « Charger plus » ou la recherche. */
const CLIENT_LIST_PAGE_SIZE = 8;

/** Même écart vertical entre cartes que l’onglet Clients (gabarit dispatch). */
const READONLY_LIST_CARD_STACK_GAP = 4;

/** Écran entreprise : clients et facturation (lecture seule selon flags). */
export default function CompanyClientsAndBillingScreen() {
  const params = useLocalSearchParams<{ section?: string }>();
  const requestedSection = params.section === "invoices" ? "invoices" : "clients";
  const clientsReadonlyEnabled = isFeatureEnabled("company_mobile_clients_readonly_enabled");
  const invoicesReadonlyEnabled = isFeatureEnabled("company_mobile_invoices_readonly_enabled");
  const [activeSection, setActiveSection] = useState<CompanyReadonlySection>(requestedSection);
  const [clientSearch, setClientSearch] = useState("");
  const [clientListLimit, setClientListLimit] = useState(CLIENT_LIST_PAGE_SIZE);
  const [invoiceSearch, setInvoiceSearch] = useState("");
  const [invoiceKpiFilter, setInvoiceKpiFilter] = useState<InvoiceKpiQuickFilter>("all");
  const [expandedClientId, setExpandedClientId] = useState<number | null>(null);
  const [expandedInvoiceKey, setExpandedInvoiceKey] = useState<string | null>(null);
  const clientsQuery = useCompanyClientsReadonlyQuery({
    q: clientSearch,
    limit: clientListLimit,
    page: 1,
  });
  const invoicesQuery = useCompanyInvoicesReadonlyQuery({ limit: 500 });
  const detailQuery = useCompanyClientReadonlyDetailQuery(expandedClientId);
  const rawClientRows = clientsQuery.data?.rows ?? [];
  const clientsTotal = clientsQuery.data?.total;
  const rows = useMemo(() => {
    const list = clientsQuery.data?.rows ?? [];
    return [...list].sort((a, b) =>
      extractClientLabel(a as Record<string, unknown>).localeCompare(
        extractClientLabel(b as Record<string, unknown>),
        "fr",
        { sensitivity: "base" }
      )
    );
  }, [clientsQuery.data]);
  const canLoadMoreClients =
    rawClientRows.length > 0 &&
    (clientsTotal != null ? rows.length < clientsTotal : rawClientRows.length >= clientListLimit);
  const invoiceRowsActive = useMemo(() => {
    const raw = invoicesQuery.data ?? [];
    const active = raw.filter((row) => !isInvoiceCancelled(row as Record<string, unknown>));
    return [...active].sort((a, b) =>
      compareInvoicesNewestFirst(a as Record<string, unknown>, b as Record<string, unknown>)
    );
  }, [invoicesQuery.data]);
  const invoiceRowsFiltered = useMemo(() => {
    const q = invoiceSearch.trim().toLowerCase();
    if (!q) return invoiceRowsActive;
    return invoiceRowsActive.filter((row) => {
      const clientLbl = extractInvoiceClientDisplay(row);
      const ids = extractInvoiceBookingIds(row);
      const blob = [
        extractInvoiceValue(row, "invoice_number", "number", "id", "invoice_id"),
        extractInvoiceValue(row, "status", "payment_status"),
        formatInvoiceStatusFr(row),
        ids.length ? ids.join(" ") : extractInvoiceValue(row, "booking_id", "mission_id", "reservation_id"),
        clientLbl,
      ]
        .join(" ")
        .toLowerCase();
      return blob.includes(q);
    });
  }, [invoiceRowsActive, invoiceSearch]);
  const unpaidCount = useMemo(
    () => invoiceRowsFiltered.filter((row) => resolveUnpaid(row)).length,
    [invoiceRowsFiltered]
  );
  const overdueCount = useMemo(
    () =>
      invoiceRowsFiltered.filter((row) => isInvoicePastDue(row as Record<string, unknown>)).length,
    [invoiceRowsFiltered]
  );
  const invoiceRowsForList = useMemo(() => {
    if (invoiceKpiFilter === "all") return invoiceRowsFiltered;
    if (invoiceKpiFilter === "unpaid") {
      return invoiceRowsFiltered.filter((row) => resolveUnpaid(row));
    }
    return invoiceRowsFiltered.filter((row) => isInvoicePastDue(row as Record<string, unknown>));
  }, [invoiceRowsFiltered, invoiceKpiFilter]);
  const t = useResponsiveTokens();

  const toggleClientExpanded = (clientId: number) => {
    setExpandedClientId((cur) => (cur === clientId ? null : clientId));
  };

  const toggleInvoiceExpanded = (key: string) => {
    setExpandedInvoiceKey((cur) => (cur === key ? null : key));
  };

  const onInvoiceKpiQuickFilterPress = (target: InvoiceKpiQuickFilter) => {
    if (target === "all") {
      setInvoiceKpiFilter("all");
      return;
    }
    setInvoiceKpiFilter((cur) => (cur === target ? "all" : target));
  };

  useEffect(() => {
    setActiveSection(requestedSection);
  }, [requestedSection]);

  useEffect(() => {
    setClientListLimit(CLIENT_LIST_PAGE_SIZE);
  }, [clientSearch]);

  useEffect(() => {
    if (activeSection !== "invoices") setInvoiceKpiFilter("all");
  }, [activeSection]);

  useEffect(() => {
    setExpandedInvoiceKey(null);
  }, [activeSection, invoiceSearch, invoiceKpiFilter]);

  useEffect(() => {
    setInvoiceKpiFilter("all");
  }, [invoiceSearch]);

  if (!clientsReadonlyEnabled && !invoicesReadonlyEnabled) {
    return (
      <PermissionGuard permission="company:dashboard:read">
        <Screen
          scroll
          backgroundColor={E.BG}
          withHorizontalPadding
          contentContainerStyle={styles.disabledScrollContent}
        >
          <ClientsAndBillingDisabledVisual />
        </Screen>
      </PermissionGuard>
    );
  }

  const contentPad = {
    paddingTop: 12,
    paddingBottom: 28,
    gap: 16,
  };

  return (
    <PermissionGuard permission="company:dashboard:read">
      <Screen scroll backgroundColor={E.BG} withHorizontalPadding contentContainerStyle={contentPad}>
        <View style={styles.enabledHero}>
          <View style={styles.enabledHeroIconWell}>
            <Ionicons name="reader-outline" size={26} color={E.BRAND} />
          </View>
          <View style={styles.enabledHeroCol}>
            <AppText variant="sectionTitle" style={styles.enabledTitle}>
              Clients et facturation
            </AppText>
            <AppText variant="caption" style={styles.enabledSubtitle}>
              Consultation en lecture seule — basculez entre l’annuaire clients et le registre des factures.
            </AppText>
          </View>
        </View>

        <View style={styles.segmentWrap}>
          <Pressable
            onPress={() => setActiveSection("clients")}
            style={({ pressed }) => [
              styles.segmentBtn,
              activeSection === "clients" && styles.segmentBtnOn,
              pressed && styles.segmentPressed,
            ]}
            accessibilityState={{ selected: activeSection === "clients" }}
          >
            <Ionicons
              name="people-outline"
              size={18}
              color={activeSection === "clients" ? E.BRAND : E.TEXT_SEC}
            />
            <AppText
              variant="body"
              style={[
                styles.segmentLabel,
                activeSection === "clients" ? styles.segmentLabelOn : styles.segmentLabelOff,
              ]}
            >
              Clients
            </AppText>
          </Pressable>
          <Pressable
            onPress={() => setActiveSection("invoices")}
            style={({ pressed }) => [
              styles.segmentBtn,
              activeSection === "invoices" && styles.segmentBtnOn,
              pressed && styles.segmentPressed,
            ]}
            accessibilityState={{ selected: activeSection === "invoices" }}
          >
            <Ionicons
              name="receipt-outline"
              size={18}
              color={activeSection === "invoices" ? E.BRAND : E.TEXT_SEC}
            />
            <AppText
              variant="body"
              style={[
                styles.segmentLabel,
                activeSection === "invoices" ? styles.segmentLabelOn : styles.segmentLabelOff,
              ]}
            >
              Facturation
            </AppText>
          </Pressable>
        </View>

        {activeSection === "clients" ? (
          <>
            {!clientsReadonlyEnabled ? (
              <AppCard variant="surface">
                <AppText variant="sectionTitle">Clients (lecture seule)</AppText>
                <AppText variant="bodyMuted" style={{ marginTop: t.fieldGap }}>
                  Cette section est désactivée par feature flag.
                </AppText>
              </AppCard>
            ) : (
              <>
                <AppInput
                  value={clientSearch}
                  onChangeText={setClientSearch}
                  placeholder="Rechercher un client"
                />
                {!clientsQuery.isLoading && rows.length > 0 ? (
                  <AppText variant="caption" style={styles.clientListCount}>
                    {clientsTotal != null
                      ? rows.length === 1
                        ? `1 client affiché sur ${clientsTotal}`
                        : `${rows.length} clients affichés sur ${clientsTotal}`
                      : rows.length === 1
                        ? "1 client affiché"
                        : `${rows.length} clients affichés`}
                  </AppText>
                ) : null}
                {clientsQuery.isLoading ? <AppSpinner size="small" /> : null}
                <View style={styles.clientListStack}>
                  {rows.map((row, index) => {
                    const id = extractClientId(row);
                    if (id == null) return null;
                    const expanded = expandedClientId === id;
                    const institution = isInstitutionClientRow(row);
                    const label = extractClientLabel(row);
                    const contactName = extractClientPersonName(row);
                    const a11yContact =
                      institution && contactName
                        ? `, personne de contact ${contactName}`
                        : "";
                    const subtitle = extractClientListSubtitle(row);
                    const initials = clientInitialsFromLabel(label);
                    const detail = detailQuery.data as Record<string, unknown> | undefined;
                    const detailIdMatch = detail ? detailPayloadId(detail) === id : false;
                    const showDetailLoading =
                      expanded && (detailQuery.isPending || (detailQuery.isFetching && !detailIdMatch));
                    return (
                      <View
                        key={`${id}-${index}`}
                        style={styles.dispatchClientCard}
                      >
                        <TouchableOpacity
                          onPress={() => toggleClientExpanded(id)}
                          activeOpacity={0.85}
                          style={[styles.clientCardSummaryPressable, DISPATCH_SUMMARY_TOUCHABLE_WEB]}
                          accessibilityRole="button"
                          accessibilityLabel={
                            expanded
                              ? `Replier la fiche de ${label}${a11yContact}, référence ${id}`
                              : `Afficher la fiche de ${label}${a11yContact}, référence ${id}`
                          }
                          accessibilityState={{ expanded }}
                        >
                          <View style={styles.dispatchSummaryRow}>
                            <View style={styles.dispatchTimeCol}>
                              <AppText style={styles.dispatchTimeText} numberOfLines={1}>
                                {initials}
                              </AppText>
                            </View>
                            <View style={styles.dispatchSummaryMain}>
                              <View style={styles.dispatchNameTap}>
                                <AppText
                                  style={styles.dispatchClientName}
                                  numberOfLines={1}
                                  ellipsizeMode="tail"
                                >
                                  {label}
                                </AppText>
                              </View>
                              <View style={styles.dispatchBadgeCol}>
                                <View
                                  style={[
                                    styles.dispatchRefBadgeShell,
                                    institution && styles.dispatchRefBadgeShellInstitution,
                                  ]}
                                >
                                  <AppText
                                    style={[
                                      styles.dispatchRefBadgeLabel,
                                      institution && styles.dispatchRefBadgeLabelInstitution,
                                    ]}
                                    numberOfLines={1}
                                    ellipsizeMode="tail"
                                  >
                                    Réf. {id}
                                  </AppText>
                                </View>
                              </View>
                            </View>
                            <View style={styles.dispatchChevronWrap}>
                              <Ionicons
                                name={expanded ? "chevron-up-outline" : "chevron-down-outline"}
                                size={16}
                                color={E.TEXT_MUTED}
                              />
                            </View>
                          </View>
                        </TouchableOpacity>
                        {subtitle ? (
                          <AppText
                            style={styles.dispatchSubtitle}
                            numberOfLines={1}
                            ellipsizeMode="tail"
                          >
                            {subtitle}
                          </AppText>
                        ) : null}
                        {expanded ? (
                          <View style={styles.clientExpandedBlock}>
                            {showDetailLoading ? (
                              <View style={styles.clientExpandedLoading}>
                                <AppSpinner size="small" />
                                <AppText variant="caption" style={styles.clientExpandedLoadingHint}>
                                  Chargement…
                                </AppText>
                              </View>
                            ) : null}
                            {detailQuery.error && expanded ? (
                              <AppText variant="error" style={styles.clientExpandedError}>
                                {detailQuery.error instanceof Error
                                  ? detailQuery.error.message
                                  : "Fiche client indisponible."}
                              </AppText>
                            ) : null}
                            {detail && detailIdMatch ? (
                              <View style={styles.clientExpandedRoutes}>
                                <ClientExpandRouteRow
                                  icon={isInstitutionClientRow(detail) ? "business-outline" : "person-outline"}
                                  lines={[
                                    {
                                      label: "Type de client",
                                      value: formatClientTypeDetailValue(detail),
                                    },
                                  ]}
                                />
                                {isInstitutionClientRow(detail) && extractClientPersonName(detail) ? (
                                  <ClientExpandRouteRow
                                    icon="person-outline"
                                    lines={[
                                      {
                                        label: "Personne de contact",
                                        value: extractClientPersonName(detail) ?? "",
                                      },
                                    ]}
                                  />
                                ) : null}
                                <ClientExpandRouteRow
                                  icon="location-outline"
                                  lines={[
                                    {
                                      label: "Domicile",
                                      value: formatDomicileLine(detail) ?? "",
                                    },
                                  ]}
                                />
                                <ClientExpandRouteRow
                                  icon="flag-outline"
                                  lines={[
                                    {
                                      label: "Autre adresse / facturation",
                                      value: formatBillingDistinctFromDomicile(detail) ?? "",
                                    },
                                  ]}
                                />
                                <ClientExpandRouteRow
                                  icon="call-outline"
                                  lines={[
                                    {
                                      label: "Téléphone",
                                      value: formatClientPhoneDetail(detail) ?? "",
                                    },
                                  ]}
                                />
                                <ClientExpandRouteRow
                                  icon="mail-outline"
                                  lines={[
                                    {
                                      label: "E-mail",
                                      value: formatDetailEmailForDisplay(detail) ?? "",
                                    },
                                  ]}
                                />
                                <ClientExpandRouteRow
                                  icon="calendar-outline"
                                  lines={[
                                    {
                                      label: "Date de naissance",
                                      value: formatBirthDate(detail) ?? "",
                                    },
                                  ]}
                                />
                                <ClientExpandRouteRow
                                  icon="key-outline"
                                  lines={[
                                    {
                                      label: "Accès domicile",
                                      value: formatAccessDomLine(detail) ?? "",
                                    },
                                  ]}
                                />
                                <ClientExpandRouteRow
                                  icon="pricetag-outline"
                                  lines={[
                                    {
                                      label: "Tarif préférentiel",
                                      value: formatPreferentialRate(detail) ?? "",
                                    },
                                  ]}
                                />
                                <ClientExpandRouteRow
                                  icon="male-female-outline"
                                  lines={[
                                    {
                                      label: "Genre",
                                      value: formatGenderFr(detail) ?? "",
                                    },
                                  ]}
                                />
                                {!formatDomicileLine(detail) &&
                                !formatBillingDistinctFromDomicile(detail) &&
                                !formatClientPhoneDetail(detail) &&
                                !formatDetailEmailForDisplay(detail) &&
                                !formatBirthDate(detail) &&
                                !formatAccessDomLine(detail) &&
                                !formatPreferentialRate(detail) &&
                                !formatGenderFr(detail) &&
                                !(isInstitutionClientRow(detail) && extractClientPersonName(detail)) ? (
                                  <AppText variant="bodyMuted" style={styles.clientExpandedMuted}>
                                    Aucune information détaillée renvoyée pour ce client.
                                  </AppText>
                                ) : null}
                              </View>
                            ) : null}
                          </View>
                        ) : null}
                      </View>
                    );
                  })}
                </View>
                {canLoadMoreClients ? (
                  <Pressable
                    onPress={() =>
                      setClientListLimit((n) => n + CLIENT_LIST_PAGE_SIZE)
                    }
                    disabled={clientsQuery.isFetching}
                    style={({ pressed }) => [
                      styles.clientLoadMoreBtn,
                      pressed && styles.clientLoadMoreBtnPressed,
                      clientsQuery.isFetching && styles.clientLoadMoreBtnDisabled,
                    ]}
                    accessibilityRole="button"
                    accessibilityLabel="Charger plus de clients"
                    accessibilityState={{ busy: clientsQuery.isFetching }}
                  >
                    {clientsQuery.isFetching ? (
                      <AppSpinner size="small" />
                    ) : (
                      <Ionicons
                        name="chevron-down-outline"
                        size={15}
                        color={E.TEXT_MUTED}
                      />
                    )}
                    <AppText variant="caption" style={styles.clientLoadMoreLabel}>
                      Charger plus
                    </AppText>
                  </Pressable>
                ) : null}
                {!clientsQuery.isLoading && rows.length === 0 ? (
                  <AppEmptyState
                    title="Aucun client"
                    description="Aucun client ne correspond à la recherche."
                  />
                ) : null}
                {clientsQuery.error ? (
                  <AppText variant="error">
                    {clientsQuery.error instanceof Error
                      ? clientsQuery.error.message
                      : "Chargement clients impossible."}
                  </AppText>
                ) : null}
              </>
            )}
          </>
        ) : (
          <>
            {!invoicesReadonlyEnabled ? (
              <AppCard variant="surface">
                <AppText variant="sectionTitle">Factures (lecture seule)</AppText>
                <AppText variant="bodyMuted" style={{ marginTop: t.fieldGap }}>
                  Cette section est désactivée par feature flag.
                </AppText>
              </AppCard>
            ) : (
              <>
                <AppInput
                  value={invoiceSearch}
                  onChangeText={setInvoiceSearch}
                  placeholder="N°, statut, client, course…"
                />
                <View style={styles.invoiceKpiRow}>
                  <Pressable
                    onPress={() => onInvoiceKpiQuickFilterPress("all")}
                    style={({ pressed }) => [
                      styles.invoiceKpiCell,
                      styles.invoiceKpiCellNeutral,
                      invoiceKpiFilter === "all" && styles.invoiceKpiCellSelected,
                      pressed && styles.invoiceKpiCellPressed,
                      INVOICE_KPI_PRESSABLE_WEB,
                    ]}
                    accessibilityRole="button"
                    accessibilityLabel="Afficher toutes les factures de la sélection"
                    accessibilityState={{ selected: invoiceKpiFilter === "all" }}
                  >
                    <AppText variant="caption" style={styles.invoiceKpiLabel}>
                      Total affiché
                    </AppText>
                    <AppText variant="sectionTitle" style={styles.invoiceKpiValue}>
                      {invoiceRowsFiltered.length}
                    </AppText>
                  </Pressable>
                  <Pressable
                    onPress={() => onInvoiceKpiQuickFilterPress("unpaid")}
                    style={({ pressed }) => [
                      styles.invoiceKpiCell,
                      unpaidCount > 0 ? styles.invoiceKpiCellCaution : styles.invoiceKpiCellNeutral,
                      invoiceKpiFilter === "unpaid" && styles.invoiceKpiCellSelected,
                      pressed && styles.invoiceKpiCellPressed,
                      INVOICE_KPI_PRESSABLE_WEB,
                    ]}
                    accessibilityRole="button"
                    accessibilityLabel="Filtrer les factures impayées ou avec solde dû"
                    accessibilityState={{ selected: invoiceKpiFilter === "unpaid" }}
                  >
                    <AppText
                      variant="caption"
                      style={unpaidCount > 0 ? styles.invoiceKpiLabelCaution : styles.invoiceKpiLabel}
                    >
                      Impayées / solde dû
                    </AppText>
                    <AppText
                      variant="sectionTitle"
                      style={unpaidCount > 0 ? styles.invoiceKpiValueCaution : styles.invoiceKpiValue}
                    >
                      {unpaidCount}
                    </AppText>
                  </Pressable>
                  <Pressable
                    onPress={() => onInvoiceKpiQuickFilterPress("overdue")}
                    style={({ pressed }) => [
                      styles.invoiceKpiCell,
                      overdueCount > 0 ? styles.invoiceKpiCellAlert : styles.invoiceKpiCellNeutral,
                      invoiceKpiFilter === "overdue" && styles.invoiceKpiCellSelected,
                      pressed && styles.invoiceKpiCellPressed,
                      INVOICE_KPI_PRESSABLE_WEB,
                    ]}
                    accessibilityRole="button"
                    accessibilityLabel="Filtrer les factures échues"
                    accessibilityState={{ selected: invoiceKpiFilter === "overdue" }}
                  >
                    <AppText
                      variant="caption"
                      style={overdueCount > 0 ? styles.invoiceKpiLabelAlert : styles.invoiceKpiLabel}
                    >
                      Échues
                    </AppText>
                    <AppText
                      variant="sectionTitle"
                      style={overdueCount > 0 ? styles.invoiceKpiValueAlert : styles.invoiceKpiValue}
                    >
                      {overdueCount}
                    </AppText>
                  </Pressable>
                </View>
                {invoicesQuery.isLoading ? <AppSpinner size="small" /> : null}
                <View style={styles.invoiceListStack}>
                  {invoiceRowsForList.map((row, index) => {
                    const rowRec = row as Record<string, unknown>;
                    const unpaid = resolveUnpaid(row);
                    const clientLbl = extractInvoiceClientDisplay(rowRec);
                    const bookingLbl = extractInvoiceBookingLabel(row);
                    const badge = invoiceStatusBadgeClasses(row);
                    const rowKey = invoiceRowStableKey(rowRec, index);
                    const expanded = expandedInvoiceKey === rowKey;
                    const primaryTitle =
                      clientLbl.trim().length > 0 ? clientLbl : invoiceTitleLabel(rowRec);
                    return (
                      <View
                        key={rowKey}
                        style={styles.dispatchClientCard}
                      >
                        <TouchableOpacity
                          onPress={() => toggleInvoiceExpanded(rowKey)}
                          activeOpacity={0.85}
                          style={[styles.clientCardSummaryPressable, DISPATCH_SUMMARY_TOUCHABLE_WEB]}
                          accessibilityRole="button"
                          accessibilityLabel={
                            expanded
                              ? `Replier les détails de ${primaryTitle}`
                              : `Afficher les détails de ${primaryTitle}`
                          }
                          accessibilityState={{ expanded }}
                        >
                          <View style={styles.dispatchSummaryRow}>
                            <View style={styles.dispatchTimeCol}>
                              <AppText style={styles.dispatchTimeText} numberOfLines={1}>
                                {invoiceSummaryInitials(rowRec)}
                              </AppText>
                            </View>
                            <View style={styles.dispatchSummaryMain}>
                              <View style={styles.dispatchNameTap}>
                                <AppText
                                  style={styles.dispatchClientName}
                                  numberOfLines={1}
                                  ellipsizeMode="tail"
                                >
                                  {primaryTitle}
                                </AppText>
                              </View>
                              <View style={styles.dispatchBadgeCol}>
                                <View style={badge.shell}>
                                  <AppText
                                    variant="caption"
                                    style={badge.label}
                                    numberOfLines={1}
                                    ellipsizeMode="tail"
                                  >
                                    {formatInvoiceStatusFr(row)}
                                  </AppText>
                                </View>
                              </View>
                            </View>
                            <View style={styles.dispatchChevronWrap}>
                              <Ionicons
                                name={expanded ? "chevron-up-outline" : "chevron-down-outline"}
                                size={16}
                                color={E.TEXT_MUTED}
                              />
                            </View>
                          </View>
                        </TouchableOpacity>
                        {expanded ? (
                          <View style={styles.invoiceCardSubtitleStack}>
                            {clientLbl.trim().length > 0 ? (
                              <AppText variant="caption" style={styles.invoiceMetaLine} numberOfLines={1}>
                                Facture · {invoiceNumberDisplay(rowRec)}
                              </AppText>
                            ) : null}
                            <AppText variant="caption" style={styles.invoiceMetaLine}>
                              Montant · {formatInvoiceAmount(row)}
                            </AppText>
                            <AppText variant="caption" style={styles.invoiceMetaLine}>
                              Émission · {formatInvoiceDateFr(row.issued_at)}
                              {"  ·  "}
                              Échéance · {formatInvoiceDateFr(row.due_date)}
                            </AppText>
                            <AppText variant="caption" style={styles.invoiceMetaLine}>
                              {bookingLbl === INVOICE_PLACEHOLDER
                                ? "Aucune course liée en tête de facture"
                                : bookingLbl}
                            </AppText>
                            {unpaid ? (
                              <AppText variant="caption" style={styles.invoiceUnpaidHint}>
                                Solde à recouvrer
                              </AppText>
                            ) : null}
                          </View>
                        ) : null}
                      </View>
                    );
                  })}
                </View>
                {!invoicesQuery.isLoading && invoiceRowsFiltered.length === 0 ? (
                  <AppEmptyState title="Aucune facture" description="Aucune facture ne correspond à la recherche." />
                ) : null}
                {!invoicesQuery.isLoading &&
                invoiceRowsFiltered.length > 0 &&
                invoiceRowsForList.length === 0 ? (
                  <AppEmptyState
                    title="Aucune facture"
                    description={
                      invoiceKpiFilter === "unpaid"
                        ? "Aucune facture impayée dans cette sélection."
                        : invoiceKpiFilter === "overdue"
                          ? "Aucune facture échue dans cette sélection."
                          : "Aucune facture ne correspond au filtre."
                    }
                  />
                ) : null}
                {invoicesQuery.error ? (
                  <AppText variant="error">
                    {invoicesQuery.error instanceof Error
                      ? invoicesQuery.error.message
                      : "Chargement factures impossible."}
                  </AppText>
                ) : null}
              </>
            )}
          </>
        )}
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  disabledScrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingTop: 24,
    paddingBottom: 40,
    minHeight: Platform.OS === "web" ? 480 : undefined,
  },
  disabledInner: {
    gap: 20,
    width: "100%",
  },
  disabledHeroBand: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 16,
    padding: 16,
    backgroundColor: E.CARD,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    ...cardShadow,
  },
  disabledDualIcons: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  disabledIconPairItem: { alignItems: "center", gap: 6, width: 72 },
  disabledIconPairPlus: { paddingHorizontal: 4, marginTop: -18 },
  disabledMiniWell: {
    width: 56,
    height: 56,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
  },
  disabledMiniWellBrand: {
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderColor: "rgba(0, 121, 107, 0.2)",
  },
  disabledMiniWellSky: {
    backgroundColor: "rgba(14, 165, 233, 0.1)",
    borderColor: "rgba(14, 165, 233, 0.25)",
  },
  disabledMiniLabel: {
    color: E.TEXT_SEC,
    fontWeight: "600",
    fontSize: 11,
  },
  disabledHeroText: { flex: 1, minWidth: 0, gap: 8 },
  disabledTitle: {
    fontSize: 21,
    fontWeight: "700",
    color: E.TEXT,
    letterSpacing: 0.15,
  },
  disabledSubtitle: {
    color: E.TEXT_SEC,
    lineHeight: 21,
    fontSize: 14,
  },
  disabledUnifiedChip: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 6,
    marginTop: 4,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 10,
    backgroundColor: "rgba(0, 121, 107, 0.07)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.15)",
  },
  disabledUnifiedChipText: {
    color: E.BRAND_DARK,
    fontWeight: "600",
    fontSize: 12,
  },
  disabledCard: {
    backgroundColor: E.CARD,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    padding: 16,
    ...cardShadow,
  },
  disabledCardHeader: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    marginBottom: 14,
  },
  disabledLockWell: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: "rgba(148, 163, 184, 0.12)",
    alignItems: "center",
    justifyContent: "center",
  },
  disabledCardTitle: { fontWeight: "700", color: E.TEXT, fontSize: 16 },
  disabledCardIntro: {
    color: E.TEXT_SEC,
    lineHeight: 19,
    marginTop: 4,
  },
  disabledPreviewGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
  },
  disabledPreviewTile: {
    flex: 1,
    minWidth: 148,
    padding: 12,
    borderRadius: 12,
    backgroundColor: "rgba(248, 250, 252, 0.95)",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.28)",
    gap: 8,
  },
  disabledPreviewTileHead: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  disabledPreviewTileTitle: { fontWeight: "700", color: E.TEXT, fontSize: 15 },
  disabledPreviewTileHint: {
    color: E.TEXT_MUTED,
    lineHeight: 18,
    fontSize: 12,
  },
  disabledPillSoft: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
    backgroundColor: "rgba(148, 163, 184, 0.15)",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.3)",
  },
  disabledPillSoftText: {
    fontSize: 10,
    fontWeight: "800",
    color: E.TEXT_SEC,
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },
  disabledFlagKey: {
    fontFamily: Platform.select({ ios: "Menlo", android: "monospace", default: "monospace" }),
    fontSize: 10,
    color: E.TEXT_MUTED,
    marginTop: 4,
    lineHeight: 14,
  },
  disabledInfoStrip: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 14,
    padding: 16,
    borderRadius: 14,
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.18)",
  },
  disabledInfoIconCircle: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    alignItems: "center",
    justifyContent: "center",
  },
  disabledInfoLead: {
    fontWeight: "700",
    color: E.BRAND_DARK,
    fontSize: 14,
  },
  disabledInfoText: {
    color: E.TEXT_SEC,
    lineHeight: 20,
    fontWeight: "500",
  },
  enabledHero: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    paddingVertical: 4,
  },
  enabledHeroIconWell: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: E.CARD,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    alignItems: "center",
    justifyContent: "center",
    ...cardShadow,
  },
  enabledHeroCol: { flex: 1, minWidth: 0, gap: 6 },
  enabledTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: E.TEXT,
    letterSpacing: 0.2,
  },
  enabledSubtitle: {
    color: E.TEXT_SEC,
    lineHeight: 18,
  },
  segmentWrap: {
    flexDirection: "row",
    gap: 8,
    backgroundColor: E.CARD,
    borderRadius: 14,
    padding: 6,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    ...cardShadow,
  },
  segmentBtn: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 10,
    paddingHorizontal: 8,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "transparent",
  },
  segmentBtnOn: {
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    borderColor: "rgba(0, 121, 107, 0.22)",
  },
  segmentPressed: { opacity: 0.88 },
  segmentLabel: { fontWeight: "700", fontSize: 14 },
  segmentLabelOn: { color: E.BRAND_DARK },
  segmentLabelOff: { color: brandTextMuted },
  invoiceKpiRow: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  invoiceKpiCell: {
    flexGrow: 1,
    flexBasis: 0,
    minWidth: 92,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderWidth: 1,
  },
  invoiceKpiCellNeutral: {
    backgroundColor: E.CARD,
    borderColor: E.SHELL_BORDER,
  },
  invoiceKpiCellAlert: {
    backgroundColor: "#FFF7F7",
    borderColor: "#FECACA",
  },
  invoiceKpiCellCaution: {
    backgroundColor: "#FFFBEB",
    borderColor: "rgba(245, 158, 11, 0.35)",
  },
  invoiceKpiLabel: { color: E.TEXT_SEC, fontWeight: "600", fontSize: 12 },
  invoiceKpiLabelAlert: { color: "#B42318", fontWeight: "600", fontSize: 12 },
  invoiceKpiLabelCaution: { color: "#B45309", fontWeight: "600", fontSize: 12 },
  invoiceKpiValue: { marginTop: 4, color: E.TEXT, fontSize: 18, fontWeight: "700" },
  invoiceKpiValueAlert: { marginTop: 4, color: "#B42318", fontSize: 18, fontWeight: "700" },
  invoiceKpiValueCaution: { marginTop: 4, color: "#B45309", fontSize: 18, fontWeight: "700" },
  invoiceKpiCellSelected: {
    borderColor: E.BRAND,
    borderWidth: 2,
  },
  invoiceKpiCellPressed: {
    opacity: 0.9,
  },
  invoiceListStack: { gap: READONLY_LIST_CARD_STACK_GAP },
  invoiceCardSubtitleStack: {
    marginTop: 6,
    marginLeft: 41,
    gap: 4,
  },
  invoiceRefBadgeShellDebt: {
    borderColor: "rgba(220, 38, 38, 0.4)",
    backgroundColor: "rgba(254, 242, 242, 0.9)",
  },
  invoiceRefBadgeLabelDebt: {
    color: "#B42318",
  },
  invoiceRefBadgeShellDraft: {
    borderColor: "rgba(100, 116, 139, 0.4)",
    backgroundColor: "rgba(241, 245, 249, 0.95)",
  },
  invoiceRefBadgeLabelDraft: {
    color: "#475569",
  },
  invoiceRefBadgeShellSent: {
    borderColor: "rgba(2, 132, 199, 0.45)",
    backgroundColor: "rgba(224, 242, 254, 0.95)",
  },
  invoiceRefBadgeLabelSent: {
    color: "#0369a1",
  },
  invoiceRefBadgeShellPaid: {
    borderColor: "rgba(22, 163, 74, 0.4)",
    backgroundColor: "rgba(220, 252, 231, 0.9)",
  },
  invoiceRefBadgeLabelPaid: {
    color: "#15803d",
  },
  invoiceMetaLine: { color: E.TEXT_SEC, fontSize: 13, lineHeight: 18, fontWeight: "400" },
  invoiceUnpaidHint: {
    marginTop: 8,
    marginLeft: 41,
    fontSize: 12,
    fontWeight: "700",
    color: E.DANGER,
  },
  clientListCount: {
    color: E.TEXT_SEC,
    fontWeight: "600",
    marginTop: -6,
    marginBottom: 0,
  },
  clientLoadMoreBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    marginTop: 4,
    paddingVertical: 9,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: BORDER_SLATE,
    backgroundColor: "transparent",
  },
  clientLoadMoreBtnPressed: {
    backgroundColor: "rgba(100, 116, 139, 0.07)",
    borderColor: "rgba(148, 163, 184, 0.32)",
  },
  clientLoadMoreBtnDisabled: {
    opacity: 0.55,
  },
  clientLoadMoreLabel: {
    fontSize: 14,
    fontWeight: "500",
    letterSpacing: 0.15,
    color: E.TEXT_SEC,
  },
  clientListStack: {
    gap: READONLY_LIST_CARD_STACK_GAP,
  },
  /** === Gabarit `DispatchRideListCard` (carte + ligne résumé) === */
  dispatchClientCard: {
    backgroundColor: E.CARD,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: E.BORDER,
    ...clientRowCardShadow,
  },
  dispatchSummaryRow: {
    flexDirection: "row",
    alignItems: "center",
    minWidth: 0,
  },
  dispatchTimeCol: {
    width: 38,
    minHeight: 32,
    marginRight: 3,
    alignItems: "flex-start",
    justifyContent: "center",
    gap: 2,
  },
  dispatchTimeText: {
    color: E.BRAND,
    fontWeight: "700",
    fontSize: 15,
    letterSpacing: 0.2,
    lineHeight: 20,
  },
  dispatchSummaryMain: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    minWidth: 0,
  },
  dispatchNameTap: {
    flex: 1,
    minWidth: 0,
    marginRight: 0,
    flexDirection: "row",
    alignItems: "center",
  },
  /** Titre : occupe l’espace restant (évite la largeur fixe trop étroite du dispatch). */
  dispatchClientName: {
    flex: 1,
    minWidth: 0,
    fontWeight: "600",
    fontSize: 14,
    color: E.TEXT,
    marginRight: 6,
    lineHeight: 20,
  },
  dispatchBadgeCol: {
    flexShrink: 0,
    alignItems: "flex-end",
  },
  dispatchRefBadgeShell: {
    width: CLIENT_REF_BADGE_W,
    height: CLIENT_REF_BADGE_H,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(107, 114, 128, 0.25)",
    overflow: "hidden",
    justifyContent: "center",
    alignItems: "stretch",
    paddingHorizontal: 6,
    backgroundColor: "#F3F4F6",
  },
  dispatchRefBadgeShellInstitution: {
    borderColor: "rgba(10, 143, 122, 0.35)",
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  /** Aligné sur `badgeLabel` dispatch (statut gris). */
  dispatchRefBadgeLabel: {
    width: "100%",
    fontSize: 10,
    lineHeight: 16,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase",
    textAlign: "center",
    color: "#6B7280",
  },
  dispatchRefBadgeLabelInstitution: {
    color: "#0a8f7a",
  },
  dispatchChevronWrap: {
    width: 24,
    alignItems: "center",
    justifyContent: "center",
    marginLeft: 2,
    marginRight: 0,
  },
  /** Ligne optionnelle — pas sur le snippet dispatch ; alignée sous le bloc texte. */
  dispatchSubtitle: {
    marginTop: 6,
    marginLeft: 41,
    fontSize: 12,
    fontWeight: "500",
    color: E.TEXT_SEC,
    lineHeight: 16,
  },
  clientCardSummaryPressable: {
    borderRadius: 0,
  },
  clientExpandedBlock: {
    marginTop: 10,
    paddingTop: 2,
  },
  clientExpandedLoading: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 8,
  },
  clientExpandedLoadingHint: { color: E.TEXT_SEC },
  clientExpandedError: { marginBottom: 8 },
  clientExpandedRoutes: { width: "100%" },
  clientExpandRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 4,
  },
  clientExpandIcon: { marginRight: 8, marginTop: 2 },
  clientExpandTexts: { flex: 1, minWidth: 0 },
  clientExpandLineBlock: { marginBottom: 6 },
  clientExpandLabel: {
    color: E.TEXT_MUTED,
    fontSize: 11,
    fontWeight: "600",
    marginBottom: 2,
  },
  clientExpandValue: {
    color: E.TEXT_SEC,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: "400",
  },
  clientExpandedMuted: { marginTop: 4 },
});
