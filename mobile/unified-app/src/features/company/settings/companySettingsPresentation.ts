export type CompanyProfileRow = { label: string; value: string; fullWidth?: boolean };

export type CompanyProfileSection = {
  id: string;
  title: string;
  icon: "business" | "call" | "document" | "map";
  rows: CompanyProfileRow[];
};

export type CompanyProfileBadge = {
  label: string;
  tone: "active" | "inactive" | "info";
};

export type CompanyProfileViewModel = {
  displayName: string;
  logoUrl: string | null;
  badges: CompanyProfileBadge[];
  sections: CompanyProfileSection[];
  vehicleCount: number;
};

export type CompanyBillingSummary = {
  label: string;
  detail: string | null;
};

export type CompanyDispatchModeId = "manual" | "semi_auto" | "fully_auto";

export type CompanyDispatchModeOption = {
  id: CompanyDispatchModeId;
  label: string;
  hint: string;
  meta: string;
  selectable: boolean;
  lockedLabel?: string;
};

const DISPATCH_MODE_LABELS: Record<string, string> = {
  manual: "Manuel",
  semi_auto: "Semi-automatique",
  fully_auto: "Totalement automatique",
};

const DISPATCH_MODE_HINTS: Record<string, string> = {
  manual: "Vous assignez chaque course à un chauffeur.",
  semi_auto: "Lirie propose des assignations, vous confirmez.",
  fully_auto: "Lirie assigne les courses automatiquement.",
};

export const COMPANY_DISPATCH_MODE_OPTIONS: CompanyDispatchModeOption[] = [
  {
    id: "manual",
    label: "Manuel",
    hint: "Contrôle total sur chaque assignation. Suggestions en lecture seule.",
    meta: "Automatisation 0 %",
    selectable: true,
  },
  {
    id: "semi_auto",
    label: "Semi-automatique",
    hint: "Dispatch optimisé avec suggestions à valider avant application.",
    meta: "Automatisation 50–70 %",
    selectable: false,
    lockedLabel: "En développement",
  },
  {
    id: "fully_auto",
    label: "Totalement automatique",
    hint: "Application automatique des décisions haute confiance.",
    meta: "Automatisation 90–95 %",
    selectable: false,
    lockedLabel: "En développement",
  },
];

const DISPATCH_STATE_LABELS: Record<string, string> = {
  idle: "Prêt",
  running: "Optimisation en cours",
  degraded: "Service partiel",
  failed: "Interruption",
  unknown: "Indisponible",
};

export function formatCompanyField(value: string | null | undefined, fallback = "—"): string {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : fallback;
}

export function formatDispatchModeFr(mode: string | null | undefined): string {
  const key = String(mode ?? "").trim().toLowerCase();
  return DISPATCH_MODE_LABELS[key] ?? formatCompanyField(mode, "Non configuré");
}

export function formatDispatchModeHint(mode: string | null | undefined): string {
  const key = String(mode ?? "").trim().toLowerCase();
  return DISPATCH_MODE_HINTS[key] ?? "Configurez le mode depuis l’application ou le portail web.";
}

export function formatDispatchStateFr(state: string | null | undefined): string {
  const key = String(state ?? "").trim().toLowerCase();
  return DISPATCH_STATE_LABELS[key] ?? formatCompanyField(state, "—");
}

export function resolveCompanyRealtimeLabel(status: string): string {
  const s = status.toLowerCase();
  if (s === "healthy") return "Connecté";
  if (s === "connecting" || s === "reconnecting") return "Connexion en cours";
  if (s === "degraded") return "Connexion limitée";
  if (s === "failed") return "Hors ligne";
  if (s === "idle") return "En attente";
  return status;
}

export function resolveUserDisplayName(
  user: {
    full_name?: string | null;
    first_name?: string | null;
    last_name?: string | null;
    username?: string | null;
    email?: string | null;
  } | null
): string {
  if (!user) return "—";
  const full = user.full_name?.trim();
  if (full) return full;
  const fromParts = [user.first_name, user.last_name].filter(Boolean).join(" ").trim();
  if (fromParts) return fromParts;
  const username = user.username?.trim();
  if (username) return username;
  const email = user.email?.trim();
  if (email) return email;
  return "—";
}

export function resolveCompanyLogoUrl(
  logoUrl: string | null | undefined,
  apiBaseUrl: string
): string | null {
  const trimmed = logoUrl?.trim();
  if (!trimmed) return null;
  if (/^(https?:|data:|blob:)/i.test(trimmed)) return trimmed;

  const normalized = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  const apiRoot = apiBaseUrl.replace(/\/$/, "");
  const origin = apiRoot.replace(/\/api\/v1$/, "");

  if (normalized.startsWith("/uploads/")) {
    return `${origin}${normalized}`;
  }

  if (normalized.startsWith("/api/")) {
    return `${origin}${normalized}`;
  }

  return `${apiRoot}${normalized}`;
}

const SERVICE_AREA_TYPE_LABELS: Record<string, string> = {
  commune: "Commune",
  district: "District",
  canton: "Canton",
};

const SWISS_CANTON_NAMES: Record<string, string> = {
  AG: "Argovie",
  AI: "Appenzell Rhodes-Intérieures",
  AR: "Appenzell Rhodes-Extérieures",
  BE: "Berne",
  BL: "Bâle-Campagne",
  BS: "Bâle-Ville",
  FR: "Fribourg",
  GE: "Genève",
  GL: "Glaris",
  GR: "Grisons",
  JU: "Jura",
  LU: "Lucerne",
  NE: "Neuchâtel",
  NW: "Nidwald",
  OW: "Obwald",
  SG: "Saint-Gall",
  SH: "Schaffhouse",
  SO: "Soleure",
  SZ: "Schwytz",
  TG: "Thurgovie",
  TI: "Tessin",
  UR: "Uri",
  VD: "Vaud",
  VS: "Valais",
  ZG: "Zoug",
  ZH: "Zurich",
};

function formatServiceAreaToken(token: string): string {
  const namedMatch = /^(commune_name|canton_name|district_name):(.+)$/i.exec(token.trim());
  if (namedMatch?.[2]) return namedMatch[2].trim();

  const rawMatch = /^(commune|district|canton):([A-Za-z0-9_-]+)$/i.exec(token.trim());
  if (!rawMatch) return token.trim();

  const zoneType = rawMatch[1].toLowerCase();
  const code = rawMatch[2];
  const typeLabel = SERVICE_AREA_TYPE_LABELS[zoneType] ?? "Zone";

  if (zoneType === "canton") {
    const cantonName = SWISS_CANTON_NAMES[code.toUpperCase()] ?? code;
    return `${typeLabel} de ${cantonName}`;
  }
  if (zoneType === "district") {
    return `${typeLabel} ${code}`;
  }
  return `${typeLabel} ${code}`;
}

export function formatServiceAreaLabel(rawValue: string | null | undefined): string | null {
  const raw = String(rawValue ?? "").trim();
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as {
      mode?: string;
      tokens?: unknown[];
    };
    const mode = typeof parsed.mode === "string" ? parsed.mode.trim().toLowerCase() : null;
    const tokens = Array.isArray(parsed.tokens)
      ? parsed.tokens
          .map((token) => (typeof token === "string" ? token.trim() : ""))
          .filter(Boolean)
      : [];

    if (tokens.length > 0) {
      const labels = tokens.map(formatServiceAreaToken);
      if (mode === "canton" && labels.length === 1) return labels[0];
      if (mode === "district" && labels.length === 1) return labels[0];
      const modeLabel = SERVICE_AREA_TYPE_LABELS[mode ?? ""] ?? null;
      if (modeLabel && labels.length > 1) {
        return `${modeLabel}s : ${labels.join(", ")}`;
      }
      return labels.join(", ");
    }
  } catch {
    // Valeur legacy (texte libre ou CSV)
  }

  const csvTokens = raw
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (csvTokens.some((token) => /^(commune|district|canton):/i.test(token))) {
    return csvTokens.map(formatServiceAreaToken).join(", ");
  }

  if (raw.startsWith("{") || raw.startsWith("[")) {
    return null;
  }

  return raw;
}

function readString(payload: Record<string, unknown>, key: string): string | null {
  const raw = payload[key];
  return typeof raw === "string" && raw.trim() ? raw.trim() : null;
}

function readBool(payload: Record<string, unknown>, key: string): boolean | null {
  const raw = payload[key];
  if (typeof raw === "boolean") return raw;
  if (raw === 1 || raw === "1") return true;
  if (raw === 0 || raw === "0") return false;
  return null;
}

function buildRows(entries: [string, string | null, boolean?][]): CompanyProfileRow[] {
  return entries
    .filter(([, value]) => value != null && value.trim().length > 0)
    .map(([label, value, fullWidth]) => ({
      label,
      value: value as string,
      fullWidth: fullWidth ?? false,
    }));
}

function formatDomicile(payload: Record<string, unknown>): string | null {
  const line1 = readString(payload, "domicile_address_line1");
  const line2 = readString(payload, "domicile_address_line2");
  const zip = readString(payload, "domicile_zip");
  const city = readString(payload, "domicile_city");
  const locality = [zip, city].filter(Boolean).join(" ").trim();
  const parts = [line1, line2, locality].filter(Boolean);
  return parts.length > 0 ? parts.join(", ") : null;
}

export function buildCompanyBillingSummary(payload: Record<string, unknown> | null): CompanyBillingSummary {
  if (!payload) {
    return { label: "À configurer", detail: "Définissez le tiers payeur par défaut." };
  }
  const defaultType =
    typeof payload.default_billed_to_type === "string" ? payload.default_billed_to_type.trim() : "";
  const contact =
    typeof payload.default_billed_to_contact === "string"
      ? payload.default_billed_to_contact.trim()
      : "";
  if (defaultType && contact) {
    return {
      label: defaultType,
      detail: contact,
    };
  }
  if (defaultType) {
    return { label: defaultType, detail: null };
  }
  return { label: "À configurer", detail: "Définissez le tiers payeur par défaut sur le portail web." };
}

export function buildCompanyProfileViewModel(
  profile: Record<string, unknown> | null | undefined,
  organizationName: string | null,
  apiBaseUrl: string
): CompanyProfileViewModel {
  const p = profile ?? {};
  const name = readString(p, "name") ?? organizationName ?? "Entreprise";
  const approved = readBool(p, "is_approved");
  const dispatchEnabled = readBool(p, "dispatch_enabled");
  const partner = readBool(p, "is_partner");
  const vehicles = Array.isArray(p.vehicles) ? p.vehicles : [];

  const badges: CompanyProfileBadge[] = [];
  if (approved === true) badges.push({ label: "Compte validé", tone: "active" });
  if (approved === false) badges.push({ label: "Validation en cours", tone: "inactive" });
  if (dispatchEnabled === true) badges.push({ label: "Dispatch actif", tone: "info" });
  if (partner === true) badges.push({ label: "Partenaire", tone: "info" });

  const identityRows = buildRows([
    ["Raison sociale", readString(p, "name") ?? organizationName],
    ["Adresse d'exploitation", readString(p, "address")],
    ["Domiciliation", formatDomicile(p), true],
    ["Zone de service", formatServiceAreaLabel(readString(p, "service_area")), true],
  ]);

  const contactRows = buildRows([
    ["E-mail contact", readString(p, "contact_email")],
    ["Téléphone", readString(p, "contact_phone")],
    ["E-mail facturation", readString(p, "billing_email")],
  ]);

  const legalRows = buildRows([
    ["UID / IDE", readString(p, "uid_ide")],
    ["Notes facturation", readString(p, "billing_notes"), true],
  ]);

  const sections: CompanyProfileSection[] = [];
  if (identityRows.length > 0) {
    sections.push({ id: "identity", title: "Identité", icon: "business", rows: identityRows });
  }
  if (contactRows.length > 0) {
    sections.push({ id: "contact", title: "Contact", icon: "call", rows: contactRows });
  }
  if (legalRows.length > 0) {
    sections.push({ id: "legal", title: "Facturation & légal", icon: "document", rows: legalRows });
  }

  return {
    displayName: name,
    logoUrl: resolveCompanyLogoUrl(readString(p, "logo_url"), apiBaseUrl),
    badges,
    sections,
    vehicleCount: vehicles.length,
  };
}
