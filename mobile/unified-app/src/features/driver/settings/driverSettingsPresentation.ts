import type { PushRegistrationBannerState } from "../../../core/notifications/pushRegistrationState";
import type { DriverProfile } from "../api";

export type DriverGpsPermissionSnapshot = {
  foregroundGranted: boolean;
  backgroundGranted: boolean;
  servicesEnabled: boolean;
};

export type DriverGpsStatusKey = "active" | "check" | "disabled";

export type DriverProfileRow = { label: string; value: string; fullWidth?: boolean };

export type DriverProfileSection = {
  id: string;
  title: string;
  icon: "person" | "car" | "medkit" | "call";
  rows: DriverProfileRow[];
};

export type DriverProfileBadge = {
  label: string;
  tone: "active" | "inactive" | "contract";
};

export type DriverProfileViewModel = {
  displayName: string;
  photoUrl: string | null;
  initials: string;
  badges: DriverProfileBadge[];
  companyName: string | null;
  sections: DriverProfileSection[];
  weeklyHours: number | null;
};

const CONTRACT_LABELS: Record<string, string> = {
  CDI: "CDI",
  CDD: "CDD",
  TEMPORARY: "Temporaire",
  INTERIM: "Intérim",
  FREELANCE: "Indépendant",
};

export function formatProfileField(value: string | null | undefined, fallback = "—"): string {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : fallback;
}

export function formatProfileDate(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Date.parse(trimmed);
  if (!Number.isFinite(parsed)) return trimmed;
  return new Date(parsed).toLocaleDateString("fr-CH", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

function readString(profile: DriverProfile, key: string): string | null {
  const raw = profile[key];
  return typeof raw === "string" && raw.trim() ? raw.trim() : null;
}

function readBool(profile: DriverProfile, key: string): boolean | null {
  const raw = profile[key];
  if (typeof raw === "boolean") return raw;
  if (raw === 1 || raw === "1") return true;
  if (raw === 0 || raw === "0") return false;
  return null;
}

function resolveVehicleLabel(profile: DriverProfile): string | null {
  const assigned = readString(profile, "vehicle_assigned");
  if (assigned) return assigned;

  const brand = readString(profile, "brand");
  const plate = readString(profile, "license_plate");
  if (brand && plate) return `${brand} - ${plate}`;

  const vehicle = profile.vehicle;
  if (vehicle && typeof vehicle === "object") {
    const v = vehicle as Record<string, unknown>;
    const model = typeof v.model === "string" ? v.model.trim() : "";
    const license = typeof v.license_plate === "string" ? v.license_plate.trim() : "";
    if (model && license) return `${model} - ${license}`;
    if (model) return model;
  }
  return null;
}

function resolveLicenseCategories(profile: DriverProfile): string | null {
  const raw = profile.license_categories;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const labels = raw.filter((item) => typeof item === "string" && item.trim()).map(String);
  return labels.length > 0 ? labels.join(", ") : null;
}

function buildRows(entries: [string, string | null, boolean?][]): DriverProfileRow[] {
  return entries
    .filter(([, value]) => value != null && value.trim().length > 0)
    .map(([label, value, fullWidth]) => ({
      label,
      value: value as string,
      fullWidth: fullWidth ?? false,
    }));
}

export function buildDriverProfileViewModel(
  profile: DriverProfile | null | undefined,
  user: { full_name?: string | null; username?: string | null; email?: string | null } | null,
  companyName: string | null
): DriverProfileViewModel {
  const p = profile ?? {};
  const firstName = readString(p, "first_name");
  const lastName = readString(p, "last_name");
  const fromParts = [firstName, lastName].filter(Boolean).join(" ").trim();
  const displayName =
    readString(p, "full_name") ||
    (fromParts.length > 0 ? fromParts : null) ||
    user?.full_name?.trim() ||
    user?.username?.trim() ||
    "Chauffeur";

  const photoRaw =
    readString(p, "photo_url") ||
    readString(p, "driver_photo") ||
    readString(p, "photo");

  const rawWeekly = p.weekly_hours;
  const weeklyHours =
    typeof rawWeekly === "number" && Number.isFinite(rawWeekly) && rawWeekly > 0
      ? Math.round(rawWeekly)
      : null;

  const isActive = readBool(p, "is_active");
  const contractRaw = readString(p, "contract_type");
  const badges: DriverProfileBadge[] = [];
  if (isActive != null) {
    badges.push({
      label: isActive ? "Actif" : "Inactif",
      tone: isActive ? "active" : "inactive",
    });
  }
  if (contractRaw) {
    badges.push({
      label: CONTRACT_LABELS[contractRaw.toUpperCase()] ?? contractRaw,
      tone: "contract",
    });
  }

  const identityRows = buildRows([
    ["Prénom", firstName],
    ["Nom", lastName],
    ["Date de naissance", formatProfileDate(p.birth_date)],
    ["Nationalité", readString(p, "nationality")],
    ["N° AVS", readString(p, "avs_number"), true],
    ["E-mail", readString(p, "email") ?? user?.email ?? null, true],
    ["Téléphone", readString(p, "phone"), true],
    ["Adresse", readString(p, "address"), true],
    ["Entreprise", companyName, true],
  ]);

  const vehicleRows = buildRows([["Véhicule assigné", resolveVehicleLabel(p), true]]);

  const licenseRows = buildRows([
    ["Catégories permis", resolveLicenseCategories(p), true],
    ["Validité permis", formatProfileDate(p.license_valid_until)],
    ["Validité médicale", formatProfileDate(p.medical_valid_until)],
    ["Début emploi", formatProfileDate(p.employment_start_date)],
  ]);

  const emergencyRows = buildRows([
    ["Nom", readString(p, "emergency_contact_name")],
    ["Téléphone", readString(p, "emergency_contact_phone"), true],
  ]);

  const sections: DriverProfileSection[] = [];
  if (identityRows.length > 0) {
    sections.push({ id: "identity", title: "Identité", icon: "person", rows: identityRows });
  }
  if (vehicleRows.length > 0) {
    sections.push({ id: "vehicle", title: "Véhicule", icon: "car", rows: vehicleRows });
  }
  if (licenseRows.length > 0) {
    sections.push({ id: "license", title: "Permis et médical", icon: "medkit", rows: licenseRows });
  }
  if (emergencyRows.length > 0) {
    sections.push({
      id: "emergency",
      title: "Contact d'urgence",
      icon: "call",
      rows: emergencyRows,
    });
  }

  return {
    displayName,
    photoUrl: photoRaw,
    initials: driverSettingsInitials(displayName),
    badges,
    companyName,
    sections,
    weeklyHours,
  };
}

export function resolveDriverWeeklyHoursMessage(weeklyHours: number | null): string {
  if (weeklyHours != null) {
    return `${weeklyHours} h / semaine au contrat. Consultez votre planning pour vos missions.`;
  }
  return "Aucun horaire configuré. Consultez votre planning de missions.";
}

export function resolveDriverGpsStatus(snapshot: DriverGpsPermissionSnapshot): {
  key: DriverGpsStatusKey;
  label: string;
  hint: string;
} {
  if (!snapshot.servicesEnabled) {
    return {
      key: "disabled",
      label: "Désactivé",
      hint: "Activez le GPS dans les réglages de votre téléphone.",
    };
  }
  if (snapshot.foregroundGranted && snapshot.backgroundGranted) {
    return {
      key: "active",
      label: "Actif",
      hint: "Votre position est partagée pendant vos missions.",
    };
  }
  if (snapshot.foregroundGranted) {
    return {
      key: "check",
      label: "À compléter",
      hint: "Autorisez la localisation « toujours » pour un suivi fiable.",
    };
  }
  return {
    key: "check",
    label: "Désactivé",
    hint: "Activez l'accès à la localisation pour recevoir des missions.",
  };
}

export function resolveNotificationsEnabled(
  state: PushRegistrationBannerState,
  permissionDenied: boolean
): boolean {
  if (permissionDenied || state === "permission_denied") return false;
  if (state === "disclosure_required") return false;
  return true;
}

export function resolveDriverNotificationStatus(
  state: PushRegistrationBannerState,
  permissionDenied: boolean
): { label: string; tone: "ok" | "warn" | "error" } {
  if (!resolveNotificationsEnabled(state, permissionDenied)) {
    return { label: "Désactivées", tone: "error" };
  }
  if (state === "registration_pending") {
    return { label: "Activation en cours…", tone: "warn" };
  }
  if (state === "registration_failed") {
    return { label: "À vérifier", tone: "warn" };
  }
  return { label: "Activées", tone: "ok" };
}

export function driverSettingsInitials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0] ?? ""}${parts[1][0] ?? ""}`.toUpperCase();
}

export const DRIVER_LOCATION_PRIVACY_TEXT =
  "Votre position est utilisée uniquement pendant vos horaires de travail ou lorsqu'une mission est active.";

/** @deprecated Utiliser buildDriverProfileViewModel */
export function resolveDriverProfileIdentity(
  profile: DriverProfile | null | undefined,
  user: { full_name?: string | null; username?: string | null; email?: string | null } | null
) {
  const vm = buildDriverProfileViewModel(profile, user, null);
  return {
    displayName: vm.displayName,
    firstName: vm.sections[0]?.rows.find((r) => r.label === "Prénom")?.value ?? null,
    lastName: vm.sections[0]?.rows.find((r) => r.label === "Nom")?.value ?? null,
    email: vm.sections[0]?.rows.find((r) => r.label === "E-mail")?.value ?? "—",
    phone: vm.sections[0]?.rows.find((r) => r.label === "Téléphone")?.value ?? "—",
    photoUrl: vm.photoUrl,
    weeklyHours: vm.weeklyHours,
  };
}
