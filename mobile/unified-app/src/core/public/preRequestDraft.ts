import * as SecureStore from "../storage/secureStoreCompat";

const PRE_REQUEST_DRAFT_KEY = "public_pre_request_draft_v1";
const PRE_REQUEST_DRAFT_INDEX_KEY = "public_pre_request_last_id_v1";

export type PublicPreRequestDraft = {
  draft_id: string;
  departure: string;
  destination: string;
  date: string;
  pickup_time?: string | null;
  /** True si l’utilisateur a ouvert « date et heure précises » (l’heure n’est plus « maintenant »). */
  pickup_schedule_exact?: boolean | null;
  /** `immediate` = accueil express + Dès que possible (heure = maintenant si pas exact). */
  reservation_urgency?: "immediate" | "planned" | null;
  trip_type?: "one_way" | "round_trip" | null;
  passengers?: number | null;
  transport_type: string;
  special_requirements?: string | null;
  contact_first_name?: string | null;
  contact_last_name?: string | null;
  contact_email?: string | null;
  contact_phone?: string | null;
  service_area_status?: "available" | "conditional" | "unavailable" | null;
  updated_at: number;
};

function safeParseDraft(raw: string | null): PublicPreRequestDraft | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as PublicPreRequestDraft;
    if (!parsed || typeof parsed !== "object" || !parsed.draft_id) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function createDraftId(): string {
  return `draft_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

export async function loadPublicPreRequestDraft(): Promise<PublicPreRequestDraft | null> {
  const raw = await SecureStore.getItemAsync(PRE_REQUEST_DRAFT_KEY);
  return safeParseDraft(raw);
}

export async function savePublicPreRequestDraft(
  draft: Omit<PublicPreRequestDraft, "updated_at"> & { updated_at?: number }
): Promise<PublicPreRequestDraft> {
  const normalized: PublicPreRequestDraft = {
    ...draft,
    updated_at: draft.updated_at ?? Date.now(),
  };
  await SecureStore.setItemAsync(PRE_REQUEST_DRAFT_KEY, JSON.stringify(normalized));
  await SecureStore.setItemAsync(PRE_REQUEST_DRAFT_INDEX_KEY, normalized.draft_id);
  return normalized;
}

export async function clearPublicPreRequestDraft(): Promise<void> {
  await SecureStore.deleteItemAsync(PRE_REQUEST_DRAFT_KEY);
  await SecureStore.deleteItemAsync(PRE_REQUEST_DRAFT_INDEX_KEY);
}

export async function getLastDraftId(): Promise<string | null> {
  const value = await SecureStore.getItemAsync(PRE_REQUEST_DRAFT_INDEX_KEY);
  if (!value || !value.trim()) return null;
  return value.trim();
}
