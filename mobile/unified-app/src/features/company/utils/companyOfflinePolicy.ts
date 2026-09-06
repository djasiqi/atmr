import { isAxiosError } from "axios";
import { getNetworkSnapshot, type NetworkSnapshot } from "../../../core/network/networkState";

export const COMPANY_OFFLINE_ACTION_TITLE = "Connexion indisponible";
export const COMPANY_OFFLINE_ACTION_BODY =
  "Impossible d’effectuer cette action pour le moment.";
export const COMPANY_OFFLINE_ACTION_MESSAGE = `${COMPANY_OFFLINE_ACTION_TITLE}. ${COMPANY_OFFLINE_ACTION_BODY}`;
export const COMPANY_OFFLINE_DAY_TITLE = "Cette journée n’est pas disponible hors connexion.";
export const COMPANY_OFFLINE_DAY_BODY =
  "Revenez en ligne pour charger cette date. Les autres journées en cache restent consultables.";

export class CompanyOfflineActionError extends Error {
  readonly offline = true as const;

  constructor() {
    super(COMPANY_OFFLINE_ACTION_MESSAGE);
    this.name = "CompanyOfflineActionError";
  }
}

/** `connected === false` seulement — `internetReachable` est trop bruité (Android BG). */
export function isCompanyNetworkOffline(snapshot: NetworkSnapshot = getNetworkSnapshot()): boolean {
  return snapshot.connected === false;
}

export function isCompanyNetworkRequestError(error: unknown): boolean {
  if (error instanceof CompanyOfflineActionError) return true;
  if (!error || typeof error !== "object") return false;
  if (isAxiosError(error) && error.response != null) return false;
  const code = isAxiosError(error) ? String(error.code ?? "") : "";
  const message = error instanceof Error ? error.message : String(error);
  const haystack = `${code} ${message}`.toLowerCase();
  return (
    code === "ERR_NETWORK" ||
    code === "ECONNABORTED" ||
    code === "ENOTFOUND" ||
    code === "ECONNREFUSED" ||
    /network request failed|failed to fetch|network error|timeout|connexion|internet/i.test(
      haystack
    )
  );
}

export function assertCompanyOnlineForMutation(): void {
  if (isCompanyNetworkOffline()) {
    throw new CompanyOfflineActionError();
  }
}

/** Retry query : 1 essai max, jamais si offline / erreur réseau. */
export function shouldRetryCompanyQuery(failureCount: number, error: unknown): boolean {
  if (isCompanyNetworkOffline() || isCompanyNetworkRequestError(error)) return false;
  return failureCount < 1;
}

export function shouldRetryCompanyMutation(_failureCount: number, error: unknown): boolean {
  void _failureCount;
  if (isCompanyNetworkOffline() || isCompanyNetworkRequestError(error)) return false;
  return false;
}

export type DispatchDayEmptyKind =
  | "list"
  | "loading"
  | "offline_unavailable"
  | "search_pending"
  | "search_none"
  | "empty";

export function resolveDispatchDayEmptyKind(input: {
  missionCount: number;
  hasCachedDay: boolean;
  isLoading: boolean;
  isError: boolean;
  isOffline: boolean;
  isNetworkError: boolean;
  searchActive: boolean;
  isDayComplete: boolean;
}): DispatchDayEmptyKind {
  if (input.missionCount > 0) return "list";
  if (
    !input.hasCachedDay &&
    (input.isOffline || (input.isError && input.isNetworkError))
  ) {
    return "offline_unavailable";
  }
  if (input.isLoading) return "loading";
  if (input.searchActive && !input.isDayComplete) return "search_pending";
  if (input.searchActive) return "search_none";
  return "empty";
}
