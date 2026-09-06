/**
 * OPT-07 — graphe de boot LIRIE (entreprise).
 * Ce qui est requis pour le premier écran utile (Cockpit) vs ce qui attend.
 * Semi-auto / optimizer : jamais au boot (LOCK OFF).
 */

export const COMPANY_EAGER_TAB_NAMES = new Set(["dashboard", "rides"]);

export type CompanyBootLane = "critical" | "background" | "never";

export type CompanyBootWorkItem = {
  id: string;
  lane: CompanyBootLane;
  reason: string;
};

export const COMPANY_BOOT_WORK: readonly CompanyBootWorkItem[] = [
  {
    id: "session.local",
    lane: "critical",
    reason: "credentials + bootstrap cache → entrer dans l’app",
  },
  {
    id: "shell.navigation",
    lane: "critical",
    reason: "barre + Cockpit montés (lazy:false Cockpit/Courses)",
  },
  {
    id: "snapshot.disk",
    lane: "critical",
    reason: "dernier état J / dashboard / roster pour affichage immédiat",
  },
  {
    id: "rides.j.page1",
    lane: "critical",
    reason: "journée J page 1 — premier écran",
  },
  {
    id: "dashboard.realtime",
    lane: "critical",
    reason: "résumé cockpit",
  },
  {
    id: "drivers.live",
    lane: "critical",
    reason: "roster + positions (âge GPS réel, jamais LIVE synthétique)",
  },
  {
    id: "dispatch.status",
    lane: "critical",
    reason: "mode + état — GET /mode redondant interdit",
  },
  {
    id: "inbox.notifications",
    lane: "background",
    reason: "pas requis pour utiliser le Cockpit",
  },
  {
    id: "chat.unread",
    lane: "background",
    reason: "badge barre — après premier écran",
  },
  {
    id: "offers.pending",
    lane: "background",
    reason: "badge — après premier écran",
  },
  {
    id: "rides.delays",
    lane: "background",
    reason: "secondaire Courses / cockpit",
  },
  {
    id: "rides.completeDay",
    lane: "background",
    reason: "pages 2..N seulement quand Courses est utile",
  },
  {
    id: "prefetch.adjacent",
    lane: "background",
    reason: "J±1 page 1 — confort nav, pas premier rendu",
  },
  {
    id: "session.bootstrap.network",
    lane: "background",
    reason: "validation / refresh serveur après entrée locale",
  },
  {
    id: "tabs.code.preload",
    lane: "background",
    reason: "NAV-01 — JS Chat / Menu après premier écran, sans GET",
  },
  {
    id: "billing.invoices",
    lane: "never",
    reason: "écran non ouvert",
  },
  {
    id: "clients.list",
    lane: "never",
    reason: "écran non ouvert",
  },
  {
    id: "companies.me",
    lane: "never",
    reason: "profil réglages — aucun champ Cockpit",
  },
  {
    id: "dispatch.mode.get",
    lane: "never",
    reason: "status fournit déjà le mode",
  },
  {
    id: "optimizer.status",
    lane: "never",
    reason: "LOCK OFF — aucun engine au boot",
  },
  {
    id: "history.edit.exports",
    lane: "never",
    reason: "modules secondaires / tap utilisateur",
  },
] as const;

export function resolveCompanyTabLazy(routeName: string): boolean {
  return !COMPANY_EAGER_TAB_NAMES.has(routeName);
}

export function companyBootWorkByLane(lane: CompanyBootLane): string[] {
  return COMPANY_BOOT_WORK.filter((item) => item.lane === lane).map((item) => item.id);
}

export function isCompanyBootWorkAllowedAtLane(
  id: string,
  currentLane: Exclude<CompanyBootLane, "never">
): boolean {
  const item = COMPANY_BOOT_WORK.find((row) => row.id === id);
  if (!item || item.lane === "never") return false;
  if (currentLane === "critical") return item.lane === "critical";
  return item.lane === "critical" || item.lane === "background";
}
