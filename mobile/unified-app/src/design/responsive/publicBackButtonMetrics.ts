import type { AppViewport } from "./useAppViewport";

const BG_IDLE_PILL = "rgba(0, 121, 107, 0.06)";
const BG_PRESSED = "rgba(0, 121, 107, 0.12)";

/** Mesures alignées sur les rendus web (ghost vs pastille verte, icônes 22 / 20 / 18). */
export type PublicBackButtonMetrics = {
  iconSize: number;
  paddingVertical: number;
  paddingHorizontal: number;
  marginLeft: number;
  marginBottom: number;
  borderRadius: number;
  backgroundColorIdle: string;
  backgroundColorPressed: string;
};

export function getPublicBackButtonMetrics(
  viewport: Pick<AppViewport, "isTiny" | "isCompact">
): PublicBackButtonMetrics {
  const pill = viewport.isTiny || viewport.isCompact;
  const iconSize = viewport.isTiny ? 18 : viewport.isCompact ? 20 : 22;
  return {
    iconSize,
    paddingVertical: pill ? 8 : 10,
    paddingHorizontal: pill ? 10 : 12,
    marginLeft: pill ? -6 : 0,
    marginBottom: 14,
    borderRadius: 14,
    backgroundColorIdle: pill ? BG_IDLE_PILL : "transparent",
    backgroundColorPressed: BG_PRESSED,
  };
}
