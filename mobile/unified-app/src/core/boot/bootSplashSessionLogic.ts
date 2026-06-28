export type BootSplashSessionStatus = "idle" | "bootstrapping" | "ready" | "error";

export function resolveBootSplashSessionBlocksOverlay(
  status: BootSplashSessionStatus,
  hasCompletedInitialBoot: boolean,
  introGateDone: boolean
): boolean {
  if ((status === "idle" || status === "bootstrapping") && !hasCompletedInitialBoot) {
    return true;
  }
  if (status === "ready" && !introGateDone) {
    return true;
  }
  return false;
}
