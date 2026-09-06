/**
 * DRIVER-COLD-01 — le splash natif ne se retire que lorsque
 * la surface de boot (overlay) est peinte, ou si elle ne s’affichera jamais.
 */
export function shouldReleaseNativeSplash(input: {
  overlayLaidOut: boolean;
  overlayWillNeverShow: boolean;
}): boolean {
  return input.overlayLaidOut || input.overlayWillNeverShow;
}
