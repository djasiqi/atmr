import * as Location from "expo-location";
import { setTrackingPermissionsReady } from "./trackingPermissionsReady";

/** Hydrate permissionsReady depuis les permissions OS (FG+BG). */
export async function hydrateTrackingPermissionsReady(): Promise<boolean> {
  try {
    const fg = await Location.getForegroundPermissionsAsync().catch(() => null);
    const bg = await Location.getBackgroundPermissionsAsync().catch(() => null);
    const ready = Boolean(fg?.granted && bg?.granted);
    setTrackingPermissionsReady(ready);
    return ready;
  } catch {
    setTrackingPermissionsReady(false);
    return false;
  }
}
