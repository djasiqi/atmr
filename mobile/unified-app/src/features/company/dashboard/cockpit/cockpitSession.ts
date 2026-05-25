import AsyncStorage from "@react-native-async-storage/async-storage";
import type { CockpitDriverSheetSnap } from "./cockpitTypes";

const SESSION_KEY = "@lirie/cockpit_session_v1";

export type CockpitSessionSnapshot = {
  selectedDriverId: number | null;
  selectedMissionId: number | null;
  driverSheetSnap: CockpitDriverSheetSnap;
  /** Overrides opérateur — survivent au background / redémarrage (TTL 30 min). */
  killSwitch?: boolean;
  autoFocusOff?: boolean;
  savedAt: number;
};

export async function saveCockpitSession(snapshot: CockpitSessionSnapshot): Promise<void> {
  try {
    await AsyncStorage.setItem(SESSION_KEY, JSON.stringify(snapshot));
  } catch {
    // non bloquant
  }
}

export async function loadCockpitSession(): Promise<CockpitSessionSnapshot | null> {
  try {
    const raw = await AsyncStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CockpitSessionSnapshot;
    if (Date.now() - parsed.savedAt > 30 * 60_000) return null;
    return parsed;
  } catch {
    return null;
  }
}
