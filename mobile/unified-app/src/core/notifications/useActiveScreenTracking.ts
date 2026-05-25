import { useFocusEffect } from "expo-router";
import { useCallback } from "react";
import type { ActiveScreenRole } from "./activeScreenStore";
import {
  clearActiveMissionScreen,
  clearActiveThreadScreen,
  setActiveMissionScreen,
  setActiveThreadScreen,
} from "./activeScreenStore";

export function useActiveMissionScreenTracking(missionId: number | null): void {
  useFocusEffect(
    useCallback(() => {
      if (!Number.isFinite(missionId)) return undefined;
      setActiveMissionScreen(missionId as number);
      return () => {
        clearActiveMissionScreen(missionId as number);
      };
    }, [missionId])
  );
}

export function useActiveThreadScreenTracking(
  threadId: string | null | undefined,
  role: ActiveScreenRole
): void {
  useFocusEffect(
    useCallback(() => {
      if (!threadId) return undefined;
      setActiveThreadScreen(threadId, role);
      return () => {
        clearActiveThreadScreen(threadId);
      };
    }, [role, threadId])
  );
}
