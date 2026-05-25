import { useMemo } from "react";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import { useAppViewport } from "../../../design/responsive/useAppViewport";
import {
  MISSION_MAP_HEIGHT_MAX,
  MISSION_MAP_HEIGHT_PHONE,
  MISSION_MAP_HEIGHT_TABLET_CAP,
} from "../components/maps/missionMapShared";

const TABLET_HEIGHT_RATIO = 0.24;

export type MissionLayout = {
  contentWidth: number;
  mapHeight: number;
  horizontalPadding: number;
  isTablet: boolean;
  isLargeScreen: boolean;
};

/**
 * Layout mission : géométrie depuis useAppViewport (usableWidth, contentWidth).
 * Texte très grand : carte un peu plus basse pour laisser la place au contenu scrollable parent.
 */
export function useMissionLayout(): MissionLayout {
  const viewport = useAppViewport();
  const { isVeryLargeText } = useAccessibilityScale();

  return useMemo(() => {
    const { contentWidth, usableHeight, isTablet, longest } = viewport;
    const isLargeScreen = longest >= 1024;

    let mapHeight = MISSION_MAP_HEIGHT_PHONE;
    if (isTablet) {
      mapHeight = Math.min(MISSION_MAP_HEIGHT_TABLET_CAP, Math.round(usableHeight * TABLET_HEIGHT_RATIO));
      mapHeight = Math.max(MISSION_MAP_HEIGHT_PHONE, Math.min(MISSION_MAP_HEIGHT_MAX, mapHeight));
    }
    if (isVeryLargeText) {
      mapHeight = Math.max(MISSION_MAP_HEIGHT_PHONE, Math.round(mapHeight * 0.88));
    }

    return {
      contentWidth: Math.max(0, contentWidth),
      mapHeight,
      horizontalPadding: viewport.horizontalPadding,
      isTablet,
      isLargeScreen,
    };
  }, [isVeryLargeText, viewport]);
}
