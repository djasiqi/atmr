import { useMemo } from "react";
import { useWindowDimensions } from "react-native";

const WIDTH_SM = 375;
const WIDTH_MD = 768;
const WIDTH_LG = 1024;
const CONTENT_MAX_PHONE = 400;
const CONTENT_MAX_TABLET = 520;
const CONTENT_MAX_DESKTOP = 600;
const HORIZONTAL_PADDING = 16;
const MAP_HEIGHT_PHONE = 180;
const MAP_HEIGHT_TABLET = 260;
const MAP_HEIGHT_MAX = 340;

export type MissionLayout = {
  contentWidth: number;
  mapHeight: number;
  horizontalPadding: number;
  isTablet: boolean;
  isLargeScreen: boolean;
};

export function useMissionLayout(): MissionLayout {
  const { width, height } = useWindowDimensions();

  return useMemo(() => {
    const horizontalPadding = Math.max(12, Math.min(HORIZONTAL_PADDING, width * 0.05));
    const isTablet = width >= WIDTH_MD;
    const isLargeScreen = width >= WIDTH_LG;

    let contentWidth: number;
    if (width < WIDTH_SM) {
      contentWidth = width - horizontalPadding * 2;
    } else if (width < WIDTH_MD) {
      contentWidth = Math.min(CONTENT_MAX_PHONE, width - horizontalPadding * 2);
    } else if (width < WIDTH_LG) {
      contentWidth = Math.min(CONTENT_MAX_TABLET, width - horizontalPadding * 2);
    } else {
      contentWidth = Math.min(CONTENT_MAX_DESKTOP, width - horizontalPadding * 2);
    }

    let mapHeight = MAP_HEIGHT_PHONE;
    if (isTablet) {
      mapHeight = Math.min(MAP_HEIGHT_TABLET, Math.round(height * 0.24));
      mapHeight = Math.max(MAP_HEIGHT_PHONE, Math.min(MAP_HEIGHT_MAX, mapHeight));
    }

    return {
      contentWidth: Math.max(280, contentWidth),
      mapHeight,
      horizontalPadding,
      isTablet,
      isLargeScreen,
    };
  }, [height, width]);
}

