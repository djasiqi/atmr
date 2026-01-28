import { useWindowDimensions } from "react-native";
import { useMemo } from "react";

/** Breakpoints (largeur écran) pour layout responsive Mission */
const WIDTH_SM = 375;
const WIDTH_MD = 768;
const WIDTH_LG = 1024;

const CONTENT_MAX_PHONE = 400;
const CONTENT_MAX_TABLET = 520;
const CONTENT_MAX_DESKTOP = 600;
const HORIZONTAL_PADDING = 20;
const MAP_HEIGHT_PHONE = 180;
const MAP_HEIGHT_TABLET = 260;
const MAP_HEIGHT_MAX = 320;

export type MissionLayout = {
  /** Largeur cible pour la card et la carte (alignées) */
  contentWidth: number;
  /** Hauteur du bloc carte (web placeholder ou natif) */
  mapHeight: number;
  /** Marge horizontale autour du contenu */
  horizontalPadding: number;
  /** true si écran tablette (≥768) */
  isTablet: boolean;
  /** true si écran large (iPad 12", desktop) */
  isLargeScreen: boolean;
};

/**
 * Calcule les dimensions responsive pour la page Mission :
 * iPhone (petit/grand), Android, iPad, web.
 */
export function useMissionLayout(): MissionLayout {
  const { width, height } = useWindowDimensions();

  return useMemo(() => {
    const horizontalPadding = Math.max(16, Math.min(HORIZONTAL_PADDING, width * 0.05));
    const isTablet = width >= WIDTH_MD;
    const isLargeScreen = width >= WIDTH_LG;

    // Largeur contenu : ne pas dépasser l'écran moins les paddings
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
    contentWidth = Math.max(280, contentWidth);

    // Hauteur carte : proportionnelle sur tablette, fixe raisonnable sur phone
    let mapHeight: number;
    if (isTablet) {
      mapHeight = Math.min(MAP_HEIGHT_TABLET, Math.round(height * 0.22));
      mapHeight = Math.max(MAP_HEIGHT_PHONE, Math.min(MAP_HEIGHT_MAX, mapHeight));
    } else {
      mapHeight = MAP_HEIGHT_PHONE;
    }

    return {
      contentWidth,
      mapHeight,
      horizontalPadding,
      isTablet,
      isLargeScreen,
    };
  }, [width, height]);
}
