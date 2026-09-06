import { Easing } from "react-native";
import { Motion, MotionEasing } from "../design/navigation/navigationMotion";
import { resolveMotionDuration } from "../design/navigation/applyNavigationMotion";
import { lirieTabFadeSlide } from "../design/navigation/tabSceneInterpolator";

/** Options communes aux barres d'onglets flottantes (driver + entreprise). */
export const FLOATING_TAB_PAGE_BG = {
  driver: "#F5F7F6",
  company: "#f4f7fc",
} as const;

/**
 * Barre flottante en overlay : la scène garde toute la hauteur (carte full-bleed).
 * À appliquer uniquement là où le contenu gère déjà un clearance (cockpit, chat, settings).
 */
export const FLOATING_TAB_BAR_OVERLAY_STYLE = {
  position: "absolute" as const,
  backgroundColor: "transparent",
  borderTopWidth: 0,
  elevation: 0,
  shadowOpacity: 0,
};

/** Empêche le navigator de réserver le safe-area inférieur à la scène. */
export const FLOATING_TAB_SAFE_AREA_NONE = { top: 0, bottom: 0 } as const;

/**
 * Spec transition LIRIE — fade + slide 8 px, 180 ms, easing unique.
 * Conserve `Easing` import pour compat ascendante des consommateurs externes.
 */
export const floatingTabTransitionSpec = {
  animation: "timing" as const,
  config: {
    duration: Motion.page,
    easing: MotionEasing,
  },
};

/**
 * @deprecated Conservé pour compat. Préférer `lirieTabFadeSlide` (fade + 8 px) qui
 * incarne le contrat LIRIE — slide pleine largeur abandonné en MVP transitions.
 */
export function opaqueHorizontalTabSlide(_slideWidth: number, pageBg: string) {
  return lirieTabFadeSlide(pageBg);
}

/**
 * Transition LIRIE : fade + slide 8 px, sans démontage ni gel des écrans adjacents.
 * `reduceMotion=true` raccourcit la durée (≤ 80 ms) et neutralise le slide.
 */
export function buildFloatingTabScreenOptions(
  pageBg: string,
  _slideWidth: number,
  reduceMotion = false
) {
  return {
    headerShown: false as const,
    animation: "shift" as const,
    transitionSpec: {
      animation: "timing" as const,
      config: {
        duration: resolveMotionDuration(Motion.page, reduceMotion),
        easing: MotionEasing,
      },
    },
    sceneStyle: { flex: 1, backgroundColor: pageBg },
    sceneStyleInterpolator: lirieTabFadeSlide(pageBg, reduceMotion),
    lazy: false,
    freezeOnBlur: false,
    detachInactiveScreens: false,
    inactiveBehavior: "none" as const,
    /** Évite l'espace blanc entre champs et clavier quand la tab bar flottante reste montée. */
    tabBarHideOnKeyboard: true,
  };
}

/** Requis pour que l'animation custom (sans fondu) remplace le moteur natif. */
export const FLOATING_TAB_IMPLEMENTATION = "custom" as const;

/** Réexport pour migrations en cours — éviter d'utiliser dans du code neuf. */
export { Easing };
