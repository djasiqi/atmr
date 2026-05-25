import { Easing } from "react-native";

/** Palette motion LIRIE — ne pas étendre sans revue produit. */
export const Motion = {
  page: 180,
  detail: 220,
  pop: 180,
  modal: 240,
  modalClose: 180,
  micro: 120,
  data: 150,
  context: 280,
  /** Documentaire — chantier boot séparé (BootSplashGate). */
  boot: 350,
} as const;

/** Dev uniquement — active l'instrumentation motion. */
export const MotionDebug = {
  enabled: typeof __DEV__ !== "undefined" && __DEV__,
} as const;

export const MotionEasing = Easing.bezier(0.22, 1, 0.36, 1);

export const MotionDistance = {
  tabSlidePx: 8,
  /** Intention design ; Expo stack = carte pleine largeur. */
  pushSlidePx: 24,
  modalSlideYPx: 32,
  fadeFrom: 0.96,
  fadeTo: 1,
  dataFadeFrom: 0,
  dataFadeTo: 1,
  scaleInactive: 0.985,
} as const;
