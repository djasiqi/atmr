import { Motion } from "./navigationMotion";

/**
 * Push hiérarchique cohérent (Expo Router slide_from_right = carte pleine largeur).
 * MotionDistance.pushSlidePx est une intention design, non appliquée par le moteur natif.
 */
export const stackPushOptions = {
  animation: "slide_from_right" as const,
  animationDuration: Motion.detail,
  gestureEnabled: true,
  fullScreenGestureEnabled: true,
};

export const stackPopOptions = {
  animation: "slide_from_right" as const,
  animationDuration: Motion.pop,
};

export const stackFadeOptions = {
  animation: "fade" as const,
  animationDuration: Motion.context,
};

export const stackNoneOptions = {
  animation: "none" as const,
};
