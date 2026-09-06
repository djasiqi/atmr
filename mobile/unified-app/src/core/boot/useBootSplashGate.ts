import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Animated, AppState, Easing, Platform, StyleSheet } from "react-native";
import { useAppViewport } from "../../design/responsive/useAppViewport";
import { useSession } from "../sessionProvider";
import { BOOT_LOTTIE_FIRST_LAUNCH_ONLY } from "./bootSplashConfig";
import { getBootLottieIntroSeen, setBootLottieIntroSeen } from "./bootSplashStorage";
import { resolveBootSplashSessionBlocksOverlay } from "./bootSplashSessionLogic";
import { shouldReportBootSplashFallback } from "./bootSplashFallbackLogic";
import { computeBootLottieDisplaySize } from "./bootLottieLayout";
import { reportBootFallback } from "../observability/bootDiagnostics";
import { resolveBootLottieSource } from "./resolveBootLottieSource";
import type { BootLottieAsset } from "./bootLottieAssets";
import { SPLASH_BACKGROUND_COLOR } from "./bootSurface";
import { markBootMilestone } from "../observability/bootMilestones";

export { SPLASH_BACKGROUND_COLOR } from "./bootSurface";

type IntroState = "loading" | "play" | "skip";

/** Durée du fondu entre la fin du splash et la première page (ms). */
export const SPLASH_FADE_OUT_MS = 420;
/** Petit délai pour laisser la landing publique se peindre avant le fade-out. */
export const SPLASH_EXIT_HOLD_MS = 60;
/** Fondu du calque Lottie pour une sortie plus douce. */
export const SPLASH_LOTTIE_FADE_OUT_MS = 260;
/**
 * Sur web il n’y a pas de Lottie natif (sans dépendance dotlottie) : durée de simulation
 * avant de considérer l’intro terminée (~2,6 s ≈ op 157 @ 60 fps sur les JSON fournis).
 */
export const WEB_SPLASH_LOTTIE_SIM_MS = 2600;

/** Safety timeout : force animFinished si Lottie ne déclenche jamais onAnimationFinish. */
export const SPLASH_LOTTIE_FALLBACK_TIMEOUT_MS = 4000;

/**
 * Durée naturelle de l'animation d'intro (~op 157 @ 60 fps sur les JSON fournis).
 * Sert à terminer l'intro de façon DÉTERMINISTE sur natif, sans dépendre de
 * `onAnimationFinish` (callback non fiable sous Fabric/Hermes : il peut ne jamais
 * arriver, laissant le splash bloqué jusqu'au filet de secours à 4 s).
 */
export const SPLASH_LOTTIE_INTRO_MS = 2600;

const shouldUseNativeDriver = Platform.OS !== "web";

export function useBootSplashGate(): {
  overlayMounted: boolean;
  fadeOpacity: Animated.Value;
  lottieOpacity: Animated.Value;
  showOverlay: boolean;
  pointerEvents: "auto" | "none";
  showLottieLayer: boolean;
  source: BootLottieAsset;
  onLottieFinish: () => void;
  /** Insets alignés sur `useAppViewport` (safe top/bottom ≥ 16) pour cohérence avec le reste de l’UI. */
  insets: { top: number; bottom: number; left: number; right: number };
  styles: typeof styles;
  lottieStyle: { width: number; height: number; backgroundColor: string; alignSelf: "stretch" };
} {
  const { width, height, topInset, bottomInset, safeLeft, safeRight } = useAppViewport();
  const [animFinished, setAnimFinished] = useState(false);
  const [introState, setIntroState] = useState<IntroState>(() =>
    BOOT_LOTTIE_FIRST_LAUNCH_ONLY ? "loading" : "play"
  );
  const { status } = useSession();
  const prevStatusRef = useRef(status);
  const bootStartedAtRef = useRef(Date.now());
  const overlayBootAtRef = useRef(Date.now());
  const lastSnapshotRef = useRef("");
  const fallbackWarnedRef = useRef(false);
  const fallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const overlayMountedRef = useRef(false);
  const animFinishedRef = useRef(false);
  const statusRef = useRef(status);
  const introStateRef = useRef(introState);
  statusRef.current = status;
  introStateRef.current = introState;
  /** Après le premier boot réussi, ne plus bloquer l'UI sur idle/bootstrapping (ex. logout). */
  const hasCompletedInitialBootRef = useRef(false);
  const fadeOpacity = useRef(new Animated.Value(1)).current;
  const lottieOpacity = useRef(new Animated.Value(1)).current;

  const source = useMemo(() => resolveBootLottieSource(width, height), [width, height]);

  const lottieStyle = useMemo(() => {
    const { width: lottieWidth, height: lottieHeight } = computeBootLottieDisplaySize(width, height, source);
    return {
      width: lottieWidth,
      height: lottieHeight,
      backgroundColor: SPLASH_BACKGROUND_COLOR,
      alignSelf: "stretch" as const,
    };
  }, [height, source, width]);

  useEffect(() => {
    if (!BOOT_LOTTIE_FIRST_LAUNCH_ONLY) {
      setIntroState("play");
      return;
    }
    let cancelled = false;
    void (async () => {
      const seen = await getBootLottieIntroSeen();
      if (!cancelled) {
        setIntroState(seen ? "skip" : "play");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (prevStatusRef.current !== status) {
      console.log("[BootSplash] status transition", prevStatusRef.current, "->", status);
    }
    if (status === "ready" || status === "error") {
      hasCompletedInitialBootRef.current = true;
    }
    if (prevStatusRef.current === "error" && status === "bootstrapping") {
      setAnimFinished(false);
    }
    prevStatusRef.current = status;
  }, [status]);

  const introGateDone =
    !BOOT_LOTTIE_FIRST_LAUNCH_ONLY
      ? animFinished
      : introState === "skip"
        ? true
        : introState === "loading"
          ? false
          : animFinished;

  /** Cold start suivant : pas de Lottie, mais l’overlay reste jusqu’au shell prêt. */
  const skipLottie = BOOT_LOTTIE_FIRST_LAUNCH_ONLY && introState === "skip";
  const waitingIntroStorage = BOOT_LOTTIE_FIRST_LAUNCH_ONLY && introState === "loading";

  const sessionBlocksOverlay = resolveBootSplashSessionBlocksOverlay(
    status,
    hasCompletedInitialBootRef.current,
    introGateDone
  );

  const showOverlay =
    status !== "error" && (waitingIntroStorage || sessionBlocksOverlay);

  const [overlayMounted, setOverlayMounted] = useState(showOverlay);

  useEffect(() => {
    if (showOverlay) {
      fadeOpacity.stopAnimation(() => undefined);
      lottieOpacity.stopAnimation(() => undefined);
      fadeOpacity.setValue(1);
      lottieOpacity.setValue(1);
      setOverlayMounted(true);
    }
  }, [showOverlay, fadeOpacity, lottieOpacity]);

  const isExiting = overlayMounted && !showOverlay;

  useEffect(() => {
    if (showOverlay || !overlayMounted) {
      return;
    }

    const timer = setTimeout(() => {
      Animated.parallel([
        Animated.timing(fadeOpacity, {
          toValue: 0,
          duration: SPLASH_FADE_OUT_MS,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: shouldUseNativeDriver,
        }),
        Animated.timing(lottieOpacity, {
          toValue: 0,
          duration: SPLASH_LOTTIE_FADE_OUT_MS,
          easing: Easing.out(Easing.quad),
          useNativeDriver: shouldUseNativeDriver,
        }),
      ]).start(({ finished }) => {
        if (finished) {
          console.log(
            "[BootSplash] overlay hidden after",
            Date.now() - bootStartedAtRef.current,
            "ms",
          );
          markBootMilestone("OVERLAY_HIDDEN", {
            overlay_hidden_ms: Date.now() - bootStartedAtRef.current,
          });
          markBootMilestone("DASHBOARD_INTERACTIVE", {
            meaning: "overlay_hidden",
            overlay_hidden_ms: Date.now() - bootStartedAtRef.current,
          });
          setOverlayMounted(false);
          fadeOpacity.setValue(1);
          lottieOpacity.setValue(1);
        }
      });
    }, SPLASH_EXIT_HOLD_MS);

    return () => {
      clearTimeout(timer);
      fadeOpacity.stopAnimation(() => undefined);
      lottieOpacity.stopAnimation(() => undefined);
    };
  }, [showOverlay, overlayMounted, fadeOpacity, lottieOpacity]);

  const lottieBaseAllowed =
    !skipLottie &&
    !waitingIntroStorage &&
    (introState === "play" || !BOOT_LOTTIE_FIRST_LAUNCH_ONLY);

  const showLottieLayer = overlayMounted && lottieBaseAllowed && (showOverlay || isExiting);

  const onLottieFinish = useCallback(() => {
    setAnimFinished(true);
    if (BOOT_LOTTIE_FIRST_LAUNCH_ONLY && introState === "play") {
      void setBootLottieIntroSeen();
    }
  }, [introState]);

  const onLottieFinishRef = useRef(onLottieFinish);
  onLottieFinishRef.current = onLottieFinish;

  const clearSplashFallbackTimer = useCallback(() => {
    if (fallbackTimerRef.current != null) {
      clearTimeout(fallbackTimerRef.current);
      fallbackTimerRef.current = null;
    }
  }, []);

  const triggerSplashFallback = useCallback(() => {
    const elapsedMs = Date.now() - bootStartedAtRef.current;
    const elapsedSinceOverlayMs = Date.now() - overlayBootAtRef.current;
    if (!fallbackWarnedRef.current) {
      fallbackWarnedRef.current = true;
      if (
        shouldReportBootSplashFallback({
          elapsedSinceOverlayMs,
          elapsedSinceSessionMs: elapsedMs,
        })
      ) {
        reportBootFallback("BootSplashFallbackTriggered", {
          elapsedMs,
          elapsedSinceOverlayMs,
          status: statusRef.current,
          introState: introStateRef.current,
        });
      }
    }
    onLottieFinishRef.current();
  }, []);

  const armSplashFallbackTimer = useCallback(() => {
    if (Platform.OS === "web") {
      return;
    }
    clearSplashFallbackTimer();
    fallbackTimerRef.current = setTimeout(() => {
      fallbackTimerRef.current = null;
      triggerSplashFallback();
    }, SPLASH_LOTTIE_FALLBACK_TIMEOUT_MS);
  }, [clearSplashFallbackTimer, triggerSplashFallback]);

  // Complétion DÉTERMINISTE de l'intro, indépendante de `onAnimationFinish`.
  // Ce callback Lottie est non fiable sous Fabric/Hermes (il peut ne jamais arriver
  // → splash bloqué jusqu'au filet de secours à 4 s, cf. BootSplashFallbackTriggered).
  // On termine donc l'intro après la durée naturelle de l'animation sur toutes les
  // plateformes ; `onAnimationFinish` ne sert plus que de raccourci s'il arrive avant.
  useEffect(() => {
    if (!showLottieLayer || !showOverlay || animFinished) {
      return;
    }
    const durationMs =
      Platform.OS === "web" ? WEB_SPLASH_LOTTIE_SIM_MS : SPLASH_LOTTIE_INTRO_MS;
    const id = setTimeout(() => {
      onLottieFinish();
    }, durationMs);
    return () => {
      clearTimeout(id);
    };
  }, [showLottieLayer, showOverlay, animFinished, onLottieFinish]);

  useEffect(() => {
    animFinishedRef.current = animFinished;
  }, [animFinished]);

  useEffect(() => {
    overlayMountedRef.current = overlayMounted;
    if (!overlayMounted) {
      fallbackWarnedRef.current = false;
      clearSplashFallbackTimer();
      return;
    }
    if (showOverlay) {
      overlayBootAtRef.current = Date.now();
    }
  }, [clearSplashFallbackTimer, overlayMounted, showOverlay]);

  useEffect(() => {
    if (Platform.OS === "web") {
      return;
    }
    // Filet dès que l'overlay est monté (pas seulement le calque Lottie).
    if (!overlayMounted || animFinished) {
      clearSplashFallbackTimer();
      return;
    }

    armSplashFallbackTimer();
    return clearSplashFallbackTimer;
  }, [animFinished, armSplashFallbackTimer, clearSplashFallbackTimer, overlayMounted]);

  useEffect(() => {
    if (Platform.OS === "web") {
      return;
    }
    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState !== "active") {
        clearSplashFallbackTimer();
        return;
      }
      if (!overlayMountedRef.current || animFinishedRef.current) {
        return;
      }
      const elapsedSinceOverlayMs = Date.now() - overlayBootAtRef.current;
      // Timer throttlé en arrière-plan : terminer sans alerter Sentry si hors fenêtre boot.
      if (!shouldReportBootSplashFallback({
        elapsedSinceOverlayMs,
        elapsedSinceSessionMs: Date.now() - bootStartedAtRef.current,
      })) {
        onLottieFinishRef.current();
        return;
      }
      armSplashFallbackTimer();
    });
    return () => {
      subscription.remove();
      clearSplashFallbackTimer();
    };
  }, [armSplashFallbackTimer, clearSplashFallbackTimer]);

  useEffect(() => {
    const elapsedMs = Date.now() - bootStartedAtRef.current;
    const snapshot = JSON.stringify({
      status,
      introState,
      animFinished,
      showLottieLayer,
      showOverlay,
      overlayMounted,
      elapsedMs,
    });

    if (snapshot !== lastSnapshotRef.current) {
      lastSnapshotRef.current = snapshot;
      console.log("[BootSplash] state", snapshot);
    }
  }, [status, introState, animFinished, showLottieLayer, showOverlay, overlayMounted]);

  const insets = useMemo(
    () => ({ top: topInset, bottom: bottomInset, left: safeLeft, right: safeRight }),
    [bottomInset, safeLeft, safeRight, topInset]
  );

  return {
    overlayMounted,
    fadeOpacity,
    lottieOpacity,
    showOverlay,
    pointerEvents: showOverlay ? "auto" : "none",
    showLottieLayer,
    source,
    onLottieFinish,
    insets,
    styles,
    lottieStyle,
  };
}

const styles = StyleSheet.create({
  layer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 100000,
    elevation: 100000,
    backgroundColor: SPLASH_BACKGROUND_COLOR,
  },
  lottieLayer: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "transparent",
  },
});
