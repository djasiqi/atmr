import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Animated, Easing, Platform, StyleSheet } from "react-native";
import { useAppViewport } from "../../design/responsive/useAppViewport";
import { useSession } from "../sessionProvider";
import { BOOT_LOTTIE_FIRST_LAUNCH_ONLY } from "./bootSplashConfig";
import { getBootLottieIntroSeen, setBootLottieIntroSeen } from "./bootSplashStorage";
import { computeBootLottieDisplaySize } from "./bootLottieLayout";
import { resolveBootLottieSource } from "./resolveBootLottieSource";
import type { BootLottieAsset } from "./bootLottieAssets";

type IntroState = "loading" | "play" | "skip";

/** Durée du fondu entre la fin du splash et la première page (ms). */
export const SPLASH_FADE_OUT_MS = 420;
/** Petit délai pour laisser la landing publique se peindre avant le fade-out. */
export const SPLASH_EXIT_HOLD_MS = 60;
/** Fondu du calque Lottie pour une sortie plus douce. */
export const SPLASH_LOTTIE_FADE_OUT_MS = 260;
/** Fond aligné avec les écrans publics pour éviter l'effet de flash. */
export const SPLASH_BACKGROUND_COLOR = "#EAF3F1";

/**
 * Sur web il n’y a pas de Lottie natif (sans dépendance dotlottie) : durée de simulation
 * avant de considérer l’intro terminée (~2,6 s ≈ op 157 @ 60 fps sur les JSON fournis).
 */
export const WEB_SPLASH_LOTTIE_SIM_MS = 2600;

/** Safety timeout : force animFinished si Lottie ne déclenche jamais onAnimationFinish. */
export const SPLASH_LOTTIE_FALLBACK_TIMEOUT_MS = 4000;

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
  const lastSnapshotRef = useRef("");
  const fallbackWarnedRef = useRef(false);
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

  const skipEntireIntro = BOOT_LOTTIE_FIRST_LAUNCH_ONLY && introState === "skip";
  const waitingIntroStorage = BOOT_LOTTIE_FIRST_LAUNCH_ONLY && introState === "loading";

  const sessionBlocksOverlay =
    status === "idle" || status === "bootstrapping" || (status === "ready" && !introGateDone);

  const showOverlay =
    status !== "error" && !skipEntireIntro && (waitingIntroStorage || sessionBlocksOverlay);

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
    !waitingIntroStorage && (introState === "play" || !BOOT_LOTTIE_FIRST_LAUNCH_ONLY);

  const showLottieLayer = overlayMounted && lottieBaseAllowed && (showOverlay || isExiting);

  const onLottieFinish = useCallback(() => {
    setAnimFinished(true);
    if (BOOT_LOTTIE_FIRST_LAUNCH_ONLY && introState === "play") {
      void setBootLottieIntroSeen();
    }
  }, [introState]);

  useEffect(() => {
    if (Platform.OS !== "web") {
      return;
    }
    if (!showLottieLayer || !showOverlay) {
      return;
    }
    const id = setTimeout(() => {
      onLottieFinish();
    }, WEB_SPLASH_LOTTIE_SIM_MS);
    return () => {
      clearTimeout(id);
    };
  }, [showLottieLayer, showOverlay, onLottieFinish]);

  useEffect(() => {
    if (Platform.OS === "web") {
      return;
    }
    if (!showLottieLayer || animFinished) {
      return;
    }

    const id = setTimeout(() => {
      if (!fallbackWarnedRef.current) {
        fallbackWarnedRef.current = true;
        console.warn(
          "[BootSplash] Lottie fallback triggered after",
          SPLASH_LOTTIE_FALLBACK_TIMEOUT_MS,
          "ms without onAnimationFinish",
        );
      }
      onLottieFinish();
    }, SPLASH_LOTTIE_FALLBACK_TIMEOUT_MS);

    return () => clearTimeout(id);
  }, [showLottieLayer, animFinished, onLottieFinish]);

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
