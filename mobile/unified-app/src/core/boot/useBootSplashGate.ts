import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Animated, Easing, Platform, StyleSheet } from "react-native";
import { useAppViewport } from "../../design/responsive/useAppViewport";
import { useSession } from "../sessionProvider";
import { BOOT_LOTTIE_FIRST_LAUNCH_ONLY } from "./bootSplashConfig";
import { getBootLottieIntroSeen, setBootLottieIntroSeen } from "./bootSplashStorage";
import { resolveBootLottieSource } from "./resolveBootLottieSource";
import type { BootLottieAsset } from "./bootLottieAssets";

type IntroState = "loading" | "play" | "skip";

/** Durée du fondu entre la fin du splash et la première page (ms). */
export const SPLASH_FADE_OUT_MS = 420;

/**
 * Sur web il n’y a pas de Lottie natif (sans dépendance dotlottie) : durée de simulation
 * avant de considérer l’intro terminée (~2,6 s ≈ op 157 @ 60 fps sur les JSON fournis).
 */
export const WEB_SPLASH_LOTTIE_SIM_MS = 2600;

const shouldUseNativeDriver = Platform.OS !== "web";

export function useBootSplashGate(): {
  overlayMounted: boolean;
  fadeOpacity: Animated.Value;
  showOverlay: boolean;
  pointerEvents: "auto" | "none";
  showLottieLayer: boolean;
  source: BootLottieAsset;
  onLottieFinish: () => void;
  /** Insets alignés sur `useAppViewport` (safe top/bottom ≥ 16) pour cohérence avec le reste de l’UI. */
  insets: { top: number; bottom: number; left: number; right: number };
  styles: typeof styles;
} {
  const { width, height, topInset, bottomInset, safeLeft, safeRight } = useAppViewport();
  const [animFinished, setAnimFinished] = useState(false);
  const [introState, setIntroState] = useState<IntroState>(() =>
    BOOT_LOTTIE_FIRST_LAUNCH_ONLY ? "loading" : "play"
  );
  const { status } = useSession();
  const prevStatusRef = useRef(status);
  const fadeOpacity = useRef(new Animated.Value(1)).current;

  const source = useMemo(() => resolveBootLottieSource(width, height), [width, height]);

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
      fadeOpacity.setValue(1);
      setOverlayMounted(true);
    }
  }, [showOverlay, fadeOpacity]);

  const isExiting = overlayMounted && !showOverlay;

  useEffect(() => {
    if (showOverlay || !overlayMounted) {
      return;
    }

    const anim = Animated.timing(fadeOpacity, {
      toValue: 0,
      duration: SPLASH_FADE_OUT_MS,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: shouldUseNativeDriver,
    });

    anim.start(({ finished }) => {
      if (finished) {
        setOverlayMounted(false);
        fadeOpacity.setValue(1);
      }
    });

    return () => {
      anim.stop();
    };
  }, [showOverlay, overlayMounted, fadeOpacity]);

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

  const insets = useMemo(
    () => ({ top: topInset, bottom: bottomInset, left: safeLeft, right: safeRight }),
    [bottomInset, safeLeft, safeRight, topInset]
  );

  return {
    overlayMounted,
    fadeOpacity,
    showOverlay,
    pointerEvents: showOverlay ? "auto" : "none",
    showLottieLayer,
    source,
    onLottieFinish,
    insets,
    styles,
  };
}

const styles = StyleSheet.create({
  layer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 100000,
    elevation: 100000,
    backgroundColor: "#ffffff",
    justifyContent: "center",
    alignItems: "center",
  },
  lottie: {
    width: "100%",
    height: "100%",
  },
});
