import { useMemo } from "react";
import { PixelRatio } from "react-native";
import { useSegments } from "expo-router";
import { useAppViewport } from "./useAppViewport";
import { useKeyboardLayout } from "../../features/chat/useKeyboardLayout";
import { useNavigationMetrics } from "../navigation/navigationMetrics";

export type LayoutDebugMetrics = {
  currentRoute: string;
  screenName: string;
  orientation: "portrait" | "landscape";
  deviceWidth: number;
  deviceHeight: number;
  pixelRatio: number;
  fontScale: number;
  usableWidth: number;
  usableHeight: number;
  safeTop: number;
  safeBottom: number;
  topInset: number;
  bottomInset: number;
  keyboardVisible: boolean;
  keyboardHeight: number;
  resizeDelta: number;
  isResizedBySystem: boolean;
  visibleBottomInset: number;
  footerOffset: number;
  tabBarClearance: number | null;
  floatingBarHeight: number | null;
};

export function useLayoutDebugMetrics(): LayoutDebugMetrics {
  const segments = useSegments();
  const viewport = useAppViewport();
  const keyboard = useKeyboardLayout();
  const nav = useNavigationMetrics();

  const orientation: "portrait" | "landscape" = viewport.width >= viewport.height ? "landscape" : "portrait";
  const currentRoute = segments.join("/") || "/";
  const screenName = segments[segments.length - 1] ?? "index";

  return useMemo(
    () => ({
      currentRoute,
      screenName,
      orientation,
      deviceWidth: viewport.width,
      deviceHeight: viewport.height,
      pixelRatio: PixelRatio.get(),
      fontScale: PixelRatio.getFontScale(),
      usableWidth: viewport.usableWidth,
      usableHeight: viewport.usableHeight,
      safeTop: viewport.safeTop,
      safeBottom: viewport.safeBottom,
      topInset: viewport.topInset,
      bottomInset: viewport.bottomInset,
      keyboardVisible: keyboard.keyboardVisible,
      keyboardHeight: keyboard.keyboardHeight,
      resizeDelta: keyboard.resizeDelta,
      isResizedBySystem: keyboard.isResizedBySystem,
      visibleBottomInset: keyboard.visibleBottomInset,
      footerOffset: keyboard.footerOffset(0),
      tabBarClearance: nav.tabBarClearance,
      floatingBarHeight: nav.floatingBarHeight,
    }),
    [currentRoute, screenName, orientation, viewport, keyboard, nav]
  );
}
