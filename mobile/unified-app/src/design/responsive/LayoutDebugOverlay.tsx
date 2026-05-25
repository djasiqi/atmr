import { useCallback, useState } from "react";
import { Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { useLayoutDebugMetrics } from "./useLayoutDebugMetrics";
import { FONT_SIZE } from "./typographyTokens";
import { useChatLayoutKpis } from "./chatLayoutKpis";

const ENABLED =
  __DEV__ &&
  (process.env.EXPO_PUBLIC_LAYOUT_DEBUG === "1" ||
    process.env.EXPO_PUBLIC_LAYOUT_DEBUG === "true");

type Props = {
  /** Force l'affichage (tests). */
  forceEnabled?: boolean;
};

/**
 * Overlay debug layout — `__DEV__` + `EXPO_PUBLIC_LAYOUT_DEBUG=1`.
 * Affiche viewport, clavier, route, orientation, fontScale.
 */
export function LayoutDebugOverlay({ forceEnabled = false }: Props) {
  const active = forceEnabled || ENABLED;
  const [collapsed, setCollapsed] = useState(false);
  const toggle = useCallback(() => setCollapsed((c) => !c), []);

  if (!active) return null;

  return <LayoutDebugOverlayInner collapsed={collapsed} onToggle={toggle} />;
}

function LayoutDebugOverlayInner({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  const m = useLayoutDebugMetrics();
  const chat = useChatLayoutKpis();
  const chatActive = chat.composerKbGap != null && m.keyboardVisible;
  const composerKpiStatus =
    chat.composerKbGap == null
      ? "n/a"
      : Math.abs(chat.composerKbGap) <= 12
        ? "OK"
        : "OUT";

  const lines = collapsed
    ? [`${m.screenName} · ${m.orientation} · kb:${m.keyboardVisible ? "on" : "off"}`]
    : [
        `route: ${m.currentRoute}`,
        `screen: ${m.screenName} · ${m.orientation}`,
        `device: ${Math.round(m.deviceWidth)}×${Math.round(m.deviceHeight)} pr:${m.pixelRatio.toFixed(2)} fs:${m.fontScale.toFixed(2)}`,
        `usable: ${Math.round(m.usableWidth)}×${Math.round(m.usableHeight)}`,
        `safe: top ${m.safeTop} bottom ${m.safeBottom}`,
        `inset: top ${m.topInset} bottom ${m.bottomInset}`,
        `kb: vis ${m.keyboardVisible} h ${m.keyboardHeight} Δ ${m.resizeDelta} resized ${m.isResizedBySystem}`,
        `kb: inset ${m.visibleBottomInset} footerOff ${m.footerOffset}`,
        chatActive
          ? `chat: gap ${chat.composerKbGap}px (${composerKpiStatus}) shellGap ${chat.shellBottomGap} footerH ${chat.footerHeight}`
          : null,
        m.tabBarClearance != null
          ? `nav: tabClr ${m.tabBarClearance} floatH ${m.floatingBarHeight}`
          : "nav: (metrics n/a)",
      ].filter((l): l is string => l != null);

  return (
    <View style={styles.wrap} pointerEvents="box-none">
      <Pressable onPress={onToggle} style={styles.panel} accessibilityLabel="Layout debug metrics">
        {lines.map((line) => (
          <Text key={line} style={styles.line}>
            {line}
          </Text>
        ))}
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: "absolute",
    right: 8,
    bottom: 8,
    left: 8,
    zIndex: 99999,
    alignItems: "flex-end",
  },
  panel: {
    maxWidth: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.82)",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  line: {
    color: "#a7f3d0",
    fontSize: FONT_SIZE.px10,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    lineHeight: 14,
  },
});
