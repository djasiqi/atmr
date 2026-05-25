import { LinearGradient } from "expo-linear-gradient";
import { useEffect } from "react";
import { StyleSheet, View, type ViewProps, type ViewStyle } from "react-native";
import {
  LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS,
  LIRIE_GOOGLE_MAP_ATTRIBUTION_MASK_COLOR,
  LIRIE_GOOGLE_MAP_LOGO_CLIP_PX,
  lirieWebMapHostStyle,
} from "./lirieMapChrome";

const LIRIE_MAP_CHROME_STYLE_ID = "lirie-google-map-chrome-hide";

/** Masque logo / contrôles Google résiduels (aligné natif + web). */
function useHideGoogleMapDefaultChrome() {
  useEffect(() => {
    if (typeof document === "undefined") return;
    if (document.getElementById(LIRIE_MAP_CHROME_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = LIRIE_MAP_CHROME_STYLE_ID;
    style.textContent = LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS;
    document.head.appendChild(style);
  }, []);
}

/**
 * Même composition visuelle que le hero CRA (`Home.module.css` : `.heroVisual` + `.mapShowcase` + overlay).
 * Cadre pour la carte flotte lorsque le rendu passe par Google Maps JavaScript (variante navigateur du bundle).
 */
type FrameProps = {
  height: number;
  mapDomId: string;
  accessibilityLabel?: string;
  /** Carte mission compacte : pas de bandeau bas (CSS masque déjà Google). */
  compact?: boolean;
};

export function LiriWebMapFrame({
  height,
  mapDomId,
  accessibilityLabel,
  compact = false,
}: FrameProps) {
  useHideGoogleMapDefaultChrome();
  return (
    <View style={[styles.heroVisual, { height }]} accessibilityLabel={accessibilityLabel}>
      <View
        style={[styles.mapShowcase, compact && styles.mapShowcaseCompact]}
        className="liri-web-map-showcase"
      >
        <View
          nativeID={mapDomId}
          // @ts-expect-error id web — complète nativeID pour getElementById
          id={mapDomId}
          style={compact ? styles.mapHostCompact : styles.mapHost}
        />
        {!compact ? (
          <>
            <LinearGradient
              pointerEvents="none"
              colors={["rgba(2, 6, 23, 0.01)", "rgba(2, 6, 23, 0)", "rgba(2, 6, 23, 0.06)"]}
              locations={[0, 0.45, 1]}
              style={styles.mapShowcaseOverlay}
            />
            <View
              pointerEvents="none"
              style={[
                styles.googleAttributionMask,
                {
                  height: LIRIE_GOOGLE_MAP_LOGO_CLIP_PX + 8,
                  backgroundColor: LIRIE_GOOGLE_MAP_ATTRIBUTION_MASK_COLOR,
                },
              ]}
            />
          </>
        ) : null}
      </View>
    </View>
  );
}

type PlaceholderProps = ViewProps & {
  height: number;
  /** Style appliqué au bloc intérieur (ex. fond d’erreur). */
  showcaseStyle?: ViewStyle;
};

/** Même cadre (coins + ombre) pour états chargement / erreur sans carte Google. */
export function LiriWebMapFramePlaceholder({ height, style, showcaseStyle, children, ...rest }: PlaceholderProps) {
  useHideGoogleMapDefaultChrome();
  return (
    <View style={[styles.heroVisual, { height }, style]} {...rest}>
      <View style={[styles.mapShowcase, styles.placeholderInner, showcaseStyle]} className="liri-web-map-showcase">
        {children}
      </View>
    </View>
  );
}

export const liriWebMapFrameRadius = 24;

const styles = StyleSheet.create({
  heroVisual: {
    width: "100%",
    position: "relative",
    zIndex: 0,
  },
  mapShowcase: {
    flex: 1,
    width: "100%",
    height: "100%",
    borderRadius: 24,
    overflow: "hidden",
    backgroundColor: "#F8FAFC",
    position: "relative",
    zIndex: 0,
    isolation: "isolate",
    boxShadow: "0 20px 36px -14px rgba(15, 23, 42, 0.18), 0 0 0 1px rgba(15, 23, 42, 0.05)",
  },
  mapHost: {
    ...lirieWebMapHostStyle(),
    width: "100%",
    height: "100%",
  },
  mapHostCompact: {
    ...StyleSheet.absoluteFillObject,
    width: "100%",
    height: "100%",
  },
  mapShowcaseCompact: {
    borderRadius: 16,
    boxShadow: undefined,
    backgroundColor: "transparent",
  },
  mapShowcaseOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1,
  },
  googleAttributionMask: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 2,
  },
  placeholderInner: {
    justifyContent: "center",
    padding: 12,
  },
});
