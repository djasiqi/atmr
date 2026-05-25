import { ActivityIndicator, StyleSheet, View, type ViewStyle } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { D } from "../theme/driverDashboardTheme";
import { MISSION_MAP_FALLBACK_HEIGHT } from "./maps/missionMapShared";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

type Props = {
  height?: number;
  style?: ViewStyle;
  pickupLocation?: string | null;
  dropoffLocation?: string | null;
  reason?: "missing_coords" | "missing_api_key";
  loading?: boolean;
};

function trimAddress(value: string | null | undefined): string | null {
  const t = value?.trim();
  return t && t.length > 0 ? t : null;
}

export function MissionMapUnavailable({
  height = MISSION_MAP_FALLBACK_HEIGHT,
  style,
  pickupLocation,
  dropoffLocation,
  reason = "missing_coords",
  loading = false,
}: Props) {
  const pickup = trimAddress(pickupLocation);
  const dropoff = trimAddress(dropoffLocation);
  const hasRouteHint = Boolean(pickup || dropoff);

  return (
    <View
      style={[styles.root, { minHeight: height, flex: 1, alignSelf: "stretch" }, style]}
      accessibilityLabel="Carte mission indisponible"
    >
      <View style={styles.backdrop} accessibilityElementsHidden>
        <View style={styles.gridLineH1} />
        <View style={styles.gridLineH2} />
        <View style={styles.gridLineV1} />
        <View style={styles.gridLineV2} />
        <View style={styles.backdropTint} />
      </View>

      <View style={styles.content}>
        <View style={styles.iconWell}>
          {loading ? (
            <ActivityIndicator size="small" color={D.brand} />
          ) : (
            <Ionicons name="map-outline" size={20} color={D.brand} />
          )}
        </View>
        <AppText variant="label" style={styles.title}>
          {loading ? "Chargement de la carte…" : "Carte indisponible"}
        </AppText>
        <AppText variant="caption" style={styles.subtitle}>
          {loading
            ? "Recherche des coordonnées GPS à partir des adresses du trajet."
            : reason === "missing_coords"
              ? "Coordonnées GPS du trajet non disponibles pour le moment."
              : "Configuration carte requise sur cet appareil."}
        </AppText>

        {hasRouteHint ? (
          <View style={styles.routeCard}>
            {pickup ? (
              <View style={styles.routeRow}>
                <View style={[styles.routeDot, styles.pickupDot]} accessibilityElementsHidden />
                <View style={styles.routeTextCol}>
                  <AppText variant="caption" style={styles.routeKey}>
                    DÉPART
                  </AppText>
                  <AppText variant="caption" style={styles.routeAddress} numberOfLines={2}>
                    {pickup}
                  </AppText>
                </View>
              </View>
            ) : null}
            {pickup && dropoff ? <View style={styles.routeConnector} accessibilityElementsHidden /> : null}
            {dropoff ? (
              <View style={styles.routeRow}>
                <View style={[styles.routeDot, styles.dropoffDot]} accessibilityElementsHidden />
                <View style={styles.routeTextCol}>
                  <AppText variant="caption" style={styles.routeKey}>
                    ARRIVÉE
                  </AppText>
                  <AppText variant="caption" style={styles.routeAddress} numberOfLines={2}>
                    {dropoff}
                  </AppText>
                </View>
              </View>
            ) : null}
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    width: "100%",
    overflow: "hidden",
    backgroundColor: "#EEF3F1",
    justifyContent: "center",
    alignItems: "center",
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  backdropTint: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(245, 247, 246, 0.72)",
  },
  gridLineH1: {
    position: "absolute",
    left: 0,
    right: 0,
    top: "32%",
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  gridLineH2: {
    position: "absolute",
    left: 0,
    right: 0,
    top: "68%",
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  gridLineV1: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: "28%",
    width: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  gridLineV2: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: "72%",
    width: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  content: {
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 10,
    maxWidth: "100%",
    gap: 4,
    zIndex: 1,
  },
  iconWell: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 2,
  },
  title: {
    color: D.text,
    fontWeight: "800",
    fontSize: FONT_SIZE.px13,
    textAlign: "center",
  },
  subtitle: {
    color: D.textMuted,
    fontWeight: "500",
    fontSize: FONT_SIZE.px11,
    lineHeight: 15,
    textAlign: "center",
    maxWidth: 280,
  },
  routeCard: {
    marginTop: 6,
    width: "100%",
    maxWidth: 320,
    backgroundColor: D.cardBg,
    borderWidth: 1,
    borderColor: D.cardBorder,
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
    gap: 6,
  },
  routeRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
  },
  routeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 4,
  },
  pickupDot: {
    backgroundColor: D.brand,
  },
  dropoffDot: {
    backgroundColor: D.flag,
  },
  routeConnector: {
    width: StyleSheet.hairlineWidth,
    height: 10,
    backgroundColor: D.stepLine,
    marginLeft: 3.5,
  },
  routeTextCol: {
    flex: 1,
    minWidth: 0,
    gap: 1,
  },
  routeKey: {
    color: D.routeLabel,
    fontWeight: "800",
    fontSize: FONT_SIZE.px8,
    letterSpacing: 0.35,
  },
  routeAddress: {
    color: D.routeText,
    fontWeight: "600",
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
  },
});
