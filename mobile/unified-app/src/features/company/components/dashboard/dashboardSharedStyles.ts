import { Platform, StyleSheet } from "react-native";
import { D, dashboardCardShadowWeb } from "../../theme/companyDashboardTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

export const dashboardSharedStyles = StyleSheet.create({
  card: {
    backgroundColor: D.cardBg,
    borderRadius: D.radiusCard,
    borderWidth: 1,
    borderColor: D.border,
    ...dashboardCardShadowWeb,
  },
  sectionTitle: {
    color: D.text,
    fontSize: FONT_SIZE.px16,
    fontWeight: "700",
  },
  sectionLink: {
    color: D.brand,
    fontSize: FONT_SIZE.px13,
    fontWeight: "700",
  },
  glass: {
    backgroundColor: D.glassBg,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: D.glassBorder,
    ...(Platform.OS === "web"
      ? ({ backdropFilter: "blur(12px)" } as object)
      : {}),
  },
});
