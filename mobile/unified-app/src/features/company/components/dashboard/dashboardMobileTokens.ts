import { Platform, StyleSheet } from "react-native";
import { D } from "../../theme/companyDashboardTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

/** Densité mobile native (sous la carte). */
export const M = {
  gapSection: 6,
  gapRow: 4,
  padH: 12,
  padV: 8,
  padRow: 7,
  radiusSurface: 16,
  radiusRow: 12,
  hairline: D.border,
  iconSm: 32,
  iconMd: 36,
  barW: 3,
  feedIcon: 34,
} as const;

export const opsSurface = StyleSheet.create({
  root: {
    marginTop: 4,
    borderRadius: M.radiusSurface,
    backgroundColor: D.cardBg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: M.hairline,
    overflow: "hidden",
    ...(Platform.OS === "web"
      ? ({ boxShadow: "0 1px 0 rgba(15, 23, 42, 0.04)" } as object)
      : {}),
  },
  hairline: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: M.hairline,
    marginLeft: M.padH,
  },
  sectionHead: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: M.padH,
    paddingTop: 10,
    paddingBottom: 4,
  },
  sectionTitle: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "700",
    color: D.text,
    letterSpacing: -0.2,
  },
  sectionLink: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    color: D.brandDark,
  },
});
