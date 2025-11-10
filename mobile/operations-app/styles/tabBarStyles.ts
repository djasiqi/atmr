import { Platform, ViewStyle, TextStyle } from "react-native";

type ViewStyleStrict = ViewStyle & {
  alignItems?: "flex-start" | "flex-end" | "center" | "stretch" | "baseline";
  justifyContent?:
    | "flex-start"
    | "flex-end"
    | "center"
    | "space-between"
    | "space-around"
    | "space-evenly";
};

const palette = {
  background: "#07130E",
  barSurface: "rgba(10,34,26,0.94)",
  border: "rgba(46,128,94,0.32)",
  shadow: "#04150F",
  indicator: "rgba(30,185,128,0.18)",
  label: "#E6F2EA",
  labelInactive: "rgba(184,214,198,0.65)",
};

export const tabBarStyles = {
  tabBarStyle: {
    position: "absolute",
    backgroundColor: palette.barSurface,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderTopWidth: 1,
    borderColor: palette.border,
    elevation: 8,
    shadowColor: palette.shadow,
    shadowOpacity: 0.28,
    shadowOffset: { width: 0, height: -4 },
    shadowRadius: 12,
    height: Platform.OS === "ios" ? 78 : 68,
    paddingBottom: Platform.OS === "ios" ? 22 : 14,
    paddingTop: 6,
    paddingHorizontal: 10,
  } as ViewStyle,

  tabBarItemStyle: {
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 6,
    borderRadius: 14,
    paddingVertical: 6,
    paddingHorizontal: 10,
  } as ViewStyleStrict,

  tabBarLabelStyle: {
    fontSize: 12,
    fontWeight: "700",
    marginTop: 0,
    letterSpacing: 0.3,
    color: palette.label,
  } as TextStyle,

  palette,
};
