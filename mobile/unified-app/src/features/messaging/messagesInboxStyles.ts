import { StyleSheet } from "react-native";
import { M } from "./messagingTheme";
import { FONT_SIZE } from "../../design/responsive/typographyTokens";

export const messagesInboxStyles = StyleSheet.create({
  headerBlock: {
    backgroundColor: M.CARD,
    gap: 8,
    paddingBottom: 4,
  },
  search: {
    backgroundColor: M.PAGE_BG,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: FONT_SIZE.px15,
    color: M.TEXT,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: M.BORDER,
  },
  syncBannerSlot: {
    minHeight: 28,
    justifyContent: "center",
  },
  offlineChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    alignSelf: "flex-start",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    backgroundColor: M.OFFLINE_CHIP_BG,
  },
  offlineText: { color: M.OFFLINE_CHIP_TEXT, fontSize: FONT_SIZE.px12 },
  list: { flex: 1, backgroundColor: M.CARD },
});
