import { StyleSheet } from "react-native";

const C_HEADING = "#111827";
const C_BODY = "#6b7280";
const C_BUBBLE_OWN = "#0d9488";

/** Aligné sur `CHAT_IMAGE_INNER_MAX_W` dans MessageBubble. */
export const VOICE_GROUP_MAX_W = 320 - 32;

export const voiceStyles = StyleSheet.create({
  voiceRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    padding: 10,
    borderRadius: 8,
    backgroundColor: "rgba(0,0,0,0.04)",
  },
  voiceRowOwn: {
    backgroundColor: "rgba(255,255,255,0.2)",
  },
  voiceRowPressed: {
    opacity: 0.9,
  },
  voiceIconBox: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(13,148,136,0.12)",
    alignItems: "center",
    justifyContent: "center",
  },
  voiceIconBoxOwn: {
    backgroundColor: "rgba(255,255,255,0.25)",
  },
  voiceTextBlock: {
    flex: 1,
    minWidth: 0,
  },
  voiceTitle: {
    fontSize: 14,
    fontWeight: "600",
    color: C_HEADING,
  },
  voiceTitleOwn: {
    color: "#fff",
  },
  voiceMeta: {
    fontSize: 12,
    color: C_BODY,
    marginTop: 2,
  },
  voiceMetaOwn: {
    color: "rgba(255,255,255,0.75)",
  },
});

export { C_BUBBLE_OWN };
