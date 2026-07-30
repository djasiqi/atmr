import { StyleSheet } from "react-native";
import { CHAT_BUBBLE_OWN } from "../chatPalette";

const C_BUBBLE_OWN = CHAT_BUBBLE_OWN;

/** Aligné sur `CHAT_IMAGE_INNER_MAX_W` dans MessageBubble. */
export const VOICE_GROUP_MAX_W = 248;

export const VOICE_WAVEFORM_BAR_COUNT = 28;
export const VOICE_WAVEFORM_HEIGHT = 28;

export const voiceStyles = StyleSheet.create({
  voiceRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 4,
    paddingHorizontal: 2,
    minWidth: 200,
    maxWidth: VOICE_GROUP_MAX_W,
    alignSelf: "stretch",
  },
  voiceRowOwn: {},
  voiceRowPressed: {
    opacity: 0.92,
  },
  playButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    backgroundColor: "rgba(13,148,136,0.14)",
  },
  playButtonOwn: {
    backgroundColor: "rgba(255,255,255,0.28)",
  },
  waveColumn: {
    flex: 1,
    minWidth: 0,
    gap: 4,
    justifyContent: "center",
  },
  waveTrack: {
    height: VOICE_WAVEFORM_HEIGHT,
    flexDirection: "row",
    alignItems: "center",
    gap: 2,
  },
  waveBar: {
    flex: 1,
    borderRadius: 2,
    minWidth: 2,
    maxWidth: 3.5,
  },
  scrubDot: {
    position: "absolute",
    width: 10,
    height: 10,
    borderRadius: 5,
    marginLeft: -5,
    top: (VOICE_WAVEFORM_HEIGHT - 10) / 2,
  },
  durationText: {
    fontVariant: ["tabular-nums"],
  },
  durationOwn: {
    color: "rgba(255,255,255,0.82)",
  },
  durationIn: {
    color: "#6B7280",
  },
  durationError: {
    color: "#B91C1C",
  },
  durationErrorOwn: {
    color: "#FECACA",
  },
});

export { C_BUBBLE_OWN };
