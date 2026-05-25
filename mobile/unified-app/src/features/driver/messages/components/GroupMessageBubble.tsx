import { useEffect, useMemo, useState } from "react";
import { Image, Linking, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useAppViewport } from "../../../../design/responsive/useAppViewport";
import { AppText } from "../../../../design/ui/AppText";
import { resolveMediaUrl } from "../../../../core/api/mediaUrl";
import { CHAT_BUBBLE_OWN } from "../../../chat/chatPalette";
import { missionActiveCardShadow } from "../../theme/driverDashboardTheme";
import { createShadow } from "../../../../styles/shadowStyles";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

const groupBubbleShadow = createShadow(missionActiveCardShadow);
import type { SharedChatMessage } from "../../../chat/types";
import { VoiceMessageBar } from "../../../chat/components/VoiceMessageBar";
import {
  avatarColor,
  initialsFromName,
  senderColor,
  type GroupMessageMeta,
} from "../groupMessageLayout";

const MAX_IMAGE_W = 240;
const MAX_IMAGE_H = 160;

type Props = {
  message: SharedChatMessage;
  ownSenderId?: string | null;
  ownDisplayName?: string;
  group: GroupMessageMeta;
  onOpenImage?: (url: string) => void;
  onOpenPdf?: (url: string) => void;
  onMediaLayout?: () => void;
};

function formatTime(iso: string): string {
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return "";
  return new Date(d).toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function computeImageSize(naturalW: number, naturalH: number, maxW: number, maxH: number) {
  if (naturalW <= 0 || naturalH <= 0) {
    return { width: maxW, height: Math.min(maxW * 0.75, maxH) };
  }
  const ratio = naturalW / naturalH;
  let w = Math.min(naturalW, maxW);
  let h = w / ratio;
  if (h > maxH) {
    h = maxH;
    w = h * ratio;
  }
  return { width: w, height: h };
}

function isOwnMessage(
  message: SharedChatMessage,
  ownSenderId: string | null,
): boolean {
  return (
    ownSenderId != null &&
    message.senderId != null &&
    String(message.senderId) === String(ownSenderId)
  );
}

/** Bulle type WhatsApp groupe : nom dans la bulle, image + légende, heure en bas à droite. */
export function GroupMessageBubble({
  message,
  ownSenderId = null,
  ownDisplayName = "Moi",
  group,
  onOpenImage,
  onOpenPdf,
  onMediaLayout,
}: Props) {
  const sender = message.senderName ?? message.senderRole ?? "Équipe";
  const isOwn = isOwnMessage(message, ownSenderId);
  const displayName = isOwn ? ownDisplayName : sender;
  const time = formatTime(message.timestamp);
  const { usableWidth } = useAppViewport();
  const maxBubbleW = Math.min(300, Math.round(usableWidth * 0.78));
  const maxImageW = Math.min(MAX_IMAGE_W, Math.round(usableWidth * 0.65));

  const imageUri = useMemo(() => resolveMediaUrl(message.imageUrl), [message.imageUrl]);
  const pdfUri = useMemo(() => resolveMediaUrl(message.pdfUrl), [message.pdfUrl]);
  const audioUri = useMemo(() => resolveMediaUrl(message.audioUrl), [message.audioUrl]);
  const [imageNatural, setImageNatural] = useState<{ w: number; h: number } | null>(null);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    setImageNatural(null);
    setImageFailed(false);
  }, [imageUri]);

  useEffect(() => {
    if (imageNatural) onMediaLayout?.();
  }, [imageNatural, onMediaLayout]);

  const imageSize = useMemo(() => {
    if (imageNatural) {
      return computeImageSize(imageNatural.w, imageNatural.h, maxImageW, MAX_IMAGE_H);
    }
    return { width: maxImageW, height: Math.min(maxImageW * 0.65, MAX_IMAGE_H) };
  }, [imageNatural, maxImageW]);

  const raw = message.content?.trim() ?? "";
  const hasImage = Boolean(message.imageUrl);
  const hasPdf = Boolean(message.pdfUrl);
  const hasAudio = Boolean(message.audioUrl) && Boolean(audioUri);
  const isPlaceholder =
    raw === "(piece jointe)" ||
    (hasImage && raw.toLowerCase() === "image jointe") ||
    (hasPdf && raw.length > 0 && raw === (message.pdfFilename ?? "").trim()) ||
    (hasAudio && (raw === "" || raw.toLowerCase() === "message vocal"));
  const caption = isPlaceholder ? "" : raw;
  const showCaption = Boolean(caption);

  const openImage = () => {
    if (!imageUri) return;
    if (onOpenImage) onOpenImage(imageUri);
    else void Linking.openURL(imageUri);
  };

  return (
    <View
      style={[
        styles.row,
        isOwn && styles.rowOwn,
        !group.isFirstInGroup && styles.rowContinuation,
        group.isLastInGroup && styles.rowGroupEnd,
      ]}
    >
      {!isOwn ? (
        group.showAvatar ? (
          <View style={[styles.avatar, { backgroundColor: avatarColor(displayName) }]}>
            <AppText variant="caption" style={styles.avatarText}>
              {initialsFromName(displayName)}
            </AppText>
          </View>
        ) : (
          <View style={styles.avatarSpacer} />
        )
      ) : null}

      <View style={[styles.column, { maxWidth: maxBubbleW }, isOwn && styles.columnOwn]}>
        <View
          style={[
            styles.bubble,
            isOwn ? styles.bubbleOwn : styles.bubbleIn,
            group.isFirstInGroup && !isOwn && styles.bubbleTailIn,
            group.isFirstInGroup && isOwn && styles.bubbleTailOwn,
            !group.isFirstInGroup && !isOwn && styles.bubbleStackedIn,
            !group.isFirstInGroup && isOwn && styles.bubbleStackedOwn,
          ]}
        >
          {group.showSenderName && !isOwn ? (
            <AppText variant="caption" style={[styles.senderInBubble, { color: senderColor(displayName) }]}>
              {displayName}
            </AppText>
          ) : null}

          {hasImage && imageUri && !imageFailed ? (
            <Pressable
              onPress={openImage}
              style={({ pressed }) => [
                styles.imageWrap,
                {
                  width: imageSize.width,
                  height: imageSize.height,
                  marginTop: group.showSenderName ? 4 : 0,
                },
                pressed && styles.pressed,
              ]}
              accessibilityRole="button"
              accessibilityLabel="Ouvrir l'image"
            >
              <Image
                source={{ uri: imageUri }}
                style={{ width: imageSize.width, height: imageSize.height, borderRadius: 8 }}
                resizeMode="cover"
                onLoad={(e) => {
                  const w = e.nativeEvent.source.width;
                  const h = e.nativeEvent.source.height;
                  if (w > 0 && h > 0) setImageNatural({ w, h });
                }}
                onError={() => setImageFailed(true)}
              />
            </Pressable>
          ) : null}

          {showCaption ? (
            <AppText
              variant="body"
              style={[
                styles.caption,
                isOwn && styles.captionOwn,
                hasImage && styles.captionAfterImage,
              ]}
            >
              {caption}
            </AppText>
          ) : null}

          {hasPdf && pdfUri ? (
            <Pressable
              onPress={() => (onOpenPdf ? onOpenPdf(pdfUri) : Linking.openURL(pdfUri))}
              style={styles.pdfRow}
            >
              <Ionicons name="document-text-outline" size={22} color={isOwn ? "#fff" : "#0D9488"} />
              <AppText variant="caption" style={isOwn ? styles.captionOwn : styles.caption} numberOfLines={2}>
                {message.pdfFilename ?? "Document PDF"}
              </AppText>
            </Pressable>
          ) : null}

          {hasAudio && audioUri ? (
            <VoiceMessageBar uri={audioUri} isOwn={isOwn} />
          ) : null}

          <View style={styles.footer}>
            <AppText variant="caption" style={[styles.time, isOwn && styles.timeOwn]}>
              {time}
            </AppText>
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 4,
    paddingHorizontal: 4,
    marginBottom: 0,
  },
  rowOwn: {
    justifyContent: "flex-end",
  },
  rowContinuation: {
    marginTop: 0,
    marginBottom: 0,
  },
  rowGroupEnd: {
    marginBottom: 3,
  },
  avatar: {
    width: 26,
    height: 26,
    borderRadius: 13,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 0,
  },
  avatarSpacer: {
    width: 26,
  },
  avatarText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: FONT_SIZE.px11,
  },
  column: {
    flexShrink: 1,
  },
  columnOwn: {
    alignItems: "flex-end",
  },
  bubble: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingTop: 6,
    paddingBottom: 6,
    borderWidth: 0,
    maxWidth: "100%",
    alignSelf: "flex-start",
  },
  bubbleIn: {
    backgroundColor: "#FFFFFF",
    ...groupBubbleShadow,
  },
  bubbleOwn: {
    backgroundColor: CHAT_BUBBLE_OWN,
    borderWidth: 0,
    alignSelf: "flex-end",
  },
  bubbleTailIn: {
    borderTopLeftRadius: 4,
  },
  bubbleTailOwn: {
    borderTopRightRadius: 4,
  },
  bubbleStackedIn: {
    borderTopLeftRadius: 12,
  },
  bubbleStackedOwn: {
    borderTopRightRadius: 12,
  },
  senderInBubble: {
    fontWeight: "700",
    fontSize: FONT_SIZE.px12,
    marginBottom: 1,
    lineHeight: 16,
  },
  imageWrap: {
    borderRadius: 8,
    overflow: "hidden",
    backgroundColor: "#F3F4F6",
  },
  pressed: { opacity: 0.92 },
  caption: {
    color: "#111827",
    fontSize: FONT_SIZE.px15,
    lineHeight: 20,
    marginTop: 2,
  },
  captionAfterImage: {
    marginTop: 6,
  },
  captionOwn: {
    color: "#FFFFFF",
  },
  pdfRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 6,
  },
  footer: {
    flexDirection: "row",
    justifyContent: "flex-end",
    marginTop: 1,
    minHeight: 14,
  },
  time: {
    color: "#9CA3AF",
    fontSize: FONT_SIZE.px11,
  },
  timeOwn: {
    color: "rgba(255,255,255,0.75)",
  },
});
