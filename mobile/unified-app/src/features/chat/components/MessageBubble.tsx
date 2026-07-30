import { memo, useEffect, useMemo, useState } from "react";
import { Image, Pressable, View, Platform, StyleSheet } from "react-native";
import { useAppViewport } from "../../../design/responsive/useAppViewport";
import { AppText } from "../../../design/ui/AppText";
import * as Linking from "expo-linking";
import { Ionicons } from "@expo/vector-icons";
import { resolveMediaUrl } from "../../../core/api/mediaUrl";
import { CHAT_BUBBLE_OWN } from "../chatPalette";
import { SharedChatMessage } from "../types";
import { VoiceMessageBar } from "./VoiceMessageBar";
import { VOICE_GROUP_MAX_W } from "./voiceMessageStyles";

/**
 * Bulles : contenu long → mise à l’échelle large (`maxFontSizeMultiplier` ~1.5) ; métadonnées courtes → cap plus bas.
 * Jamais `allowFontScaling={false}` sur le fil.
 */

type MessageBubbleProps = {
  message: SharedChatMessage;
  ownSenderId?: string | number | null;
  ownSenderRoles?: string[];
  onOpenImage?: (url: string) => void;
  onOpenPdf?: (url: string) => void;
  /** Appelé quand une image/PDF change la hauteur de la bulle (recalage scroll fil). */
  onMediaLayout?: () => void;
  /** Fil équipe / mobile : bulles et images plus compactes. */
  density?: "default" | "compact";
};

const C_HEADING = "#111827";
const C_BODY = "#6b7280";
const C_BUBBLE_IN = "#f3f4f6";
const C_BUBBLE_OWN = CHAT_BUBBLE_OWN;
const C_BORDER = "#e5e7eb";
const C_FAIL = "#b91c1c";
const AVATAR_PALETTE = ["#6366f1", CHAT_BUBBLE_OWN, "#d97706", "#7c3aed", "#db2777", "#2563eb"];

function initialsFromName(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase() || "?";
}

function avatarColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i += 1) h = (h + name.charCodeAt(i) * (i + 1)) % 997;
  return AVATAR_PALETTE[h % AVATAR_PALETTE.length];
}

function formatMessageTime(iso: string): string {
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return "";
  return new Date(d).toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function isOwnMessage(
  message: SharedChatMessage,
  ownSenderId: string | number | null | undefined,
  ownSenderRoles: string[]
): boolean {
  if (ownSenderId != null && message.senderId != null) {
    return String(message.senderId) === String(ownSenderId);
  }
  const role = message.senderRole?.toUpperCase();
  if (!role) return false;
  return ownSenderRoles.some((candidate) => candidate.toUpperCase() === role);
}

const DEFAULT_ASPECT = 4 / 3;
/** Colonne type Flowbite `max-w-[320px]`, moins le `p-4` (16+16) de la bulle. */
const CHAT_IMAGE_INNER_MAX_W = 320 - 32;
/** Aperçu compact mais lisible (évite les images trop hautes dans le fil). */
const CHAT_IMAGE_MAX_H = 200;
const CHAT_IMAGE_MAX_H_COMPACT = 132;
const CHAT_IMAGE_MAX_W_COMPACT = 220;
const IMAGE_BLOCK_GAP = 10;

/** Largeur dispo. image (s’aligne sur la largeur de bulle, plafond 288). */
function capChatImageWidth(windowW: number): number {
  return Math.min(CHAT_IMAGE_INNER_MAX_W, windowW * 0.9);
}

/**
 * Taille d’affichage en conservant le ratio, plafond largeur + hauteur
 * (paysage → priorité à la largeur, portrait → limite surtout la hauteur).
 */
function computeImageDisplaySize(
  naturalW: number,
  naturalH: number,
  maxW: number,
  maxH: number
): { width: number; height: number } {
  if (naturalW <= 0 || naturalH <= 0) {
    return { width: maxW, height: maxH / DEFAULT_ASPECT };
  }
  const ratio = naturalW / naturalH;
  let outW = Math.min(naturalW, maxW);
  let outH = outW / ratio;
  if (outH > maxH) {
    outH = maxH;
    outW = outH * ratio;
  }
  return { width: outW, height: outH };
}

export const MessageBubble = memo(function MessageBubble({
  message,
  ownSenderId,
  ownSenderRoles = ["COMPANY"],
  onOpenImage,
  onOpenPdf,
  onMediaLayout,
  density = "default",
}: MessageBubbleProps) {
  const sender = message.senderName ?? message.senderRole ?? "Équipe";
  const isOwn = isOwnMessage(message, ownSenderId, ownSenderRoles);
  const time = formatMessageTime(message.timestamp);
  const { usableWidth } = useAppViewport();
  const isCompact = density === "compact";
  const columnMaxW = isCompact ? Math.min(300, Math.round(usableWidth * 0.82)) : 320;
  const maxImageW = isCompact
    ? Math.min(CHAT_IMAGE_MAX_W_COMPACT, Math.round(usableWidth * 0.68))
    : capChatImageWidth(usableWidth);
  const maxImageH = isCompact ? CHAT_IMAGE_MAX_H_COMPACT : CHAT_IMAGE_MAX_H;

  const imageUri = useMemo(() => resolveMediaUrl(message.imageUrl), [message.imageUrl]);
  const pdfUri = useMemo(() => resolveMediaUrl(message.pdfUrl), [message.pdfUrl]);
  const audioUri = useMemo(() => resolveMediaUrl(message.audioUrl), [message.audioUrl]);
  const [imageLoadFailed, setImageLoadFailed] = useState(false);
  const [imageNatural, setImageNatural] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    setImageLoadFailed(false);
    setImageNatural(null);
  }, [imageUri]);

  useEffect(() => {
    if (imageNatural) onMediaLayout?.();
  }, [imageNatural, onMediaLayout]);

  const imageDisplaySize = useMemo(() => {
    if (imageNatural) {
      return computeImageDisplaySize(imageNatural.w, imageNatural.h, maxImageW, maxImageH);
    }
    const w = maxImageW;
    const h = w / DEFAULT_ASPECT;
    return { width: w, height: Math.min(h, maxImageH) };
  }, [imageNatural, maxImageH, maxImageW]);

  const handleImageLoad = (
    e: { nativeEvent?: { source?: { width?: number; height?: number; uri?: string } } } | undefined
  ) => {
    const source = e?.nativeEvent?.source;
    const width = source?.width;
    const height = source?.height;
    if (typeof width === "number" && typeof height === "number" && width > 0 && height > 0) {
      setImageNatural({ w: width, h: height });
      return;
    }
    if (imageUri) {
      Image.getSize(
        imageUri,
        (w, h) => {
          if (w > 0 && h > 0) setImageNatural({ w, h });
        },
        () => {}
      );
    }
  };

  const openImage = () => {
    if (!imageUri) return;
    if (onOpenImage) onOpenImage(imageUri);
    else void Linking.openURL(imageUri);
  };
  const openPdf = () => {
    if (!pdfUri) return;
    if (onOpenPdf) onOpenPdf(pdfUri);
    else void Linking.openURL(pdfUri);
  };

  const hasAudio = Boolean(message.audioUrl) && Boolean(audioUri);
  // Ne jamais traiter une pièce vocale comme image (évite le rectangle vertical cassé).
  const hasImage = Boolean(message.imageUrl) && !hasAudio;
  const hasPdf = Boolean(message.pdfUrl) && !hasAudio;
  const failed = message.content.includes("(echec envoi)");
  const raw = message.content?.trim() ?? "";
  const isAttachmentPlaceholder =
    raw === "(piece jointe)" ||
    (hasImage && raw.toLowerCase() === "image jointe") ||
    (hasPdf && raw.length > 0 && raw === (message.pdfFilename ?? "").trim()) ||
    (hasAudio && (raw === "" || raw.toLowerCase() === "message vocal"));
  const showText = Boolean(raw) && !isAttachmentPlaceholder;
  const imageOnly = hasImage && !showText && !hasPdf && !hasAudio;

  const bubbleShadow =
    Platform.OS === "web"
      ? { boxShadow: "0 1px 2px rgba(0,0,0,0.05)" }
      : {
          shadowColor: "#000",
          shadowOffset: { width: 0, height: 1 },
          shadowOpacity: 0.05,
          shadowRadius: 2,
          elevation: 1,
        };

  return (
    <View style={[styles.row, isCompact && styles.rowCompact, isOwn && styles.rowOwn]}>
      {!isOwn && (
        <View style={[styles.avatar, { backgroundColor: avatarColor(sender) }]}>
          <AppText variant="caption" maxFontSizeMultiplier={1.22} style={styles.avatarText}>
            {initialsFromName(sender)}
          </AppText>
        </View>
      )}

      <View
        style={[
          styles.column,
          { maxWidth: columnMaxW },
          isOwn && styles.columnOwn,
          isCompact && styles.columnCompact,
        ]}
      >
        <View style={[styles.header, isOwn && styles.headerOwn, isCompact && styles.headerCompact]}>
          <AppText variant="label" maxFontSizeMultiplier={1.28} style={styles.senderName} numberOfLines={1}>
            {sender}
          </AppText>
          {time ? (
            <AppText variant="caption" style={styles.timeHeader} accessibilityLabel={`Heure ${time}`}>
              {time}
            </AppText>
          ) : null}
        </View>

        <View
          style={[
            styles.bubble,
            isOwn ? styles.bubbleOwn : styles.bubbleIn,
            isCompact && styles.bubbleCompact,
            isCompact && imageOnly && styles.bubbleCompactImageOnly,
            isCompact && isOwn && styles.bubbleCompactOwn,
            bubbleShadow,
            failed && styles.bubbleFailed,
          ]}
        >
          {showText ? (
            <AppText
              variant="body"
              maxFontSizeMultiplier={1.5}
              style={[styles.bodyText, isOwn && styles.bodyTextOwn, failed && styles.bodyTextFail]}
            >
              {message.content}
            </AppText>
          ) : null}

          {hasImage && imageUri && !imageLoadFailed ? (
            <View
              style={[
                styles.imageGroup,
                isCompact && styles.imageGroupCompact,
                isCompact && isOwn && styles.imageGroupCompactOwn,
                {
                  marginTop: showText ? (isCompact ? 6 : IMAGE_BLOCK_GAP) : 0,
                  marginBottom: isCompact ? 0 : IMAGE_BLOCK_GAP,
                },
              ]}
            >
              <Pressable
                onPress={openImage}
                style={({ pressed }) => [
                  styles.imageWrap,
                  isCompact && styles.imageWrapCompact,
                  {
                    width: imageDisplaySize.width,
                    height: imageDisplaySize.height,
                  },
                  pressed && styles.imagePressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Télécharger ou ouvrir l’image"
              >
                <Image
                  source={{ uri: imageUri }}
                  style={[
                    styles.image,
                    {
                      width: imageDisplaySize.width,
                      height: imageDisplaySize.height,
                    },
                  ]}
                  resizeMode={isCompact ? "cover" : "contain"}
                  onLoad={handleImageLoad}
                  onError={() => setImageLoadFailed(true)}
                />
                <View style={[styles.imageDownloadFab, isCompact && styles.imageDownloadFabCompact]}>
                  <Ionicons name="download-outline" size={isCompact ? 18 : 22} color="#fff" />
                </View>
              </Pressable>
            </View>
          ) : null}
          {hasImage && (imageLoadFailed || !imageUri) ? (
            <Pressable
              onPress={imageUri ? openImage : undefined}
              style={[
                styles.imageErrorBox,
                { marginTop: showText ? IMAGE_BLOCK_GAP : 0 },
              ]}
            >
              <Ionicons name="image-outline" size={20} color={C_BODY} />
              <AppText variant="caption" style={[styles.imageErrorText, isOwn && styles.bodyTextOwn]}>
                {imageLoadFailed
                  ? "Image indisponible. Touchez pour ouvrir dans le navigateur."
                  : "Lien d’image invalide."}
              </AppText>
            </Pressable>
          ) : null}

          {hasPdf ? (
            <Pressable
              onPress={openPdf}
              style={({ pressed }) => [styles.pdfCard, isOwn && styles.pdfCardOwn, pressed && styles.pdfPressed]}
              accessibilityRole="button"
              accessibilityLabel="Ouvrir le document PDF"
            >
              <View style={[styles.pdfIconBox, isOwn && styles.pdfIconBoxOwn]}>
                <Ionicons name="document-text" size={22} color={isOwn ? "#ecfdf5" : C_BUBBLE_OWN} />
              </View>
              <View style={styles.pdfTextBlock}>
                <AppText
                  variant="label"
                  maxFontSizeMultiplier={1.35}
                  style={[styles.pdfTitle, isOwn && styles.pdfTitleOwn]}
                  numberOfLines={2}
                >
                  {message.pdfFilename ?? message.content ?? "Document PDF"}
                </AppText>
                <AppText variant="caption" maxFontSizeMultiplier={1.22} style={[styles.pdfMeta, isOwn && styles.pdfMetaOwn]}>
                  PDF
                </AppText>
              </View>
              <Ionicons name="arrow-down-circle-outline" size={22} color={isOwn ? "rgba(255,255,255,0.85)" : C_BODY} />
            </Pressable>
          ) : null}

          {hasAudio && audioUri ? (
            <View
              style={[
                styles.voiceGroup,
                { marginTop: showText || hasImage || hasPdf ? IMAGE_BLOCK_GAP : 0 },
              ]}
            >
              <VoiceMessageBar uri={audioUri} isOwn={isOwn} />
            </View>
          ) : null}
        </View>

        {isOwn ? (
          <AppText variant="caption" maxFontSizeMultiplier={1.25} style={[styles.status, failed && styles.statusFail]}>
            {failed ? "Échec d’envoi" : "Envoyé"}
          </AppText>
        ) : null}
      </View>

      {isOwn && (
        <View style={[styles.avatar, { backgroundColor: avatarColor(sender) }]}>
          <AppText variant="caption" maxFontSizeMultiplier={1.22} style={styles.avatarText}>
            {initialsFromName(sender)}
          </AppText>
        </View>
      )}
    </View>
  );
});

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
    marginBottom: 12,
    maxWidth: "100%",
    paddingHorizontal: 2,
  },
  rowOwn: {
    flexDirection: "row",
    justifyContent: "flex-end",
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  avatarText: {
    // DS_EXCEPTION: initiales sur pastille colorée (contraste forcé)
    color: "#fff",
    fontWeight: "700",
  },
  rowCompact: {
    marginBottom: 8,
    gap: 8,
  },
  column: {
    maxWidth: 320,
    flexShrink: 1,
    gap: 4,
  },
  columnCompact: {
    gap: 3,
  },
  columnOwn: {
    alignItems: "flex-end",
    alignSelf: "flex-end",
  },
  headerCompact: {
    gap: 4,
    marginBottom: 1,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    flexWrap: "wrap",
  },
  headerOwn: {
    justifyContent: "flex-end",
  },
  senderName: {
    fontWeight: "600",
    color: C_HEADING,
    maxWidth: "70%",
  },
  timeHeader: {
    color: C_BODY,
  },
  bubble: {
    borderRadius: 12,
    padding: 16,
    width: "100%",
    borderWidth: 1,
    borderColor: "transparent",
  },
  bubbleIn: {
    backgroundColor: C_BUBBLE_IN,
    borderColor: C_BORDER,
    borderTopLeftRadius: 4,
  },
  bubbleOwn: {
    backgroundColor: C_BUBBLE_OWN,
    borderColor: C_BUBBLE_OWN,
    borderTopRightRadius: 4,
  },
  bubbleFailed: {
    borderColor: C_FAIL,
    backgroundColor: "#fef2f2",
  },
  bodyText: {
    color: C_HEADING,
  },
  bodyTextOwn: {
    color: "#fff",
  },
  bodyTextFail: {
    color: C_FAIL,
  },
  imageGroup: {
    width: "100%",
    maxWidth: CHAT_IMAGE_INNER_MAX_W,
    alignSelf: "center",
  },
  imageGroupCompact: {
    width: undefined,
    alignSelf: "flex-start",
    maxWidth: CHAT_IMAGE_MAX_W_COMPACT,
  },
  imageGroupCompactOwn: {
    alignSelf: "flex-end",
  },
  imageWrap: {
    borderRadius: 8,
    overflow: "hidden",
    position: "relative",
    maxWidth: "100%",
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.04)",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(0,0,0,0.08)",
  },
  imageWrapCompact: {
    alignSelf: "flex-start",
    borderRadius: 10,
  },
  imagePressed: {
    opacity: 0.88,
  },
  image: {
    backgroundColor: C_BORDER,
    maxWidth: "100%",
  },
  /** Cercle type Flowbite `h-10 w-10` (aperçu / ouvrir, pas seulement au survol). */
  imageDownloadFabCompact: {
    right: 6,
    bottom: 6,
    width: 32,
    height: 32,
    borderRadius: 16,
  },
  imageDownloadFab: {
    position: "absolute",
    right: 10,
    bottom: 10,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.35)",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(255,255,255,0.4)",
    pointerEvents: "none",
  },
  imageErrorBox: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: IMAGE_BLOCK_GAP,
    padding: 10,
    borderRadius: 8,
    backgroundColor: "rgba(0,0,0,0.05)",
    borderWidth: 1,
    borderColor: C_BORDER,
    maxWidth: "100%",
  },
  imageErrorText: {
    flex: 1,
    color: C_BODY,
  },
  pdfCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginTop: 8,
    padding: 10,
    borderRadius: 8,
    backgroundColor: "rgba(0,0,0,0.04)",
  },
  pdfCardOwn: {
    backgroundColor: "rgba(255,255,255,0.2)",
  },
  pdfPressed: {
    opacity: 0.9,
  },
  pdfIconBox: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: "rgba(13,148,136,0.12)",
    alignItems: "center",
    justifyContent: "center",
  },
  pdfIconBoxOwn: {
    backgroundColor: "rgba(255,255,255,0.25)",
  },
  pdfTextBlock: {
    flex: 1,
    minWidth: 0,
  },
  pdfTitle: {
    fontWeight: "600",
    color: C_HEADING,
  },
  pdfTitleOwn: {
    color: "#fff",
  },
  pdfMeta: {
    color: C_BODY,
    marginTop: 2,
  },
  pdfMetaOwn: {
    color: "rgba(255,255,255,0.75)",
  },
  voiceGroup: {
    width: VOICE_GROUP_MAX_W,
    maxWidth: "100%",
    alignSelf: "stretch",
  },
  status: {
    color: C_BODY,
  },
  statusFail: {
    color: C_FAIL,
  },
});
