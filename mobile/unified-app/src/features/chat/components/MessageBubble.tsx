import { useEffect, useMemo, useState } from "react";
import { Image, Pressable, Text, useWindowDimensions, View, Platform, StyleSheet } from "react-native";
import * as Linking from "expo-linking";
import { Ionicons } from "@expo/vector-icons";
import { resolveMediaUrl } from "../../../core/api/mediaUrl";
import { SharedChatMessage } from "../types";
import { VoiceMessageBar } from "./VoiceMessageBar";
import { VOICE_GROUP_MAX_W } from "./voiceMessageStyles";

type MessageBubbleProps = {
  message: SharedChatMessage;
  onOpenImage?: (url: string) => void;
  onOpenPdf?: (url: string) => void;
};

const C_HEADING = "#111827";
const C_BODY = "#6b7280";
const C_BUBBLE_IN = "#f3f4f6";
const C_BUBBLE_OWN = "#0d9488";
const C_BORDER = "#e5e7eb";
const C_FAIL = "#b91c1c";
const AVATAR_PALETTE = ["#6366f1", "#0d9488", "#d97706", "#7c3aed", "#db2777", "#2563eb"];

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

function isOwnMessage(message: SharedChatMessage): boolean {
  return message.senderRole?.toUpperCase() === "COMPANY";
}

const DEFAULT_ASPECT = 4 / 3;
/** Colonne type Flowbite `max-w-[320px]`, moins le `p-4` (16+16) de la bulle. */
const CHAT_IMAGE_INNER_MAX_W = 320 - 32;
/** Aperçu compact mais lisible (évite les images trop hautes dans le fil). */
const CHAT_IMAGE_MAX_H = 200;
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

export function MessageBubble({ message, onOpenImage, onOpenPdf }: MessageBubbleProps) {
  const sender = message.senderName ?? message.senderRole ?? "Équipe";
  const isOwn = isOwnMessage(message);
  const time = formatMessageTime(message.timestamp);
  const { width: windowW } = useWindowDimensions();
  const maxImageW = capChatImageWidth(windowW);

  const imageUri = useMemo(() => resolveMediaUrl(message.imageUrl), [message.imageUrl]);
  const pdfUri = useMemo(() => resolveMediaUrl(message.pdfUrl), [message.pdfUrl]);
  const audioUri = useMemo(() => resolveMediaUrl(message.audioUrl), [message.audioUrl]);
  const [imageLoadFailed, setImageLoadFailed] = useState(false);
  const [imageNatural, setImageNatural] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    setImageLoadFailed(false);
    setImageNatural(null);
  }, [imageUri]);

  const imageDisplaySize = useMemo(() => {
    if (imageNatural) {
      return computeImageDisplaySize(imageNatural.w, imageNatural.h, maxImageW, CHAT_IMAGE_MAX_H);
    }
    const w = maxImageW;
    const h = w / DEFAULT_ASPECT;
    return { width: w, height: Math.min(h, CHAT_IMAGE_MAX_H) };
  }, [imageNatural, maxImageW]);

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

  const hasImage = Boolean(message.imageUrl);
  const hasPdf = Boolean(message.pdfUrl);
  const hasAudio = Boolean(message.audioUrl) && Boolean(audioUri);
  const failed = message.content.includes("(echec envoi)");
  const raw = message.content?.trim() ?? "";
  const isAttachmentPlaceholder =
    raw === "(piece jointe)" ||
    (hasImage && raw.toLowerCase() === "image jointe") ||
    (hasPdf && raw.length > 0 && raw === (message.pdfFilename ?? "").trim()) ||
    (hasAudio && (raw === "" || raw.toLowerCase() === "message vocal"));
  const showText = Boolean(raw) && !isAttachmentPlaceholder;

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
    <View style={[styles.row, isOwn && styles.rowOwn]}>
      {!isOwn && (
        <View style={[styles.avatar, { backgroundColor: avatarColor(sender) }]}>
          <Text style={styles.avatarText}>{initialsFromName(sender)}</Text>
        </View>
      )}

      <View style={[styles.column, isOwn && styles.columnOwn]}>
        <View style={[styles.header, isOwn && styles.headerOwn]}>
          <Text style={styles.senderName} numberOfLines={1}>
            {sender}
          </Text>
          {time ? (
            <Text style={styles.timeHeader} accessibilityLabel={`Heure ${time}`}>
              {time}
            </Text>
          ) : null}
        </View>

        <View
          style={[
            styles.bubble,
            isOwn ? styles.bubbleOwn : styles.bubbleIn,
            bubbleShadow,
            failed && styles.bubbleFailed,
          ]}
        >
          {showText ? (
            <Text style={[styles.bodyText, isOwn && styles.bodyTextOwn, failed && styles.bodyTextFail]}>
              {message.content}
            </Text>
          ) : null}

          {hasImage && imageUri && !imageLoadFailed ? (
            <View
              style={[
                styles.imageGroup,
                { marginTop: showText ? IMAGE_BLOCK_GAP : 0, marginBottom: IMAGE_BLOCK_GAP },
              ]}
            >
              <Pressable
                onPress={openImage}
                style={({ pressed }) => [styles.imageWrap, pressed && styles.imagePressed]}
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
                  resizeMode="contain"
                  onLoad={handleImageLoad}
                  onError={() => setImageLoadFailed(true)}
                />
                <View style={styles.imageDownloadFab} pointerEvents="none">
                  <Ionicons name="download-outline" size={22} color="#fff" />
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
              <Text style={[styles.imageErrorText, isOwn && styles.bodyTextOwn]}>
                {imageLoadFailed
                  ? "Image indisponible. Touchez pour ouvrir dans le navigateur."
                  : "Lien d’image invalide."}
              </Text>
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
                <Text style={[styles.pdfTitle, isOwn && styles.pdfTitleOwn]} numberOfLines={2}>
                  {message.pdfFilename ?? message.content ?? "Document PDF"}
                </Text>
                <Text style={[styles.pdfMeta, isOwn && styles.pdfMetaOwn]}>PDF</Text>
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
          <Text style={[styles.status, failed && styles.statusFail]}>
            {failed ? "Échec d’envoi" : "Envoyé"}
          </Text>
        ) : null}
      </View>

      {isOwn && (
        <View style={[styles.avatar, { backgroundColor: avatarColor(sender) }]}>
          <Text style={styles.avatarText}>{initialsFromName(sender)}</Text>
        </View>
      )}
    </View>
  );
}

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
    color: "#fff",
    fontSize: 12,
    fontWeight: "700",
  },
  column: {
    maxWidth: 320,
    flexShrink: 1,
    gap: 4,
  },
  columnOwn: {
    alignItems: "flex-end",
    alignSelf: "flex-end",
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
    fontSize: 14,
    fontWeight: "600",
    color: C_HEADING,
    maxWidth: "70%",
  },
  timeHeader: {
    fontSize: 14,
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
    fontSize: 14,
    lineHeight: 20,
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
  imagePressed: {
    opacity: 0.88,
  },
  image: {
    backgroundColor: C_BORDER,
    maxWidth: "100%",
  },
  /** Cercle type Flowbite `h-10 w-10` (aperçu / ouvrir, pas seulement au survol). */
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
    fontSize: 13,
    color: C_BODY,
    lineHeight: 18,
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
    fontSize: 14,
    fontWeight: "600",
    color: C_HEADING,
  },
  pdfTitleOwn: {
    color: "#fff",
  },
  pdfMeta: {
    fontSize: 12,
    color: C_BODY,
    marginTop: 2,
  },
  pdfMetaOwn: {
    color: "rgba(255,255,255,0.75)",
  },
  voiceGroup: {
    width: "100%",
    maxWidth: VOICE_GROUP_MAX_W,
    alignSelf: "center",
  },
  status: {
    fontSize: 12,
    color: C_BODY,
  },
  statusFail: {
    color: C_FAIL,
  },
});
