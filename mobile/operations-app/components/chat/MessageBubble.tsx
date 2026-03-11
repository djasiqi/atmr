import React from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet, Platform } from "react-native";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
  withDelay,
} from "react-native-reanimated";
import { Ionicons } from "@expo/vector-icons";
import { Message } from "@/services/api";
import { resolveMediaUrl } from "@/services/mediaUrl";
import Avatar from "./Avatar";

const BRAND = "#00796b";
const TXT = "#0f172a";
const TXT_SEC = "#6b7280";
const BORDER = "#e5e7eb";

interface Props {
  message: Message;
  currentUserId?: number | null;
  onPressImage?: (uri: string) => void;
  onPressPdf?: (uri: string) => void;
}

export default function MessageBubble({ message, currentUserId, onPressImage, onPressPdf }: Props) {
  const isOwnMessage =
    currentUserId != null &&
    message.sender_id != null &&
    Number(message.sender_id) === Number(currentUserId);

  const opacity = useSharedValue(0);
  const translateY = useSharedValue(10);

  React.useEffect(() => {
    opacity.value = withDelay(30, withTiming(1, { duration: 160 }));
    translateY.value = withDelay(20, withTiming(0, { duration: 160 }));
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }],
  }));

  const formatSize = (bytes: number | null | undefined) => {
    if (!bytes) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const imageUri = resolveMediaUrl(message.image_url || message.image);
  const pdfUri = resolveMediaUrl(message.pdf_url || message.pdf);

  return (
    <Animated.View
      style={[s.wrapper, isOwnMessage ? s.rightWrapper : s.leftWrapper, animatedStyle]}
    >
      {!isOwnMessage && (
        <View style={s.avatarBox}>
          <Avatar photo={null} name={message.sender_name || undefined} size={30} />
        </View>
      )}

      <View style={[s.bubbleCol, isOwnMessage ? s.colRight : s.colLeft]}>
        {!isOwnMessage && message.sender_name && (
          <Text style={s.senderName}>{message.sender_name}</Text>
        )}

        <View style={[s.bubble, isOwnMessage ? s.bubbleOwn : s.bubbleOther]}>
          {imageUri && (
            <TouchableOpacity onPress={() => onPressImage?.(imageUri)} activeOpacity={0.8}>
              <Image source={{ uri: imageUri }} style={s.image} resizeMode="cover" />
            </TouchableOpacity>
          )}

          {pdfUri && (
            <TouchableOpacity onPress={() => onPressPdf?.(pdfUri)} style={s.pdfRow} activeOpacity={0.85}>
              <View style={s.pdfIconBox}>
                <Ionicons name="document-text" size={20} color={BRAND} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[s.pdfName, isOwnMessage && { color: "#fff" }]} numberOfLines={1}>
                  {message.pdf_filename || "Document PDF"}
                </Text>
                {message.pdf_size != null && (
                  <Text style={[s.pdfSize, isOwnMessage && { color: "rgba(255,255,255,0.7)" }]}>
                    {formatSize(message.pdf_size)}
                  </Text>
                )}
              </View>
              <Ionicons name="download-outline" size={16} color={isOwnMessage ? "rgba(255,255,255,0.7)" : TXT_SEC} />
            </TouchableOpacity>
          )}

          {!!message.content && (
            <Text style={[s.text, isOwnMessage ? s.textOwn : s.textOther]}>
              {message.content}
            </Text>
          )}

          {message.timestamp && (
            <Text style={[s.time, isOwnMessage ? s.timeOwn : s.timeOther]}>
              {new Date(message.timestamp).toLocaleTimeString("fr-FR", {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false,
              })}
            </Text>
          )}
        </View>
      </View>
    </Animated.View>
  );
}

const bubbleShadow = Platform.OS === "web"
  ? { boxShadow: "0 1px 2px rgba(0,0,0,0.06)" }
  : { shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.06, shadowRadius: 2, elevation: 1 };

const s = StyleSheet.create({
  wrapper: { flexDirection: "row", marginVertical: 2, paddingHorizontal: 4, alignItems: "flex-end" },
  leftWrapper: { justifyContent: "flex-start" },
  rightWrapper: { justifyContent: "flex-end" },
  avatarBox: { marginRight: 6, marginBottom: 1 },
  bubbleCol: { maxWidth: "78%" },
  colLeft: { alignItems: "flex-start" },
  colRight: { alignItems: "flex-end" },
  senderName: { fontSize: 11, fontWeight: "600", color: BRAND, marginBottom: 2, marginLeft: 4, letterSpacing: 0.1 },
  bubble: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 18,
    ...bubbleShadow,
  },
  bubbleOwn: {
    backgroundColor: BRAND,
    borderBottomRightRadius: 4,
  },
  bubbleOther: {
    backgroundColor: "#FFFFFF",
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: BORDER,
  },
  text: { fontSize: 14, lineHeight: 20 },
  textOwn: { color: "#FFFFFF" },
  textOther: { color: TXT },
  time: { fontSize: 10, alignSelf: "flex-end", marginTop: 3 },
  timeOwn: { color: "rgba(255,255,255,0.65)" },
  timeOther: { color: TXT_SEC },
  image: {
    width: 200,
    height: 200,
    borderRadius: 12,
    marginBottom: 6,
    backgroundColor: "#e5e7eb",
  },
  pdfRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0,121,107,0.06)",
    borderRadius: 10,
    padding: 8,
    marginBottom: 4,
    gap: 8,
  },
  pdfIconBox: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: "rgba(0,121,107,0.1)",
    justifyContent: "center",
    alignItems: "center",
  },
  pdfName: { fontSize: 13, fontWeight: "600", color: TXT, marginBottom: 1 },
  pdfSize: { fontSize: 11, color: TXT_SEC },
});
