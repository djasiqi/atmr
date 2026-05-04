import { useCallback, useEffect, useRef, useState } from "react";
import { Pressable, TextInput, View, Platform } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { Audio } from "expo-av";
import { Ionicons } from "@expo/vector-icons";
import {
  C_BRAND,
  C_MUTED,
  C_RECORDING,
  C_TEXT,
  styles,
  textInputWebFix,
} from "./chatComposerStyles";

type ExpoAudioRecording = InstanceType<typeof Audio.Recording>;

type ChatComposerProps = {
  value: string;
  onChangeText: (value: string) => void;
  onSend: () => void;
  onTypingStateChange?: (typing: boolean) => void;
  onPickImage?: () => void;
  onPickPdf?: () => void;
  onVoiceMessage?: (localUri: string) => void;
  sendLabel?: string;
  inputAccessibilityLabel?: string;
  placeholder?: string;
};

const MIN_VOICE_MS = 450;

export function ChatComposer({
  value,
  onChangeText,
  onSend,
  onTypingStateChange,
  onPickImage,
  onPickPdf,
  onVoiceMessage,
  sendLabel = "Envoyer",
  inputAccessibilityLabel = "Saisie du message",
  placeholder = "Écrire un message…",
}: ChatComposerProps) {
  const [inputFocused, setInputFocused] = useState(false);
  const [dialOpen, setDialOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const recordingRef = useRef<ExpoAudioRecording | null>(null);
  const cancelStartRef = useRef(false);
  const pressStartMsRef = useRef(0);

  const closeDial = useCallback(() => setDialOpen(false), []);
  const toggleDial = useCallback(() => setDialOpen((o) => !o), []);

  const runPick = (fn?: () => void) => {
    closeDial();
    fn?.();
  };

  const hasDial = Boolean(onPickImage || onPickPdf);
  const trimmed = value.trim();
  const hasText = trimmed.length > 0;
  const voiceAllowed = Boolean(onVoiceMessage) && Platform.OS !== "web";

  const finalizeRecording = useCallback(async () => {
    const rec = recordingRef.current;
    recordingRef.current = null;
    setIsRecording(false);
    if (!rec) return;
    try {
      await rec.stopAndUnloadAsync();
      const uri = rec.getURI();
      const elapsed = Date.now() - pressStartMsRef.current;
      if (uri && elapsed >= MIN_VOICE_MS && onVoiceMessage) {
        onVoiceMessage(uri);
      }
    } catch {
      /* ignore */
    }
  }, [onVoiceMessage]);

  const startRecording = useCallback(async () => {
    if (!onVoiceMessage) return;
    cancelStartRef.current = false;
    pressStartMsRef.current = Date.now();
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted || cancelStartRef.current) return;
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      if (cancelStartRef.current) return;
      const { recording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      if (cancelStartRef.current) {
        await recording.stopAndUnloadAsync();
        return;
      }
      recordingRef.current = recording;
      setIsRecording(true);
    } catch {
      recordingRef.current = null;
      setIsRecording(false);
    }
  }, [onVoiceMessage]);

  const onVoicePressIn = useCallback(() => {
    closeDial();
    void startRecording();
  }, [closeDial, startRecording]);

  const onVoicePressOut = useCallback(() => {
    cancelStartRef.current = true;
    void finalizeRecording();
  }, [finalizeRecording]);

  useEffect(() => {
    return () => {
      cancelStartRef.current = true;
      const rec = recordingRef.current;
      recordingRef.current = null;
      if (rec) void rec.stopAndUnloadAsync();
    };
  }, []);

  const fieldShadow =
    Platform.OS === "web"
      ? { boxShadow: "0 1px 2px 0 rgba(0,0,0,0.04)" as const }
      : { elevation: 0 };

  const sendCircleBg = isRecording ? C_RECORDING : C_BRAND;
  const sendCircleBorder = isRecording ? C_RECORDING : C_BRAND;

  const a11yPrimary = hasText
    ? sendLabel
    : voiceAllowed
      ? "Maintenir pour enregistrer un message vocal"
      : "Saisissez du texte pour envoyer un message";

  return (
    <View style={styles.root} pointerEvents="box-none">
      <View style={styles.mainRow}>
        <View
          style={[
            styles.fieldShell,
            fieldShadow,
            inputFocused && styles.fieldShellFocused,
          ]}
        >
          {dialOpen && hasDial ? (
            <View style={styles.dialMenu} pointerEvents="box-none" accessibilityViewIsModal>
              {onPickImage ? (
                <Pressable
                  onPress={() => runPick(onPickImage)}
                  style={({ pressed }) => [styles.actionChip, pressed && styles.pressedOp]}
                  accessibilityLabel="Envoyer une image"
                >
                  <Ionicons name="image-outline" size={16} color={C_MUTED} />
                  <AppText variant="caption" style={styles.actionLabel} numberOfLines={1}>
                    Image
                  </AppText>
                </Pressable>
              ) : null}
              {onPickPdf ? (
                <Pressable
                  onPress={() => runPick(onPickPdf)}
                  style={({ pressed }) => [styles.actionChip, pressed && styles.pressedOp]}
                  accessibilityLabel="Envoyer un document PDF"
                >
                  <Ionicons name="document-text-outline" size={16} color={C_MUTED} />
                  <AppText variant="caption" style={styles.actionLabel} numberOfLines={1}>
                    PDF
                  </AppText>
                </Pressable>
              ) : null}
            </View>
          ) : null}

          <TextInput
            value={value}
            accessibilityLabel={inputAccessibilityLabel}
            onChangeText={(next) => {
              onChangeText(next);
              onTypingStateChange?.(next.trim().length > 0);
            }}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
            placeholder={placeholder}
            placeholderTextColor={C_MUTED}
            returnKeyType="default"
            underlineColorAndroid="transparent"
            style={[
              styles.textInput,
              hasDial && styles.textInputWithAttach,
              Platform.OS === "web" && textInputWebFix,
            ]}
          />

          {hasDial ? (
            <Pressable
              onPress={toggleDial}
              style={({ pressed }) => [styles.attachBtn, pressed && styles.attachBtnPressed]}
              accessibilityRole="button"
              accessibilityLabel={dialOpen ? "Fermer les actions de pièces jointes" : "Ouvrir les actions de pièces jointes"}
              accessibilityState={{ expanded: dialOpen }}
              hitSlop={6}
            >
              <Ionicons
                name={dialOpen ? "close" : "attach-outline"}
                size={dialOpen ? 22 : 24}
                color={dialOpen ? C_TEXT : C_MUTED}
              />
            </Pressable>
          ) : null}
        </View>

        {hasText ? (
          <Pressable
            onPress={onSend}
            style={({ pressed }) => [
              styles.sendCircle,
              { backgroundColor: C_BRAND, borderColor: C_BRAND },
              pressed && styles.sendCirclePressed,
            ]}
            accessibilityRole="button"
            accessibilityLabel={a11yPrimary}
          >
            <Ionicons name="send" size={22} color="#fff" />
          </Pressable>
        ) : voiceAllowed ? (
          <Pressable
            onPressIn={onVoicePressIn}
            onPressOut={onVoicePressOut}
            style={({ pressed }) => [
              styles.sendCircle,
              { backgroundColor: sendCircleBg, borderColor: sendCircleBorder },
              pressed && !isRecording && styles.sendCirclePressed,
            ]}
            accessibilityRole="button"
            accessibilityLabel={a11yPrimary}
          >
            <Ionicons name={isRecording ? "stop" : "mic"} size={22} color="#fff" />
          </Pressable>
        ) : (
          <View
            style={[styles.sendCircle, styles.sendCircleDisabled]}
            accessibilityRole="button"
            accessibilityLabel="La saisie vocale n’est pas disponible sur le web. Utilisez le clavier."
            pointerEvents="none"
          >
            <Ionicons name="mic-off-outline" size={22} color="rgba(255,255,255,0.75)" />
          </View>
        )}
      </View>
    </View>
  );
}
