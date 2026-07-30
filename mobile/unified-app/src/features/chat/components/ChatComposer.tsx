import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Linking, Pressable, TextInput, View, Platform } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { Ionicons } from "@expo/vector-icons";
import { useChatVoiceRecorder } from "../services/audioAdapter";
import {
  C_BRAND,
  C_FIELD_ICON,
  C_FIELD_PLACEHOLDER,
  C_FIELD_TEXT,
  C_ICON_DISABLED,
  C_MUTED,
  C_RECORDING,
  styles,
  textInputWebFix,
  webFieldShellFocusOutline,
} from "./chatComposerStyles";

type ChatComposerProps = {
  value: string;
  onChangeText: (value: string) => void;
  onSend: () => void;
  onInputFocus?: () => void;
  onTypingStateChange?: (typing: boolean) => void;
  onPickImage?: () => void;
  onPickPdf?: () => void;
  onVoiceMessage?: (localUri: string) => void | Promise<void>;
  onVoiceError?: (message: string, kind?: "permission" | "generic") => void;
  sendLabel?: string;
  inputAccessibilityLabel?: string;
  placeholder?: string;
};

const MIN_VOICE_MS = 450;

export const ChatComposer = memo(function ChatComposer({
  value,
  onChangeText,
  onSend,
  onInputFocus,
  onTypingStateChange,
  onPickImage,
  onPickPdf,
  onVoiceMessage,
  onVoiceError,
  sendLabel = "Envoyer",
  inputAccessibilityLabel = "Saisie du message",
  placeholder = "Écrire un message…",
}: ChatComposerProps) {
  const [inputFocused, setInputFocused] = useState(false);
  const [dialOpen, setDialOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);

  const { startRecording: startVoiceSession, stopRecording: stopVoiceSession, abortRecording } =
    useChatVoiceRecorder();
  const abortRecordingRef = useRef(abortRecording);
  abortRecordingRef.current = abortRecording;
  /** Incrémenté à chaque pression / relâchement pour annuler un démarrage d’enregistrement encore en cours. */
  const voiceInteractionEpochRef = useRef(0);
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
    if (voiceBusy) return;
    setVoiceBusy(true);
    setIsRecording(false);
    const elapsed = Date.now() - pressStartMsRef.current;
    const result = await stopVoiceSession();
    try {
      if (!result.ok) {
        if (result.reason !== "no_active_recording" && result.reason !== "aborted") {
          onVoiceError?.("Impossible d'enregistrer le message vocal.");
        }
        return;
      }
      if (!result.data || elapsed < MIN_VOICE_MS) {
        onVoiceError?.("Enregistrement trop court. Appuyez sur le micro et parlez.");
        return;
      }
      if (onVoiceMessage) {
        await onVoiceMessage(result.data);
      }
    } finally {
      setVoiceBusy(false);
    }
  }, [onVoiceError, onVoiceMessage, stopVoiceSession, voiceBusy]);

  const startRecording = useCallback(
    async (attemptEpoch: number) => {
      if (!onVoiceMessage) return;
      pressStartMsRef.current = Date.now();
      try {
        const started = await startVoiceSession({
          isAborted: () => voiceInteractionEpochRef.current !== attemptEpoch,
        });
        if (!started.ok) {
          setIsRecording(false);
          if (started.reason === "permission_denied") {
            void Linking.openSettings();
            onVoiceError?.(
              "Activez le micro dans les réglages du téléphone pour envoyer des messages vocaux.",
              "permission"
            );
          } else if (started.reason !== "aborted") {
            onVoiceError?.("Impossible de démarrer l'enregistrement vocal.");
          }
          return;
        }
        setIsRecording(true);
      } catch {
        await abortRecording();
        setIsRecording(false);
      }
    },
    [abortRecording, onVoiceError, onVoiceMessage, startVoiceSession]
  );

  const onVoicePress = useCallback(() => {
    if (voiceBusy) return;
    if (isRecording) {
      void finalizeRecording();
      return;
    }
    closeDial();
    voiceInteractionEpochRef.current += 1;
    const attemptEpoch = voiceInteractionEpochRef.current;
    void startRecording(attemptEpoch);
  }, [closeDial, finalizeRecording, isRecording, startRecording, voiceBusy]);

  useEffect(() => {
    return () => {
      voiceInteractionEpochRef.current += 1;
      void abortRecordingRef.current();
    };
  }, []);

  const sendCircleBg = isRecording ? C_RECORDING : C_BRAND;

  const a11yPrimary = hasText
    ? sendLabel
    : voiceAllowed
      ? isRecording
        ? "Appuyez pour envoyer le message vocal"
        : "Appuyez pour enregistrer un message vocal"
      : "Saisissez du texte pour envoyer un message";

  return (
    <View style={styles.root}>
      <View style={styles.mainRow}>
        <View
          style={[
            styles.fieldShell,
            inputFocused && styles.fieldShellFocused,
            webFieldShellFocusOutline(inputFocused),
          ]}
        >
          {dialOpen && hasDial ? (
            <View style={styles.dialMenu} accessibilityViewIsModal>
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
            onFocus={() => {
              setInputFocused(true);
              onInputFocus?.();
            }}
            onBlur={() => setInputFocused(false)}
            placeholder={placeholder}
            placeholderTextColor={C_FIELD_PLACEHOLDER}
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
            >
              <Ionicons
                name={dialOpen ? "close" : "attach-outline"}
                size={dialOpen ? 22 : 24}
                color={dialOpen ? C_FIELD_TEXT : C_FIELD_ICON}
              />
            </Pressable>
          ) : null}
        </View>

        {hasText ? (
          <Pressable
            onPress={onSend}
            style={({ pressed }) => [
              styles.sendCircle,
              { backgroundColor: C_BRAND },
              pressed && styles.sendCirclePressed,
            ]}
            accessibilityRole="button"
            accessibilityLabel={a11yPrimary}
          >
            <Ionicons name="send" size={22} color="#fff" />
          </Pressable>
        ) : voiceAllowed ? (
          <Pressable
            onPress={onVoicePress}
            disabled={voiceBusy}
            hitSlop={8}
            style={({ pressed }) => [
              styles.sendCircle,
              { backgroundColor: sendCircleBg },
              pressed && !isRecording && !voiceBusy && styles.sendCirclePressed,
              voiceBusy && styles.sendCircleBusy,
            ]}
            accessibilityRole="button"
            accessibilityLabel={a11yPrimary}
            accessibilityState={{ disabled: voiceBusy, selected: isRecording }}
          >
            <Ionicons
              name={voiceBusy ? "hourglass-outline" : isRecording ? "stop" : "mic"}
              size={22}
              color="#fff"
            />
          </Pressable>
        ) : (
          <View
            style={[styles.sendCircle, styles.sendCircleDisabled]}
            accessibilityRole="button"
            accessibilityLabel="La saisie vocale n’est pas disponible sur le web. Utilisez le clavier."
          >
            <Ionicons name="mic-off-outline" size={22} color={C_ICON_DISABLED} />
          </View>
        )}
      </View>
    </View>
  );
});
