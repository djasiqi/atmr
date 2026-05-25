import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Pressable, TextInput, View, Platform } from "react-native";
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
  onVoiceMessage?: (localUri: string) => void;
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
  sendLabel = "Envoyer",
  inputAccessibilityLabel = "Saisie du message",
  placeholder = "Écrire un message…",
}: ChatComposerProps) {
  const [inputFocused, setInputFocused] = useState(false);
  const [dialOpen, setDialOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const { startRecording: startVoiceSession, stopRecording: stopVoiceSession, abortRecording } =
    useChatVoiceRecorder();
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
    setIsRecording(false);
    const elapsed = Date.now() - pressStartMsRef.current;
    const result = await stopVoiceSession();
    if (result.ok && result.data && elapsed >= MIN_VOICE_MS && onVoiceMessage) {
      onVoiceMessage(result.data);
    }
  }, [onVoiceMessage, stopVoiceSession]);

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
          return;
        }
        setIsRecording(true);
      } catch {
        await abortRecording();
        setIsRecording(false);
      }
    },
    [abortRecording, onVoiceMessage, startVoiceSession]
  );

  const onVoicePressIn = useCallback(() => {
    closeDial();
    voiceInteractionEpochRef.current += 1;
    const attemptEpoch = voiceInteractionEpochRef.current;
    void startRecording(attemptEpoch);
  }, [closeDial, startRecording]);

  const onVoicePressOut = useCallback(() => {
    voiceInteractionEpochRef.current += 1;
    void finalizeRecording();
  }, [finalizeRecording]);

  useEffect(() => {
    return () => {
      voiceInteractionEpochRef.current += 1;
      void abortRecording();
    };
  }, [abortRecording]);

  const sendCircleBg = isRecording ? C_RECORDING : C_BRAND;

  const a11yPrimary = hasText
    ? sendLabel
    : voiceAllowed
      ? "Maintenir pour enregistrer un message vocal"
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
            onPressIn={onVoicePressIn}
            onPressOut={onVoicePressOut}
            style={({ pressed }) => [
              styles.sendCircle,
              { backgroundColor: sendCircleBg },
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
          >
            <Ionicons name="mic-off-outline" size={22} color={C_ICON_DISABLED} />
          </View>
        )}
      </View>
    </View>
  );
});
