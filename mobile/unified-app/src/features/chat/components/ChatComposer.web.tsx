/**
 * Bundle web : aucun import d’`expo-av` (évite l’erreur Metro si le paquet n’est pas installé).
 * L’enregistrement vocal est indisponible sur le web ; utiliser la saisie texte.
 */
import { useCallback, useState } from "react";
import { Pressable, TextInput, View, Platform } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { Ionicons } from "@expo/vector-icons";
import {
  C_BRAND,
  C_FIELD_ICON,
  C_FIELD_PLACEHOLDER,
  C_FIELD_TEXT,
  C_ICON_DISABLED,
  C_MUTED,
  styles,
  textInputWebFix,
  webFieldShellFocusOutline,
} from "./chatComposerStyles";

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

export function ChatComposer({
  value,
  onChangeText,
  onSend,
  onTypingStateChange,
  onPickImage,
  onPickPdf,
  sendLabel = "Envoyer",
  inputAccessibilityLabel = "Saisie du message",
  placeholder = "Écrire un message…",
}: ChatComposerProps) {
  const [inputFocused, setInputFocused] = useState(false);
  const [dialOpen, setDialOpen] = useState(false);

  const closeDial = useCallback(() => setDialOpen(false), []);
  const toggleDial = useCallback(() => setDialOpen((o) => !o), []);

  const runPick = (fn?: () => void) => {
    closeDial();
    fn?.();
  };

  const hasDial = Boolean(onPickImage || onPickPdf);
  const hasText = value.trim().length > 0;

  const a11yPrimary = hasText
    ? sendLabel
    : "Saisissez du texte pour envoyer un message (la saisie vocale est reservee a l’app iOS / Android)";

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
            onFocus={() => setInputFocused(true)}
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
}
