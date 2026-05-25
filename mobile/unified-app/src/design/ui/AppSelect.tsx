import { useCallback, useState } from "react";
import { Modal, Pressable, ScrollView, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { brandPrimary, brandTextMuted } from "../responsive/brand";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { AppText } from "./AppText";

export type AppSelectOption = { value: string; label: string };

export type AppSelectProps = {
  label?: string;
  options: AppSelectOption[];
  value: string | null;
  onChange: (value: string) => void;
  placeholder?: string;
  error?: string;
  helperText?: string;
  disabled?: boolean;
  sheetMaxHeight?: number;
};

export function AppSelect({
  label,
  options,
  value,
  onChange,
  placeholder = "Choisir…",
  error,
  helperText,
  disabled = false,
  sheetMaxHeight = 360,
}: AppSelectProps) {
  const t = useResponsiveTokens();
  const [open, setOpen] = useState(false);
  const selectedLabel = options.find((o) => o.value === value)?.label ?? null;

  const UI_BORDER_SOFT = "rgba(145, 165, 157, 0.38)";
  const borderColor = error ? "#B91C1C" : UI_BORDER_SOFT;
  const close = useCallback(() => setOpen(false), []);

  return (
    <View style={{ gap: t.fieldGap }}>
      {label ? (
        <AppText variant="label" accessibilityRole="text">
          {label}
        </AppText>
      ) : null}

      <Pressable
        accessibilityRole="button"
        disabled={disabled}
        onPress={() => !disabled && setOpen(true)}
        style={{
          minHeight: t.fieldShellMinHeight,
          paddingHorizontal: t.spacingSm + 2,
          borderRadius: t.radiusMd,
          borderWidth: 1,
          borderColor,
          backgroundColor: disabled ? "rgba(241, 245, 244, 0.9)" : "#fff",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          gap: t.spacingSm,
        }}
      >
        <AppText
          variant="body"
          style={{
            flex: 1,
            color: selectedLabel ? undefined : brandTextMuted,
          }}
          numberOfLines={2}
        >
          {selectedLabel ?? placeholder}
        </AppText>
        <Ionicons name="chevron-down" size={18} color={brandTextMuted} />
      </Pressable>

      {error ? (
        <AppText variant="error" accessibilityRole="alert">
          {error}
        </AppText>
      ) : helperText ? (
        <AppText variant="caption">{helperText}</AppText>
      ) : null}

      <Modal transparent animationType="fade" visible={open} onRequestClose={close}>
        <Pressable style={styles.backdrop} onPress={close} accessibilityRole="button">
          <Pressable
            style={[styles.sheet, { borderRadius: t.radiusMd, maxHeight: sheetMaxHeight }]}
            onPress={(e) => e.stopPropagation()}
          >
            <ScrollView keyboardShouldPersistTaps="handled">
              {options.map((opt) => (
                <Pressable
                  key={opt.value}
                  accessibilityRole="menuitem"
                  onPress={() => {
                    onChange(opt.value);
                    close();
                  }}
                  style={{
                    paddingVertical: t.spacingSm,
                    paddingHorizontal: t.spacingMd,
                    borderBottomWidth: StyleSheet.hairlineWidth,
                    borderBottomColor: "rgba(148, 163, 184, 0.35)",
                    backgroundColor: opt.value === value ? "rgba(0, 121, 107, 0.08)" : "transparent",
                  }}
                >
                  <AppText variant="body" style={opt.value === value ? { color: brandPrimary } : undefined}>
                    {opt.label}
                  </AppText>
                </Pressable>
              ))}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(15, 23, 42, 0.35)",
    justifyContent: "center",
    paddingHorizontal: 24,
  },
  sheet: {
    backgroundColor: "#fff",
    overflow: "hidden",
  },
});
