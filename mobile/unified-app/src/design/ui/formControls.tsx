import { Fragment } from "react";
import { Pressable, StyleSheet, View, type StyleProp, type ViewStyle } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { brandPrimary } from "../responsive/brand";
import { appTextErrorColor } from "./typography";
import { AppInput, type AppInputProps } from "./AppInput";
import { AppText } from "./AppText";

/**
 * Gabarit du formulaire « Nouveau client », calé sur les contrôles de
 * « Créer une réservation » (déclencheur client / adresse : 35 px, texte 13).
 */
export const FORM_CONTROL_HEIGHT = 36;
export const FORM_CONTROL_RADIUS = 12;
export const FORM_CONTROL_PADDING_H = 10;
export const FORM_CONTROL_FONT_SIZE = 13;
export const FORM_LABEL_SIZE = 12;
export const FORM_LABEL_GAP = 4;
export const FORM_TEXTAREA_MIN_HEIGHT = 76;
export const FORM_SEGMENT_HEIGHT = 32;
export const FORM_CHECKBOX_SIZE = 22;
export const FORM_CHECKBOX_ROW_MIN_HEIGHT = 44;

const BORDER = "rgba(145, 165, 157, 0.38)";
const INK = "#1E293B";
const INK_SEC = "#64748B";
const FILL_DISABLED = "rgba(241, 245, 244, 0.9)";

export const formControlShell = {
  height: FORM_CONTROL_HEIGHT,
  minHeight: FORM_CONTROL_HEIGHT,
  borderRadius: FORM_CONTROL_RADIUS,
  paddingHorizontal: FORM_CONTROL_PADDING_H,
  borderWidth: 1,
  borderColor: BORDER,
  backgroundColor: "#FFFFFF",
} as const;

export const formControlInputStyle = {
  minHeight: 18,
  paddingVertical: 0,
  fontSize: FORM_CONTROL_FONT_SIZE,
  lineHeight: 17,
} as const;

const s = StyleSheet.create({
  field: { gap: FORM_LABEL_GAP, minWidth: 0 },
  label: {
    fontSize: FORM_LABEL_SIZE,
    lineHeight: 16,
    fontWeight: "600",
    color: INK_SEC,
  },
  required: { color: "#DC2626" },
  shell: { ...formControlShell },
  input: {
    minHeight: 18,
    paddingVertical: 0,
    fontSize: FORM_CONTROL_FONT_SIZE,
    lineHeight: 17,
    color: INK,
  },
  areaShell: {
    minHeight: FORM_TEXTAREA_MIN_HEIGHT,
    borderRadius: FORM_CONTROL_RADIUS,
    paddingHorizontal: FORM_CONTROL_PADDING_H,
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: "#FFFFFF",
    alignItems: "flex-start",
    paddingVertical: 12,
  },
  areaInput: {
    minHeight: FORM_TEXTAREA_MIN_HEIGHT - 24,
    paddingVertical: 0,
    textAlignVertical: "top",
  },
  select: {
    ...formControlShell,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  selectPressed: { borderColor: brandPrimary },
  selectDisabled: { backgroundColor: FILL_DISABLED },
  selectValue: { flex: 1, fontSize: FORM_CONTROL_FONT_SIZE, lineHeight: 17, fontWeight: "600", color: INK },
  placeholder: { flex: 1, fontSize: FORM_CONTROL_FONT_SIZE, lineHeight: 17, color: INK_SEC },
  error: { marginTop: 2 },
  segment: {
    flexDirection: "row",
    height: FORM_SEGMENT_HEIGHT,
    borderRadius: 11,
    overflow: "hidden",
    backgroundColor: "#FFFFFF",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(0, 121, 107, 0.28)",
  },
  segmentDivider: {
    width: StyleSheet.hairlineWidth,
    alignSelf: "stretch",
    backgroundColor: "rgba(15, 23, 42, 0.08)",
  },
  segmentItem: { flex: 1, alignItems: "center", justifyContent: "center" },
  segmentItemOn: { backgroundColor: "rgba(0, 121, 107, 0.14)" },
  segmentText: { color: INK_SEC, fontSize: FORM_CONTROL_FONT_SIZE, fontWeight: "600" },
  segmentTextOn: { color: brandPrimary, fontWeight: "700", fontSize: FORM_CONTROL_FONT_SIZE },
  checkRow: {
    minHeight: FORM_CHECKBOX_ROW_MIN_HEIGHT,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  box: {
    width: FORM_CHECKBOX_SIZE,
    height: FORM_CHECKBOX_SIZE,
    borderRadius: 4,
    borderWidth: 1.5,
    borderColor: "rgba(100, 116, 139, 0.7)",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },
  boxOn: { backgroundColor: brandPrimary, borderColor: brandPrimary },
  checkCopy: { flex: 1, minWidth: 0 },
  checkLabel: { fontSize: FORM_CONTROL_FONT_SIZE, lineHeight: 17, color: INK },
  hint: { fontSize: 12, lineHeight: 16, color: INK_SEC },
});

export function FormLabel({ label, required }: { label: string; required?: boolean }) {
  return (
    <AppText variant="label" style={s.label}>
      {label}
      {required ? <AppText variant="label" style={s.required}> *</AppText> : null}
    </AppText>
  );
}

type TextFieldProps = AppInputProps & { required?: boolean };

/** Champ texte au gabarit formulaire. Le focus et l’erreur restent ceux de `AppInput`. */
export function TextField({ label, required, multiline, error, containerStyle, shellStyle, style, ...props }: TextFieldProps) {
  return (
    <View style={[s.field, containerStyle]}>
      {label ? <FormLabel label={label} required={required} /> : null}
      <AppInput
        {...props}
        multiline={multiline}
        error={error}
        shellStyle={[multiline ? s.areaShell : s.shell, shellStyle]}
        style={[multiline ? s.areaInput : s.input, style]}
      />
    </View>
  );
}

type SelectFieldProps = {
  label?: string;
  required?: boolean;
  value: string;
  placeholder: string;
  onPress: () => void;
  disabled?: boolean;
  error?: string;
  style?: StyleProp<ViewStyle>;
};

/** Même boîte qu’un champ texte, avec un chevron. */
export function SelectField({ label, required, value, placeholder, onPress, disabled, error, style }: SelectFieldProps) {
  const filled = value.trim().length > 0;
  return (
    <View style={s.field}>
      {label ? <FormLabel label={label} required={required} /> : null}
      <Pressable
        onPress={onPress}
        disabled={disabled}
        accessibilityRole="button"
        style={({ pressed }) => [
          s.select,
          pressed && !disabled ? s.selectPressed : null,
          disabled ? s.selectDisabled : null,
          error ? { borderColor: appTextErrorColor } : null,
          style,
        ]}
      >
        <AppText variant="body" numberOfLines={1} style={filled ? s.selectValue : s.placeholder}>
          {filled ? value : placeholder}
        </AppText>
        <Ionicons name="chevron-forward" size={18} color={disabled ? "#94A3B8" : INK_SEC} />
      </Pressable>
      {error ? (
        <AppText variant="error" style={s.error}>
          {error}
        </AppText>
      ) : null}
    </View>
  );
}

export type SegmentOption<T extends string> = { value: T; label: string };

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
}: {
  options: SegmentOption<T>[];
  value: T | "";
  onChange: (value: T) => void;
}) {
  return (
    <View style={s.segment}>
      {options.map((option, index) => {
        const active = option.value === value;
        return (
          <Fragment key={option.value}>
            {index > 0 ? <View style={s.segmentDivider} /> : null}
            <Pressable
              onPress={() => onChange(option.value)}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              style={[s.segmentItem, active ? s.segmentItemOn : null]}
            >
              <AppText variant="body" style={active ? s.segmentTextOn : s.segmentText}>
                {option.label}
              </AppText>
            </Pressable>
          </Fragment>
        );
      })}
    </View>
  );
}

export function CheckboxRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <Pressable
      onPress={() => onChange(!checked)}
      accessibilityRole="checkbox"
      accessibilityState={{ checked }}
      style={s.checkRow}
    >
      <View style={[s.box, checked ? s.boxOn : null]}>
        {checked ? <Ionicons name="checkmark" size={14} color="#FFFFFF" /> : null}
      </View>
      <View style={s.checkCopy}>
        <AppText variant="body" style={s.checkLabel}>{label}</AppText>
        {hint ? (
          <AppText variant="caption" style={s.hint}>{hint}</AppText>
        ) : null}
      </View>
    </Pressable>
  );
}
