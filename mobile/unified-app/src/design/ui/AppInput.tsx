import { forwardRef, useCallback, useState, type ReactNode } from "react";
import { TextInput, type TextInputProps, View, type ViewStyle } from "react-native";
import { brandPrimary, brandText, brandTextMuted } from "../responsive/brand";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { AppText } from "./AppText";
import { appTextErrorColor } from "./typography";

const UI_BORDER_SOFT = "rgba(145, 165, 157, 0.38)";

export type AppInputProps = TextInputProps & {
  label?: string;
  error?: string;
  helperText?: string;
  rightSlot?: ReactNode;
  containerStyle?: ViewStyle;
};

export const AppInput = forwardRef<TextInput, AppInputProps>(function AppInput(
  { label, error, helperText, rightSlot, containerStyle, style, editable = true, onFocus, onBlur, ...rest },
  ref
) {
  const t = useResponsiveTokens();
  const [focused, setFocused] = useState(false);

  const onF = useCallback(
    (e: Parameters<NonNullable<TextInputProps["onFocus"]>>[0]) => {
      setFocused(true);
      onFocus?.(e);
    },
    [onFocus]
  );
  const onB = useCallback(
    (e: Parameters<NonNullable<TextInputProps["onBlur"]>>[0]) => {
      setFocused(false);
      onBlur?.(e);
    },
    [onBlur]
  );

  const borderColor = error ? appTextErrorColor : focused ? brandPrimary : UI_BORDER_SOFT;
  const shellMinH = t.fieldShellMinHeight;
  const inputMinH = t.fieldTextInputMinHeight;
  const inputPadV = t.fieldTextInputPaddingV;

  const disabled = editable === false;

  return (
    <View style={[{ gap: t.fieldGap }, containerStyle]}>
      {label ? (
        <AppText variant="label" accessibilityRole="text">
          {label}
        </AppText>
      ) : null}
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          minHeight: shellMinH,
          borderWidth: 1,
          borderColor,
          borderRadius: t.radiusMd,
          paddingHorizontal: t.spacingSm + 2,
          backgroundColor: disabled ? "rgba(241, 245, 244, 0.9)" : "#fff",
        }}
      >
        <TextInput
          ref={ref}
          editable={editable}
          placeholderTextColor={brandTextMuted}
          onFocus={onF}
          onBlur={onB}
          style={[
            {
              flex: 1,
              minHeight: inputMinH,
              paddingVertical: inputPadV,
              fontSize: t.bodyFontSize,
              lineHeight: Math.round(t.bodyFontSize * t.bodyLineHeightRatio),
              color: brandText,
            },
            style,
          ]}
          {...rest}
        />
        {rightSlot}
      </View>
      {error ? (
        <AppText variant="error" accessibilityRole="alert">
          {error}
        </AppText>
      ) : helperText ? (
        <AppText variant="caption">{helperText}</AppText>
      ) : null}
    </View>
  );
});
