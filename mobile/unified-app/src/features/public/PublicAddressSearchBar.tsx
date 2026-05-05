import { forwardRef } from "react";
import { Platform, Pressable, StyleSheet, TextInput, View } from "react-native";
import { AppText } from "../../design/ui/AppText";

/** Conservé pour les appels API (`autocompleteAddress`) — plus de sélecteur UI. */
export type AddressSearchRegion = "CH" | "FR";

const UI_BORDER = "#91A59D";
const UI_DARK = "#163A34";
/** Aligné sur `forgot-password` (`fieldInput` / `#0A8F7A`). */
const BRAND = "#0A8F7A";

/**
 * Hauteur alignée sur le champ email de `forgot-password` (`minHeight: 50`).
 */
const BAR_HEIGHT = 50;

const BAR_RADIUS = 14;

export type PublicAddressSearchBarProps = {
  value: string;
  onChangeText: (v: string) => void;
  onFocus: () => void;
  onBlur: () => void;
  focused: boolean;
  empty: boolean;
  prefilled?: boolean;
  showClear: boolean;
  onClear: () => void;
  placeholder: string;
  accessibilityLabel: string;
  fontSize: number;
  fontWeight: "400" | "500" | "600" | "700";
  clearAccessibilityLabel?: string;
};

export const PublicAddressSearchBar = forwardRef<TextInput, PublicAddressSearchBarProps>(
  function PublicAddressSearchBar(props, ref) {
    const hitSlop = { top: 8, bottom: 8, left: 4, right: 4 };
    const inputFontSize = Math.min(props.fontSize, 16);
    const inputLineHeight = Math.round(inputFontSize * 1.25);

    return (
      <View
        style={[
          styles.bar,
          props.empty && !props.focused ? styles.barEmpty : null,
          props.focused ? styles.barFocused : null,
        ]}
      >
        <View style={styles.inputWrap}>
          <TextInput
            ref={ref}
            value={props.value}
            onChangeText={props.onChangeText}
            onFocus={props.onFocus}
            onBlur={props.onBlur}
            placeholder={props.placeholder}
            placeholderTextColor="#91A59D"
            accessibilityLabel={props.accessibilityLabel}
            autoComplete="off"
            autoCorrect={false}
            spellCheck={false}
            textContentType="none"
            importantForAutofill="no"
            selectionColor="rgba(10, 143, 122, 0.35)"
            underlineColorAndroid="transparent"
            style={[
              styles.input,
              props.prefilled ? styles.inputPrefilled : null,
              props.showClear ? styles.inputWithClear : null,
              styles.inputTypography,
              {
                fontSize: inputFontSize,
                fontWeight: props.fontWeight,
                lineHeight: inputLineHeight,
              },
            ]}
          />
          {props.showClear ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={props.clearAccessibilityLabel ?? "Effacer le texte"}
              onPress={props.onClear}
              style={({ pressed }) => [styles.clearBtn, pressed ? styles.clearBtnPressed : null]}
              hitSlop={hitSlop}
            >
              <AppText variant="label" style={styles.clearText}>
                ×
              </AppText>
            </Pressable>
          ) : null}
        </View>
      </View>
    );
  }
);

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    alignItems: "stretch",
    height: BAR_HEIGHT,
    minHeight: BAR_HEIGHT,
    maxHeight: BAR_HEIGHT,
    borderRadius: BAR_RADIUS,
    borderWidth: 1,
    borderColor: UI_BORDER,
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
    ...Platform.select({
      web: {
        boxShadow: "0 1px 3px rgba(22,58,52,0.07)",
        boxSizing: "border-box" as const,
      },
      default: {},
    }),
  },
  barEmpty: {
    borderWidth: 1,
    borderColor: UI_BORDER,
    backgroundColor: "#FFFFFF",
  },
  barFocused: {
    borderColor: BRAND,
    backgroundColor: "#FFFFFF",
    ...Platform.select({
      web: { boxShadow: "0 2px 6px rgba(10, 143, 122, 0.12)" },
      default: {
        shadowColor: BRAND,
        shadowOpacity: 0.12,
        shadowRadius: 6,
        shadowOffset: { width: 0, height: 2 },
        elevation: 2,
      },
    }),
  },
  inputWrap: {
    flex: 1,
    minWidth: 0,
    position: "relative",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },
  input: {
    flexGrow: 1,
    alignSelf: "stretch",
    width: "100%",
    color: UI_DARK,
    paddingLeft: 14,
    paddingRight: 14,
    borderWidth: 0,
    ...Platform.select({
      web: {
        outlineStyle: "none" as const,
        caretColor: BRAND,
        boxSizing: "border-box" as const,
      },
      android: { textAlignVertical: "center" },
      default: {},
    }),
  },
  inputTypography: {
    paddingVertical: 0,
    ...Platform.select({
      web: {
        paddingTop: 10,
        paddingBottom: 10,
      },
      ios: {
        paddingTop: 12,
        paddingBottom: 12,
      },
      android: {
        paddingTop: 0,
        paddingBottom: 0,
        minHeight: BAR_HEIGHT - 2,
      },
      default: {},
    }),
  },
  inputWithClear: {
    paddingRight: 30,
  },
  inputPrefilled: {
    color: "#7B8E86",
  },
  clearBtn: {
    position: "absolute",
    right: 4,
    top: "50%",
    marginTop: -10,
    width: 20,
    height: 20,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(145,165,157,0.18)",
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.28)",
  },
  clearBtnPressed: {
    backgroundColor: "rgba(145,165,157,0.28)",
  },
  clearText: {
    // DS_EXCEPTION: bouton × compact sur fond gris — couleur palette recherche publique
    color: "#45655D",
    marginTop: -2,
  },
});
