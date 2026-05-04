import { Ionicons } from "@expo/vector-icons";
import { forwardRef, useCallback, useState } from "react";
import {
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  TextInput,
  View,
} from "react-native";
import { AppText } from "../../design/ui/AppText";

export type AddressSearchRegion = "CH" | "FR";

const UI_BORDER = "#91A59D";
const UI_MUTED = "#5F7369";
const UI_DARK = "#163A34";
const BRAND = "#0B9A84";

/** Hauteur fixe de la barre (sélecteur pays + champ), ex. accueil public. */
const BAR_HEIGHT = 30;

const BAR_RADIUS = 8;
/** Largeur fixe du bloc pays : évite les sauts entre 🇨🇭 et 🇫🇷. */
const REGION_SEGMENT_WIDTH = 44;

const REGION_META: Record<
  AddressSearchRegion,
  { flag: string; label: string; hint: string }
> = {
  CH: { flag: "🇨🇭", label: "CH", hint: "Suisse" },
  FR: { flag: "🇫🇷", label: "FR", hint: "France" },
};

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
  region: AddressSearchRegion;
  onRegionChange: (r: AddressSearchRegion) => void;
  clearAccessibilityLabel?: string;
};

export const PublicAddressSearchBar = forwardRef<TextInput, PublicAddressSearchBarProps>(
  function PublicAddressSearchBar(props, ref) {
    const [regionModalVisible, setRegionModalVisible] = useState(false);

    const closeModal = useCallback(() => setRegionModalVisible(false), []);

    const meta = REGION_META[props.region];
    const hitSlop = { top: 8, bottom: 8, left: 4, right: 4 };

    return (
      <>
        <View
          style={[
            styles.bar,
            props.empty && !props.focused ? styles.barEmpty : null,
            props.focused ? styles.barFocused : null,
          ]}
        >
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={`Pays de recherche, ${meta.hint} (${meta.label})`}
            onPress={() => setRegionModalVisible(true)}
            style={({ pressed }) => [styles.regionBtn, pressed ? styles.regionBtnPressed : null]}
            hitSlop={hitSlop}
          >
            <AppText variant="label" style={styles.regionFlag}>
              {meta.flag}
            </AppText>
            <Ionicons
              name="chevron-down"
              size={11}
              color={props.focused ? BRAND : UI_MUTED}
            />
          </Pressable>

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
              selectionColor="rgba(11, 154, 132, 0.35)"
              underlineColorAndroid="transparent"
              style={[
                styles.input,
                props.prefilled ? styles.inputPrefilled : null,
                props.showClear ? styles.inputWithClear : null,
                styles.inputTypography,
                {
                  fontSize: Math.min(props.fontSize, 14),
                  fontWeight: props.fontWeight,
                },
              ]}
            />
            {props.showClear ? (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={props.clearAccessibilityLabel ?? "Effacer le texte"}
                onPress={props.onClear}
                style={({ pressed }) => [
                  styles.clearBtn,
                  pressed ? styles.clearBtnPressed : null,
                ]}
                hitSlop={hitSlop}
              >
                <AppText variant="label" style={styles.clearText}>
                  ×
                </AppText>
              </Pressable>
            ) : null}
          </View>
        </View>

        <Modal
          visible={regionModalVisible}
          transparent
          animationType="fade"
          onRequestClose={closeModal}
        >
          <Pressable style={styles.modalBackdrop} onPress={closeModal}>
            <Pressable style={styles.modalCard} onPress={(e) => e.stopPropagation()}>
              <AppText variant="caption" style={styles.modalTitle}>
                Pays de recherche
              </AppText>
              {(["CH", "FR"] as const).map((code) => {
                const m = REGION_META[code];
                const selected = props.region === code;
                return (
                  <Pressable
                    key={code}
                    accessibilityRole="button"
                    accessibilityState={{ selected }}
                    accessibilityLabel={m.hint}
                    onPress={() => {
                      props.onRegionChange(code);
                      closeModal();
                    }}
                    style={({ pressed }) => [
                      styles.modalRow,
                      selected ? styles.modalRowSelected : null,
                      pressed ? styles.modalRowPressed : null,
                    ]}
                  >
                    <AppText variant="sectionTitle" style={styles.modalRowFlag}>
                      {m.flag}
                    </AppText>
                    <AppText variant="label" style={styles.modalRowLabel}>
                      {m.hint}
                    </AppText>
                    {selected ? (
                      <Ionicons name="checkmark" size={18} color={BRAND} style={styles.modalCheck} />
                    ) : null}
                  </Pressable>
                );
              })}
            </Pressable>
          </Pressable>
        </Modal>
      </>
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
    backgroundColor: "rgba(255,255,255,0.92)",
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
    borderWidth: 2,
    borderColor: "rgba(91,115,107,0.55)",
    backgroundColor: "rgba(255,255,255,0.96)",
  },
  barFocused: {
    borderColor: "#00796B",
    backgroundColor: "#FFFFFF",
    ...Platform.select({
      web: { boxShadow: "0 2px 6px rgba(0,121,107,0.08)" },
      default: {
        shadowColor: "#00796B",
        shadowOpacity: 0.08,
        shadowRadius: 6,
        shadowOffset: { width: 0, height: 2 },
        elevation: 2,
      },
    }),
  },
  regionBtn: {
    width: REGION_SEGMENT_WIDTH,
    minWidth: REGION_SEGMENT_WIDTH,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 4,
    alignSelf: "stretch",
    borderTopLeftRadius: BAR_RADIUS - 1,
    borderBottomLeftRadius: BAR_RADIUS - 1,
    borderRightWidth: StyleSheet.hairlineWidth,
    borderRightColor: "rgba(145,165,157,0.45)",
    backgroundColor: "rgba(243,247,245,0.98)",
    ...Platform.select({
      web: { cursor: "pointer" as const },
      default: {},
    }),
  },
  regionBtnPressed: {
    backgroundColor: "rgba(145,165,157,0.15)",
  },
  regionFlag: {
    fontSize: 15,
    lineHeight: 20,
    textAlign: "center",
  },
  inputWrap: {
    flex: 1,
    minWidth: 0,
    position: "relative",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.98)",
  },
  input: {
    flexGrow: 1,
    alignSelf: "stretch",
    width: "100%",
    color: UI_DARK,
    paddingLeft: 10,
    paddingRight: 10,
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
    lineHeight: 16,
    paddingVertical: 0,
    ...Platform.select({
      web: {
        paddingTop: 6,
        paddingBottom: 7,
        lineHeight: 17,
      },
      ios: {
        paddingTop: 7,
        paddingBottom: 7,
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
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(22,58,52,0.35)",
    justifyContent: "center",
    paddingHorizontal: 28,
  },
  modalCard: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#F3F7F5",
    paddingVertical: 10,
    paddingHorizontal: 8,
    maxWidth: 360,
    alignSelf: "center",
    width: "100%",
    ...Platform.select({
      web: { boxShadow: "0 8px 24px rgba(22,58,52,0.18)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.18,
        shadowRadius: 16,
        shadowOffset: { width: 0, height: 8 },
        elevation: 8,
      },
    }),
  },
  modalTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: UI_MUTED,
    paddingHorizontal: 10,
    paddingBottom: 8,
    letterSpacing: 0.2,
  },
  modalRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderRadius: 10,
    gap: 10,
  },
  modalRowSelected: {
    backgroundColor: "rgba(11,154,132,0.12)",
  },
  modalRowPressed: {
    backgroundColor: "rgba(145,165,157,0.14)",
  },
  modalRowFlag: {
    textAlign: "center",
  },
  modalRowLabel: {
    flex: 1,
    color: UI_DARK,
  },
  modalCheck: {
    marginLeft: "auto",
  },
});
