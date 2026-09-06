import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  BackHandler,
  Keyboard,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
  type LayoutChangeEvent,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  computeCreateRideResultsMaxHeight,
  computeCreateRideSheetLayout,
  useCreateRideKeyboardFrame,
} from "./createRideSheetLayout";

const CHROME_ESTIMATE = 168;
const FOOTER_ESTIMATE = 52;

type CreateRideKeyboardSheetProps = {
  visible: boolean;
  title: string;
  subtitle?: string;
  onClose: () => void;
  search: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
};

/** Feuille CREATE RIDE : hauteur au contenu, plafond au-dessus du clavier. */
export function CreateRideKeyboardSheet({
  visible,
  title,
  subtitle,
  onClose,
  search,
  footer,
  children,
}: CreateRideKeyboardSheetProps) {
  const insets = useSafeAreaInsets();
  const keyboardFrame = useCreateRideKeyboardFrame();
  const sheetLayout = computeCreateRideSheetLayout(keyboardFrame);
  const safeBottom = sheetLayout.liftBottom > 0 ? 0 : insets.bottom;
  const [chromeHeight, setChromeHeight] = useState(CHROME_ESTIMATE);
  const [footerHeight, setFooterHeight] = useState(footer ? FOOTER_ESTIMATE : 0);
  const sheetVerticalPad = 24 + safeBottom;
  const resultsMaxHeight = computeCreateRideResultsMaxHeight(
    sheetLayout.maxSheetHeight - sheetVerticalPad,
    chromeHeight,
    footer ? footerHeight : 0
  );

  const onChromeLayout = useCallback((event: LayoutChangeEvent) => {
    const next = Math.round(event.nativeEvent.layout.height);
    setChromeHeight((prev) => (Math.abs(prev - next) < 2 ? prev : next));
  }, []);

  const onFooterLayout = useCallback((event: LayoutChangeEvent) => {
    const next = Math.round(event.nativeEvent.layout.height);
    setFooterHeight((prev) => (Math.abs(prev - next) < 2 ? prev : next));
  }, []);

  const dismissThenClose = useCallback(() => {
    Keyboard.dismiss();
    onClose();
  }, [onClose]);

  const onHardwareBack = useCallback(() => {
    if (keyboardFrame.keyboardVisible) {
      Keyboard.dismiss();
      return;
    }
    onClose();
  }, [keyboardFrame.keyboardVisible, onClose]);

  useEffect(() => {
    if (!visible) return undefined;
    const sub = BackHandler.addEventListener("hardwareBackPress", () => {
      onHardwareBack();
      return true;
    });
    return () => sub.remove();
  }, [onHardwareBack, visible]);

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onHardwareBack}>
      <View style={[s.backdrop, { paddingBottom: sheetLayout.liftBottom }]}>
        <Pressable style={s.backdropTap} onPress={dismissThenClose} accessibilityLabel="Fermer" />
        <View
          style={[
            s.sheet,
            {
              maxHeight: sheetLayout.maxSheetHeight,
              paddingBottom: 10 + safeBottom,
            },
          ]}
        >
          <View style={s.fixedChrome} onLayout={onChromeLayout}>
            <View style={s.headerRow}>
              <View style={s.headerText}>
                <AppText variant="sectionTitle" style={s.title}>
                  {title}
                </AppText>
                {subtitle ? (
                  <AppText variant="caption" style={s.subtitle}>
                    {subtitle}
                  </AppText>
                ) : null}
              </View>
              <Pressable
                onPress={dismissThenClose}
                accessibilityRole="button"
                accessibilityLabel="Fermer"
                hitSlop={8}
                style={s.closeHit}
              >
                <Ionicons name="close" size={22} color={E.TEXT_SEC} />
              </Pressable>
            </View>
            {search}
          </View>
          <ScrollView
            style={{ maxHeight: resultsMaxHeight }}
            contentContainerStyle={s.resultsContent}
            keyboardShouldPersistTaps="always"
            keyboardDismissMode="none"
            showsVerticalScrollIndicator
          >
            {children}
          </ScrollView>
          {footer ? <View onLayout={onFooterLayout}>{footer}</View> : null}
        </View>
      </View>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, justifyContent: "flex-end" },
  backdropTap: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(15, 23, 42, 0.4)" },
  sheet: {
    backgroundColor: E.CARD,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    paddingTop: 14,
    paddingHorizontal: 16,
    width: "100%",
  },
  fixedChrome: { gap: 10, marginBottom: 8 },
  headerRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
  },
  headerText: { flex: 1, minWidth: 0, gap: 4 },
  title: { color: E.TEXT, fontWeight: "600" },
  subtitle: { color: E.TEXT_SEC, fontWeight: "600" },
  closeHit: {
    width: 40,
    height: 40,
    alignItems: "center",
    justifyContent: "center",
  },
  resultsContent: { paddingBottom: 4 },
});
