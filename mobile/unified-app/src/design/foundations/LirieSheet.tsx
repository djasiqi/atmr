import type { ReactNode } from "react";
import { useEffect } from "react";
import { Modal, Pressable, ScrollView, StyleSheet, View } from "react-native";
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from "react-native-reanimated";
import { useReduceMotion } from "../navigation/useReduceMotion";
import { LirieButton } from "./LirieButton";
import { LirieText } from "./LirieText";
import { lirieColor, lirieMotion, lirieRadius, lirieSpace } from "./tokens";

type Props = {
  visible: boolean;
  title: string;
  children: ReactNode;
  confirmLabel?: string;
  onClose: () => void;
  onConfirm?: () => void;
};

const SHEET_EASING = Easing.bezier(0.23, 1, 0.32, 1);

export function LirieSheet({
  visible,
  title,
  children,
  confirmLabel,
  onClose,
  onConfirm,
}: Props) {
  const reduceMotion = useReduceMotion();
  const opacity = useSharedValue(0);

  useEffect(() => {
    if (!visible) {
      opacity.value = 0;
      return;
    }
    if (reduceMotion) {
      opacity.value = 1;
      return;
    }
    opacity.value = 0;
    opacity.value = withTiming(1, { duration: lirieMotion.sheetMs, easing: SHEET_EASING });
  }, [opacity, reduceMotion, visible]);

  const backdropStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

  return (
    <Modal animationType="none" transparent visible={visible} onRequestClose={onClose}>
      <Animated.View style={[styles.backdrop, backdropStyle]}>
        <Pressable accessibilityLabel="Fermer la feuille" style={styles.dismiss} onPress={onClose} />
        <View accessibilityViewIsModal style={styles.sheet}>
          <View style={styles.header}>
            <View style={styles.title}>
              <LirieText variant="title">{title}</LirieText>
            </View>
            <LirieButton label="Fermer" variant="quiet" onPress={onClose} />
          </View>
          <ScrollView contentContainerStyle={styles.body}>{children}</ScrollView>
          {confirmLabel ? <LirieButton label={confirmLabel} onPress={onConfirm} /> : null}
        </View>
      </Animated.View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "rgba(38, 53, 50, 0.4)",
  },
  dismiss: { flex: 1 },
  sheet: {
    maxHeight: "80%",
    backgroundColor: lirieColor.white,
    borderTopLeftRadius: lirieRadius.card,
    borderTopRightRadius: lirieRadius.card,
    padding: lirieSpace.lg,
    gap: lirieSpace.md,
  },
  header: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: lirieSpace.sm,
  },
  title: { flexGrow: 1, flexShrink: 1, minWidth: 160 },
  body: { gap: lirieSpace.sm },
});
