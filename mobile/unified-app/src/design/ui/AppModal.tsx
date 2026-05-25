import { useEffect, useRef, type ReactNode } from "react";
import { Animated, Modal, Pressable, StyleSheet, View, type ViewStyle } from "react-native";
import { Motion, MotionDistance, MotionEasing } from "../navigation/navigationMotion";
import { resolveMotionDuration } from "../navigation/applyNavigationMotion";
import { startMotionAnimation } from "../navigation/motionKpi";
import { useReduceMotion } from "../navigation/useReduceMotion";

export type AppModalVariant = "bottomSheet" | "dialog";

export type AppModalProps = {
  visible: boolean;
  onClose: () => void;
  variant?: AppModalVariant;
  /** Couleur du backdrop (alpha appliqué via animation). */
  backdropColor?: string;
  /** Opacité max du backdrop (0–1). */
  backdropOpacity?: number;
  /** Style du conteneur de contenu (laisser undefined pour fullscreen sheet). */
  containerStyle?: ViewStyle;
  /** Désactive le tap-to-close sur le backdrop. */
  dismissOnBackdropPress?: boolean;
  children: ReactNode;
  screen?: string;
};

const BACKDROP_DEFAULT_COLOR = "rgba(0,0,0,1)";
const BACKDROP_DEFAULT_OPACITY = 0.4;

/**
 * Wrapper modal unifié LIRIE :
 *   - `bottomSheet` : translateY 32 → 0 + backdrop fade (240 ms ouverture / 180 ms fermeture).
 *   - `dialog`      : fade + scale 0.98 → 1 (240 / 180).
 * Pas d'`animationType` natif (RN slide non coordonné).
 */
export function AppModal({
  visible,
  onClose,
  variant = "bottomSheet",
  backdropColor = BACKDROP_DEFAULT_COLOR,
  backdropOpacity = BACKDROP_DEFAULT_OPACITY,
  containerStyle,
  dismissOnBackdropPress = true,
  children,
  screen,
}: AppModalProps) {
  const reduceMotion = useReduceMotion();
  const progress = useRef(new Animated.Value(visible ? 1 : 0)).current;
  const mounted = useRef(visible);
  const lastTargetRef = useRef<number>(visible ? 1 : 0);
  const animationRef = useRef<Animated.CompositeAnimation | null>(null);
  const endKpiRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (visible) mounted.current = true;
    const toValue = visible ? 1 : 0;

    if (lastTargetRef.current === toValue) {
      return;
    }
    lastTargetRef.current = toValue;

    animationRef.current?.stop();
    endKpiRef.current = null;

    const duration = resolveMotionDuration(
      visible ? Motion.modal : Motion.modalClose,
      reduceMotion
    );
    const end = startMotionAnimation({
      layer: "modal",
      kind: variant === "bottomSheet" ? "bottom_sheet" : "dialog",
      duration_expected_ms: visible ? Motion.modal : Motion.modalClose,
      screen,
      source: `motion.modal.${variant}`,
    });
    endKpiRef.current = end;

    const animation = Animated.timing(progress, {
      toValue,
      duration,
      easing: MotionEasing,
      useNativeDriver: true,
    });
    animationRef.current = animation;
    animation.start(({ finished }) => {
      if (finished && endKpiRef.current === end) {
        end();
        endKpiRef.current = null;
      }
    });

    return () => {
      if (endKpiRef.current === end) {
        endKpiRef.current = null;
      }
    };
  }, [visible, progress, reduceMotion, variant, screen]);

  const backdropStyle = {
    opacity: progress.interpolate({
      inputRange: [0, 1],
      outputRange: [0, backdropOpacity],
    }),
    backgroundColor: backdropColor,
  };

  const sheetStyle =
    variant === "bottomSheet"
      ? {
          opacity: progress,
          transform: [
            {
              translateY: progress.interpolate({
                inputRange: [0, 1],
                outputRange: [MotionDistance.modalSlideYPx, 0],
              }),
            },
          ],
        }
      : {
          opacity: progress,
          transform: [
            {
              scale: progress.interpolate({
                inputRange: [0, 1],
                outputRange: [MotionDistance.scaleInactive, 1],
              }),
            },
          ],
        };

  return (
    <Modal transparent visible={visible} onRequestClose={onClose} animationType="none">
      <View style={styles.root} pointerEvents={visible ? "auto" : "none"}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Fermer"
          onPress={dismissOnBackdropPress ? onClose : undefined}
          style={styles.backdropPress}
        >
          <Animated.View style={[styles.backdrop, backdropStyle]} />
        </Pressable>
        <Animated.View
          style={[
            variant === "bottomSheet" ? styles.bottomSheet : styles.dialog,
            sheetStyle,
            containerStyle,
          ]}
          pointerEvents="box-none"
        >
          {children}
        </Animated.View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  backdropPress: {
    ...StyleSheet.absoluteFillObject,
  },
  backdrop: {
    flex: 1,
  },
  bottomSheet: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
  },
  dialog: {
    position: "absolute",
    left: 24,
    right: 24,
    top: "30%",
  },
});
