import { useMemo, useRef, useState, type ReactNode } from "react";
import {
  Animated,
  Easing,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  View,
  type LayoutRectangle,
  type ViewStyle,
} from "react-native";
import { useAppViewport } from "../design/responsive/useAppViewport";
import { AppText } from "../design/ui/AppText";

export type RadialAction = {
  key: string;
  label: string;
  icon: ReactNode;
  onPress: () => void;
  color?: string;
  disabled?: boolean;
};

export type RadialActionMenuProps = {
  actions: RadialAction[];
  mainIcon?: ReactNode;
  openIcon?: ReactNode;
  triggerVariant?: "fab" | "tab";
  actionsLayout?: "arc" | "vertical";
  position?: "bottomRight" | "bottomLeft";
  inline?: boolean;
  radius?: number;
  verticalSpacing?: number;
  verticalExtraSpacing?: number;
  actionsOffsetX?: number;
  actionsOffsetY?: number;
  bottomOffset?: number;
  sideOffset?: number;
  showLabels?: boolean;
  accessibilityLabel?: string;
};

const DEFAULT_MAIN_COLOR = "#00796B";
const SECONDARY_COLORS = ["#2563EB", "#7C3AED", "#0F766E"] as const;
const MENU_SIZE = 44;
const ACTION_SIZE = 46;
const ACTIONS_LIMIT = 3;
const EDGE_MARGIN = 12;
const LABEL_MAX_WIDTH = 112;
const LABEL_GAP = 8;

function getActionAngles(
  position: "bottomRight" | "bottomLeft",
  count: number,
  inline: boolean
): number[] {
  if (count <= 1) return [position === "bottomRight" ? 245 : 295];
  const start = inline ? (position === "bottomRight" ? 200 : 340) : position === "bottomRight" ? 210 : 330;
  const end = inline ? (position === "bottomRight" ? 260 : 280) : 270;
  const step = (end - start) / (count - 1);
  return Array.from({ length: count }, (_, idx) => start + step * idx);
}

export function RadialActionMenu({
  actions,
  mainIcon,
  openIcon,
  triggerVariant = "fab",
  actionsLayout = "arc",
  position = "bottomRight",
  inline = false,
  radius = 70,
  verticalSpacing = 34,
  verticalExtraSpacing = 0,
  actionsOffsetX = 0,
  actionsOffsetY = 0,
  bottomOffset = 84,
  sideOffset = 18,
  showLabels = true,
  accessibilityLabel = "Actions rapides",
}: RadialActionMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [anchorInWindow, setAnchorInWindow] = useState<LayoutRectangle | null>(null);
  const { width: windowWidth, height: windowHeight } = useAppViewport();
  const progress = useRef(new Animated.Value(0)).current;
  const anchorRef = useRef<View | null>(null);
  const safeActions = useMemo(() => actions.slice(0, ACTIONS_LIMIT), [actions]);
  const angles = useMemo(
    () => getActionAngles(position, safeActions.length, inline),
    [inline, position, safeActions.length]
  );

  const shellPositionStyle: ViewStyle =
    position === "bottomRight"
      ? { right: sideOffset, bottom: bottomOffset }
      : { left: sideOffset, bottom: bottomOffset };

  const openMenu = () => {
    const startAnimation = () => {
      setIsOpen(true);
      Animated.timing(progress, {
        toValue: 1,
        duration: 220,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }).start();
    };

    if (inline && anchorRef.current) {
      anchorRef.current.measureInWindow((x, y, width, height) => {
        setAnchorInWindow({ x, y, width, height });
        startAnimation();
      });
      return;
    }
    startAnimation();
  };

  const closeMenu = () => {
    Animated.timing(progress, {
      toValue: 0,
      duration: 180,
      easing: Easing.in(Easing.cubic),
      useNativeDriver: true,
    }).start(({ finished }) => {
      if (finished) setIsOpen(false);
    });
  };

  const toggleMenu = () => {
    if (isOpen) closeMenu();
    else openMenu();
  };

  const handleActionPress = (action: RadialAction) => {
    if (action.disabled) return;
    closeMenu();
    action.onPress();
  };

  const menuButton = (
    <View
      ref={inline ? anchorRef : undefined}
      onLayout={
        inline
          ? (event) => {
              if (anchorInWindow) return;
              const { width, height } = event.nativeEvent.layout;
              setAnchorInWindow({ x: 0, y: 0, width, height });
            }
          : undefined
      }
    >
      <Pressable
        onPress={toggleMenu}
        accessibilityRole={triggerVariant === "tab" ? "tab" : "button"}
        accessibilityState={triggerVariant === "tab" ? { selected: isOpen } : undefined}
        accessibilityLabel={isOpen ? `Fermer ${accessibilityLabel}` : `Ouvrir ${accessibilityLabel}`}
        style={({ pressed }) => [
          triggerVariant === "tab" ? styles.tabTrigger : styles.mainButton,
          isOpen && triggerVariant === "tab" && styles.tabTriggerOpen,
          pressed && (triggerVariant === "tab" ? styles.tabTriggerPressed : styles.mainButtonPressed),
        ]}
      >
        {isOpen ? (openIcon ?? mainIcon) : (mainIcon ?? openIcon)}
      </Pressable>
    </View>
  );

  if (!isOpen) {
    if (inline) return menuButton;
    return (
      <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
        <View pointerEvents="box-none" style={[styles.menuShell, shellPositionStyle]}>
          {menuButton}
        </View>
      </View>
    );
  }

  const anchoredShellStyle: ViewStyle | null =
    inline && anchorInWindow
      ? {
          left: anchorInWindow.x + anchorInWindow.width / 2 - MENU_SIZE / 2,
          top: anchorInWindow.y + anchorInWindow.height / 2 - MENU_SIZE / 2,
          bottom: undefined,
          right: undefined,
        }
      : null;
  const anchorCenter = inline && anchorInWindow
    ? {
        x: anchorInWindow.x + anchorInWindow.width / 2,
        y: anchorInWindow.y + anchorInWindow.height / 2,
      }
    : null;

  return (
    <Modal transparent visible onRequestClose={closeMenu} animationType="none">
      <View style={styles.overlay} pointerEvents="box-none">
        <Pressable
          style={StyleSheet.absoluteFill}
          onPress={closeMenu}
          accessibilityRole="button"
          accessibilityLabel="Fermer le menu d'actions"
        />
        <View pointerEvents="box-none" style={[styles.menuShell, anchoredShellStyle ?? shellPositionStyle]}>
          {safeActions.map((action, idx) => {
            const angle = (angles[idx] * Math.PI) / 180;
            const unitX = Math.cos(angle);
            const unitY = Math.sin(angle);
            let targetX = unitX * radius;
            let targetY = unitY * radius;
            if (actionsLayout === "vertical") {
              // En mode vertical, l'alignement se fait sur l'axe X du bouton principal.
              targetX = actionsOffsetX;
              targetY =
                -(idx + 1) * verticalSpacing - idx * verticalExtraSpacing + actionsOffsetY;
            }
            if (anchorCenter) {
              const leftLimit =
                EDGE_MARGIN +
                ACTION_SIZE / 2 +
                (showLabels && position === "bottomRight" ? LABEL_MAX_WIDTH + LABEL_GAP : 0);
              const rightLimit =
                windowWidth -
                EDGE_MARGIN -
                ACTION_SIZE / 2 -
                (showLabels && position === "bottomLeft" ? LABEL_MAX_WIDTH + LABEL_GAP : 0);
              const topLimit = EDGE_MARGIN + ACTION_SIZE / 2;
              const bottomLimit = windowHeight - EDGE_MARGIN - ACTION_SIZE / 2;

              const unclampedX = anchorCenter.x + targetX;
              const unclampedY = anchorCenter.y + targetY;
              const clampedX = Math.min(Math.max(unclampedX, leftLimit), rightLimit);
              const clampedY = Math.min(Math.max(unclampedY, topLimit), bottomLimit);
              targetX = clampedX - anchorCenter.x;
              targetY = clampedY - anchorCenter.y;
            }
            const color = action.color ?? SECONDARY_COLORS[idx % SECONDARY_COLORS.length];
            return (
              <Animated.View
                key={action.key}
                style={[
                  styles.actionContainer,
                  {
                    opacity: progress,
                    transform: [
                      { translateX: Animated.multiply(progress, targetX) },
                      { translateY: Animated.multiply(progress, targetY) },
                      { scale: progress.interpolate({ inputRange: [0, 1], outputRange: [0.7, 1] }) },
                    ],
                  },
                ]}
              >
                {showLabels ? (
                  <View
                    style={[
                      styles.labelWrap,
                      position === "bottomLeft" ? styles.labelWrapFloatingLeft : styles.labelWrapFloatingRight,
                    ]}
                  >
                    <AppText variant="caption" numberOfLines={1} style={styles.labelText}>
                      {action.label}
                    </AppText>
                  </View>
                ) : null}
                <Pressable
                  onPress={() => handleActionPress(action)}
                  disabled={action.disabled}
                  accessibilityRole="button"
                  accessibilityLabel={action.label}
                  style={({ pressed }) => [
                    styles.actionButton,
                    { backgroundColor: color },
                    action.disabled && styles.actionDisabled,
                    pressed && !action.disabled && styles.actionPressed,
                  ]}
                >
                  {action.icon}
                </Pressable>
              </Animated.View>
            );
          })}
          {menuButton}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(15, 23, 42, 0.12)",
  },
  menuShell: {
    position: "absolute",
    width: MENU_SIZE,
    height: MENU_SIZE,
    zIndex: 9999,
  },
  mainButton: {
    width: MENU_SIZE,
    height: MENU_SIZE,
    borderRadius: MENU_SIZE / 2,
    backgroundColor: DEFAULT_MAIN_COLOR,
    alignItems: "center",
    justifyContent: "center",
    ...Platform.select({
      web: {
        boxShadow: "0 8px 18px rgba(10, 58, 52, 0.22)",
      } as const,
      default: {
        elevation: 5,
        shadowColor: "#163A34",
        shadowOpacity: 0.24,
        shadowOffset: { width: 0, height: 4 },
        shadowRadius: 10,
      },
    }),
  },
  mainButtonPressed: {
    opacity: 0.9,
  },
  tabTrigger: {
    width: 44,
    height: 44,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "transparent",
  },
  tabTriggerOpen: {
    backgroundColor: "rgba(0, 121, 107, 0.1)",
  },
  tabTriggerPressed: {
    opacity: Platform.OS === "ios" ? 0.88 : 1,
  },
  actionContainer: {
    position: "absolute",
    left: (MENU_SIZE - ACTION_SIZE) / 2,
    top: (MENU_SIZE - ACTION_SIZE) / 2,
    width: ACTION_SIZE,
    height: ACTION_SIZE,
    alignItems: "center",
    justifyContent: "center",
  },
  actionButton: {
    width: ACTION_SIZE,
    height: ACTION_SIZE,
    borderRadius: ACTION_SIZE / 2,
    alignItems: "center",
    justifyContent: "center",
    ...Platform.select({
      web: {
        boxShadow: "0 6px 14px rgba(15, 23, 42, 0.22)",
      } as const,
      default: {
        elevation: 4,
        shadowColor: "#0F172A",
        shadowOpacity: 0.2,
        shadowOffset: { width: 0, height: 3 },
        shadowRadius: 8,
      },
    }),
  },
  actionPressed: {
    opacity: 0.9,
  },
  actionDisabled: {
    opacity: 0.45,
  },
  labelWrap: {
    position: "absolute",
    maxWidth: 112,
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(228, 231, 236, 0.95)",
    ...Platform.select({
      web: {
        boxShadow: "0 3px 8px rgba(15, 23, 42, 0.12)",
      } as const,
      default: {
        elevation: 2,
        shadowColor: "#0F172A",
        shadowOpacity: 0.1,
        shadowOffset: { width: 0, height: 1 },
        shadowRadius: 4,
      },
    }),
  },
  labelWrapFloatingRight: {
    right: ACTION_SIZE + LABEL_GAP,
  },
  labelWrapFloatingLeft: {
    left: ACTION_SIZE + LABEL_GAP,
  },
  labelText: {
    color: "#2D3748",
    fontWeight: "600",
    lineHeight: 16,
  },
});
