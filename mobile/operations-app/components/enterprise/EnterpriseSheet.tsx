import React from "react";
import {
  Modal,
  View,
  StyleSheet,
  TouchableWithoutFeedback,
  TouchableOpacity,
  Text,
  ScrollView,
  ViewStyle,
} from "react-native";

type SheetAction = {
  label: string;
  tone?: "primary" | "secondary" | "danger";
  onPress: () => void;
};

type EnterpriseSheetProps = {
  visible: boolean;
  onClose: () => void;
  title?: string;
  subtitle?: string;
  actions?: SheetAction[];
  children: React.ReactNode;
  snapHeight?: number;
  style?: ViewStyle;
};

export const EnterpriseSheet: React.FC<EnterpriseSheetProps> = ({
  visible,
  onClose,
  title,
  subtitle,
  actions = [],
  children,
  snapHeight,
  style,
}) => {
  return (
    <Modal animationType="slide" transparent visible={visible} onRequestClose={onClose}>
      <TouchableWithoutFeedback onPress={onClose}>
        <View style={styles.backdrop} />
      </TouchableWithoutFeedback>

      <View style={[styles.sheet, snapHeight && { maxHeight: snapHeight }, style]}>
        <View style={styles.grabber} />
        {title ? (
          <View style={styles.header}>
            <Text style={styles.title}>{title}</Text>
            {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
          </View>
        ) : null}

        <ScrollView style={{ flex: 1 }} contentContainerStyle={{ paddingBottom: 20 }}>
          {children}
        </ScrollView>

        {actions.length ? (
          <View style={styles.actions}>
            {actions.map((action) => (
              <TouchableOpacity
                key={action.label}
                style={[styles.actionButton, buttonToneStyle(action.tone)]}
                onPress={action.onPress}
              >
                <Text style={[styles.actionLabel, labelToneStyle(action.tone)]}>
                  {action.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        ) : null}
      </View>
    </Modal>
  );
};

const buttonToneStyle = (tone: SheetAction["tone"]) => {
  switch (tone) {
    case "primary":
      return { backgroundColor: "#60A5FA" };
    case "danger":
      return { backgroundColor: "#F87171" };
    default:
      return { backgroundColor: "rgba(100,116,228,0.18)" };
  }
};

const labelToneStyle = (tone: SheetAction["tone"]) => {
  switch (tone) {
    case "primary":
      return { color: "#0B1736" };
    case "danger":
      return { color: "#0B1736" };
    default:
      return { color: "#E5EDFF" };
  }
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(5,12,28,0.55)",
  },
  sheet: {
    backgroundColor: "#050B1C",
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 24,
    maxHeight: "80%",
  },
  grabber: {
    width: 52,
    height: 5,
    borderRadius: 999,
    backgroundColor: "rgba(120,134,203,0.35)",
    alignSelf: "center",
    marginBottom: 12,
  },
  header: {
    marginBottom: 12,
    gap: 4,
  },
  title: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 18,
  },
  subtitle: {
    color: "rgba(182,196,245,0.8)",
    fontSize: 13,
  },
  actions: {
    marginTop: 12,
    flexDirection: "row",
    gap: 12,
  },
  actionButton: {
    flex: 1,
    borderRadius: 16,
    paddingVertical: 12,
    alignItems: "center",
  },
  actionLabel: {
    fontWeight: "700",
    fontSize: 15,
  },
});

export default EnterpriseSheet;

