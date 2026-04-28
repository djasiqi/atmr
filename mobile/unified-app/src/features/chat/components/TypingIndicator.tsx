import { Text } from "react-native";

type TypingIndicatorProps = {
  visible: boolean;
  label?: string;
};

export function TypingIndicator({ visible, label = "Saisie en cours..." }: TypingIndicatorProps) {
  if (!visible) return null;
  return <Text style={{ color: "#666", fontStyle: "italic" }}>{label}</Text>;
}
