import { AppText } from "../../../design/ui/AppText";

type TypingIndicatorProps = {
  visible: boolean;
  label?: string;
};

export function TypingIndicator({ visible, label = "Saisie en cours..." }: TypingIndicatorProps) {
  if (!visible) return null;
  return (
    <AppText variant="caption" style={{ fontStyle: "italic" }}>
      {label}
    </AppText>
  );
}
