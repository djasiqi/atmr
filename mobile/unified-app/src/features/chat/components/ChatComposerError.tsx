import { Linking, Pressable } from "react-native";
import { AppText } from "../../../design/ui/AppText";

type ChatComposerErrorProps = {
  message: string | null;
  openSettings?: boolean;
};

export function ChatComposerError({ message, openSettings = false }: ChatComposerErrorProps) {
  if (!message) return null;

  if (openSettings) {
    return (
      <Pressable
        onPress={() => void Linking.openSettings()}
        accessibilityRole="button"
        accessibilityLabel="Rouvrir les réglages du téléphone pour autoriser le micro"
      >
        <AppText variant="error">{message}</AppText>
      </Pressable>
    );
  }

  return <AppText variant="error">{message}</AppText>;
}
