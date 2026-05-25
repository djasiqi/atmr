import { Pressable } from "react-native";
import { AppText } from "../../../design/ui/AppText";

type AssignFabProps = {
  onPress: () => void;
  label?: string;
};

export function AssignFab({ onPress, label = "Assigner" }: AssignFabProps) {
  return (
    <Pressable
      onPress={onPress}
      style={{
        position: "absolute",
        right: 20,
        bottom: 20,
        backgroundColor: "#00796B",
        borderRadius: 24,
        paddingHorizontal: 16,
        paddingVertical: 12,
        elevation: 3,
      }}
    >
      {/* DS_EXCEPTION: libellé sur pastille CTA flottante verte (contraste blanc) */}
      <AppText variant="label" style={{ color: "#fff" }}>
        {label}
      </AppText>
    </Pressable>
  );
}
