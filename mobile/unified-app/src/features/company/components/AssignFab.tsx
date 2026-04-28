import { Pressable, Text } from "react-native";

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
        backgroundColor: "#0a7ea4",
        borderRadius: 24,
        paddingHorizontal: 16,
        paddingVertical: 12,
        elevation: 3,
      }}
    >
      <Text style={{ color: "#fff", fontWeight: "700" }}>{label}</Text>
    </Pressable>
  );
}
