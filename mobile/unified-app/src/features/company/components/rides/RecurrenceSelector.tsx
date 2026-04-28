import { Pressable, Text, View } from "react-native";

type Recurrence = "none" | "daily" | "weekly";

type RecurrenceSelectorProps = {
  value: Recurrence;
  onChange: (value: Recurrence) => void;
};

export function RecurrenceSelector({ value, onChange }: RecurrenceSelectorProps) {
  const options: Recurrence[] = ["none", "daily", "weekly"];
  return (
    <View style={{ gap: 6 }}>
      <Text style={{ fontWeight: "600" }}>Recurrence</Text>
      <View style={{ flexDirection: "row", gap: 8 }}>
        {options.map((option) => (
          <Pressable
            key={option}
            onPress={() => onChange(option)}
            style={{
              borderWidth: 1,
              borderColor: value === option ? "#0a7ea4" : "#ddd",
              borderRadius: 8,
              paddingHorizontal: 10,
              paddingVertical: 6,
            }}
          >
            <Text style={{ color: value === option ? "#0a7ea4" : "#333" }}>{option}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}
