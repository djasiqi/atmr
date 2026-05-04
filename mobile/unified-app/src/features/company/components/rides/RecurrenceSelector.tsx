import { Pressable, View } from "react-native";
import { brandPrimary, brandText } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";

type Recurrence = "none" | "daily" | "weekly";

type RecurrenceSelectorProps = {
  value: Recurrence;
  onChange: (value: Recurrence) => void;
};

export function RecurrenceSelector({ value, onChange }: RecurrenceSelectorProps) {
  const options: Recurrence[] = ["none", "daily", "weekly"];
  return (
    <View style={{ gap: 6 }}>
      <AppText variant="label">Recurrence</AppText>
      <View style={{ flexDirection: "row", gap: 8 }}>
        {options.map((option) => (
          <Pressable
            key={option}
            onPress={() => onChange(option)}
            style={{
              borderWidth: 1,
              borderColor: value === option ? "#0A8F7A" : "#ddd",
              borderRadius: 8,
              paddingHorizontal: 10,
              paddingVertical: 6,
            }}
          >
            <AppText variant="body" style={{ color: value === option ? brandPrimary : brandText }}>
              {option}
            </AppText>
          </Pressable>
        ))}
      </View>
    </View>
  );
}
