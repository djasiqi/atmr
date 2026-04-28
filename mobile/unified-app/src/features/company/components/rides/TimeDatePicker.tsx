import { Text, TextInput, View } from "react-native";

const SWISS_TZ = "Europe/Zurich";

function formatSwissPreviewLocal(iso: string): string {
  const t = iso?.trim() ?? "";
  if (!t) return "";
  const d = new Date(t);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "long",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

type TimeDatePickerProps = {
  value: string;
  onChange: (value: string) => void;
};

export function TimeDatePicker({ value, onChange }: TimeDatePickerProps) {
  const preview = formatSwissPreviewLocal(value);
  return (
    <View style={{ gap: 6 }}>
      <Text style={{ fontWeight: "600" }}>Date et heure</Text>
      <Text style={{ color: "#5F6368", fontSize: 12 }}>
        Fuseau affichage : Suisse (Europe/Zurich). Saisir une date-heure au format ISO (même règle que
        côté API) — ex. 2026-04-23T19:12:00
      </Text>
      {preview ? (
        <Text style={{ color: "#163A34", fontSize: 13, fontWeight: "500" }}>
          Aperçu (CH) : {preview}
        </Text>
      ) : null}
      <TextInput
        value={value}
        onChangeText={onChange}
        placeholder="2026-04-23T19:12:00"
        autoCapitalize="none"
        accessibilityLabel="Date et heure (ISO) pour prise en charge, conforme Suisse côté affichage"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10 }}
      />
    </View>
  );
}
