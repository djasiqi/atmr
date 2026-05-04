import { useState } from "react";
import { TextInput, View } from "react-native";
import { AppButton } from "../../../design/responsive";

type NotesEditorProps = {
  initialValue?: string | null;
  onSave: (value: string) => Promise<void> | void;
  saveLabel?: string;
};

export function NotesEditor({ initialValue, onSave, saveLabel = "Enregistrer notes" }: NotesEditorProps) {
  const [value, setValue] = useState(initialValue ?? "");
  const [saving, setSaving] = useState(false);

  return (
    <View style={{ gap: 8 }}>
      <TextInput
        value={value}
        onChangeText={setValue}
        multiline
        placeholder="Notes operateur"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10, minHeight: 80 }}
      />
      <AppButton
        title={saving ? "Enregistrement..." : saveLabel}
        variant="secondary"
        onPress={async () => {
          setSaving(true);
          try {
            await onSave(value.trim());
          } finally {
            setSaving(false);
          }
        }}
        disabled={saving}
      />
    </View>
  );
}
