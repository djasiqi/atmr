import { useState } from "react";
import { Pressable, TextInput, View } from "react-native";
import { brandPrimary, brandText } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { useCompanyClientSearch } from "../../useRideForms";

type ClientSelectorProps = {
  value: number | null;
  onChange: (clientId: number | null) => void;
};

export function ClientSelector({ value, onChange }: ClientSelectorProps) {
  const [query, setQuery] = useState("");
  const clientsQuery = useCompanyClientSearch(query);
  return (
    <View style={{ gap: 6 }}>
      <AppText variant="label">Client</AppText>
      <TextInput
        value={query}
        onChangeText={setQuery}
        placeholder="Rechercher client"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10 }}
      />
      {clientsQuery.data?.map((client) => (
        <Pressable
          key={client.id}
          onPress={() => onChange(client.id)}
          style={{
            borderWidth: 1,
            borderColor: value === client.id ? "#0A8F7A" : "#ddd",
            borderRadius: 8,
            padding: 8,
          }}
        >
          <AppText variant="body" style={{ color: value === client.id ? brandPrimary : brandText }}>
            {client.label}
          </AppText>
        </Pressable>
      ))}
      {clientsQuery.isLoading ? <AppText variant="bodyMuted">Recherche clients...</AppText> : null}
    </View>
  );
}
