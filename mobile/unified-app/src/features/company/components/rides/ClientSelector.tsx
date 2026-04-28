import { useState } from "react";
import { Pressable, Text, TextInput, View } from "react-native";
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
      <Text style={{ fontWeight: "600" }}>Client</Text>
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
            borderColor: value === client.id ? "#0a7ea4" : "#ddd",
            borderRadius: 8,
            padding: 8,
          }}
        >
          <Text style={{ color: value === client.id ? "#0a7ea4" : "#333" }}>{client.label}</Text>
        </Pressable>
      ))}
      {clientsQuery.isLoading ? <Text style={{ color: "#666" }}>Recherche clients...</Text> : null}
    </View>
  );
}
