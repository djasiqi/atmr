import { useState } from "react";
import { Pressable, ScrollView, Text, TextInput } from "react-native";
import { checkServiceArea, ServiceAreaCheckResponse } from "../../src/core/api/client";

export default function ServiceAreaScreen() {
  const [departure, setDeparture] = useState("");
  const [destination, setDestination] = useState("");
  const [date, setDate] = useState("");
  const [transportType, setTransportType] = useState("assis");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ServiceAreaCheckResponse | null>(null);

  const submit = async () => {
    if (!departure.trim() || !destination.trim() || !date.trim()) {
      setError("Depart, destination et date sont requis.");
      return;
    }
    setPending(true);
    setError(null);
    try {
      const response = await checkServiceArea({
        departure: departure.trim(),
        destination: destination.trim(),
        date: date.trim(),
        transport_type: transportType.trim() || "assis",
      });
      setResult(response);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Impossible de verifier la zone.");
      setResult(null);
    } finally {
      setPending(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={{ padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 24, fontWeight: "800", color: "#0f172a" }}>
        Zone desservie
      </Text>
      <Text style={{ color: "#475569" }}>
        Verifiez rapidement si Lirie couvre votre trajet avant l&apos;inscription.
      </Text>
      <TextInput
        value={departure}
        onChangeText={setDeparture}
        placeholder="Lieu de depart"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <TextInput
        value={destination}
        onChangeText={setDestination}
        placeholder="Lieu d'arrivee"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <TextInput
        value={date}
        onChangeText={setDate}
        placeholder="Date (YYYY-MM-DD)"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <TextInput
        value={transportType}
        onChangeText={setTransportType}
        placeholder="Type (assis, pmr, accompagnement...)"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
      />
      <Pressable
        onPress={() => void submit()}
        disabled={pending}
        style={{ borderRadius: 10, backgroundColor: pending ? "#9cb7c1" : "#0a7ea4", padding: 14, alignItems: "center" }}
      >
        <Text style={{ color: "#fff", fontWeight: "700" }}>
          {pending ? "Verification..." : "Verifier"}
        </Text>
      </Pressable>
      {error ? <Text style={{ color: "#b91c1c" }}>{error}</Text> : null}
      {result ? (
        <Text style={{ color: result.status === "unavailable" ? "#b91c1c" : "#0f5132" }}>
          {result.status.toUpperCase()} - {result.message}
        </Text>
      ) : null}
    </ScrollView>
  );
}
