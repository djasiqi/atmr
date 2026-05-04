import { useState } from "react";
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { checkServiceArea, ServiceAreaCheckResponse } from "../../src/core/api/client";
import { ResponsiveContainer, Screen, useAppViewport } from "../../src/design/responsive";

export default function ServiceAreaScreen() {
  const { topInset } = useAppViewport();
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
    <Screen
      scroll
      backgroundColor="#EAF3F1"
      keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
      contentContainerStyle={styles.scroll}
    >
      <ResponsiveContainer>
        <View style={styles.card}>
          <Text style={styles.title}>Zone desservie</Text>
          <Text style={styles.lede}>Verifiez rapidement si Lirie couvre votre trajet avant l&apos;inscription.</Text>
          <TextInput
            value={departure}
            onChangeText={setDeparture}
            placeholder="Lieu de depart"
            placeholderTextColor="#91A59D"
            style={styles.input}
          />
          <TextInput
            value={destination}
            onChangeText={setDestination}
            placeholder="Lieu d'arrivee"
            placeholderTextColor="#91A59D"
            style={styles.input}
          />
          <TextInput
            value={date}
            onChangeText={setDate}
            placeholder="Date (YYYY-MM-DD)"
            placeholderTextColor="#91A59D"
            style={styles.input}
          />
          <TextInput
            value={transportType}
            onChangeText={setTransportType}
            placeholder="Type (assis, pmr, accompagnement...)"
            placeholderTextColor="#91A59D"
            style={styles.input}
          />
          <Pressable
            onPress={() => void submit()}
            disabled={pending}
            style={[styles.btn, pending && styles.btnDisabled]}
          >
            {pending ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.btnText}>Verifier</Text>
            )}
          </Pressable>
          {error ? <Text style={styles.err}>{error}</Text> : null}
          {result ? (
            <Text style={[styles.result, result.status === "unavailable" ? styles.resultBad : styles.resultOk]}>
              {result.status.toUpperCase()} - {result.message}
            </Text>
          ) : null}
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    paddingVertical: 24,
  },
  card: {
    gap: 12,
    borderRadius: 26,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
  },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: "#163A34",
  },
  lede: {
    color: "#45655D",
    lineHeight: 22,
    marginBottom: 4,
  },
  input: {
    borderWidth: 1,
    borderColor: "#91A59D",
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
    color: "#163A34",
    minHeight: 48,
  },
  btn: {
    borderRadius: 14,
    backgroundColor: "#0A8F7A",
    paddingVertical: 14,
    alignItems: "center",
    minHeight: 48,
    justifyContent: "center",
  },
  btnDisabled: {
    backgroundColor: "#84B7AE",
  },
  btnText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 16,
  },
  err: {
    color: "#b91c1c",
    fontWeight: "600",
  },
  result: {
    fontWeight: "600",
    lineHeight: 22,
  },
  resultOk: {
    color: "#0f5132",
  },
  resultBad: {
    color: "#b91c1c",
  },
});
