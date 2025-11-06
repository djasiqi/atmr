import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import dayjs from "dayjs";
import "dayjs/locale/fr";

import { useAuth } from "@/hooks/useAuth";
import {
  getDispatchSettings,
  resetAssignments,
  runDispatch,
  runOptimizer,
  updateDispatchSettings,
} from "@/services/enterpriseDispatch";
import { DispatchSettings } from "@/types/enterpriseDispatch";

dayjs.locale("fr");

const numberOr = (value: string, fallback: number): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export default function EnterpriseSettingsScreen() {
  const { refreshEnterprise } = useAuth();

  const [settings, setSettings] = useState<DispatchSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dispatchDate, setDispatchDate] = useState<string>(() =>
    dayjs().format("YYYY-MM-DD")
  );

  const loadSettings = useCallback(async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await getDispatchSettings();
      setSettings(response);
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de charger les paramètres.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const handleSave = useCallback(async () => {
    if (!settings) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      await updateDispatchSettings({
        fairness: { max_gap: settings.fairness.max_gap },
        emergency: {
          emergency_penalty: settings.emergency.emergency_penalty,
        },
        service_times: { ...settings.service_times },
      });
      await refreshEnterprise();
      Alert.alert("Paramètres mis à jour");
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de sauvegarder les paramètres.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [refreshEnterprise, settings]);

  const handleRunDispatch = useCallback(async () => {
    setLoading(true);
    try {
      await runDispatch(dispatchDate);
      Alert.alert("Dispatch lancé", `Dispatch demandé pour ${dispatchDate}.`);
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de lancer le dispatch.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [dispatchDate]);

  const handleRunOptimizer = useCallback(async () => {
    setLoading(true);
    try {
      await runOptimizer(dispatchDate);
      Alert.alert(
        "Optimiseur relancé",
        `Optimisation demandée pour ${dispatchDate}.`
      );
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de relancer l’optimiseur.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [dispatchDate]);

  const handleResetAssignments = useCallback(async () => {
    Alert.alert(
      "Réinitialiser les assignations",
      `Confirmez-vous la suppression des affectations pour ${dispatchDate} ?`,
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Réinitialiser",
          style: "destructive",
          onPress: async () => {
            setLoading(true);
            try {
              await resetAssignments(dispatchDate);
              Alert.alert("Réinitialisation effectuée");
            } catch (error: any) {
              const message =
                error?.response?.data?.error ??
                error?.message ??
                "Impossible de réinitialiser les assignations.";
              setErrorMessage(message);
            } finally {
              setLoading(false);
            }
          },
        },
      ]
    );
  }, [dispatchDate]);

  const takeSettingsField = (path: keyof DispatchSettings["service_times"]) =>
    settings ? settings.service_times[path].toString() : "";

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Paramètres dispatch</Text>
      <Text style={styles.subtitle}>
        Ajustez les règles clés et déclenchez des actions ponctuelles.
      </Text>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Règles d’équité & urgence</Text>
        <LabeledInput
          label="Fairness – écart maximal"
          value={settings ? settings.fairness.max_gap.toString() : ""}
          onChangeText={(value) =>
            setSettings((prev) =>
              prev
                ? {
                    ...prev,
                    fairness: {
                      max_gap: numberOr(value, prev.fairness.max_gap),
                    },
                  }
                : prev
            )
          }
        />
        <LabeledInput
          label="Pénalité chauffeur urgence"
          value={
            settings ? settings.emergency.emergency_penalty.toString() : ""
          }
          onChangeText={(value) =>
            setSettings((prev) =>
              prev
                ? {
                    ...prev,
                    emergency: {
                      emergency_penalty: numberOr(
                        value,
                        prev.emergency.emergency_penalty
                      ),
                    },
                  }
                : prev
            )
          }
        />
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Temps de service</Text>
        <LabeledInput
          label="Temps pickup (min)"
          value={takeSettingsField("pickup_service_min")}
          onChangeText={(value) =>
            setSettings((prev) =>
              prev
                ? {
                    ...prev,
                    service_times: {
                      ...prev.service_times,
                      pickup_service_min: numberOr(
                        value,
                        prev.service_times.pickup_service_min
                      ),
                    },
                  }
                : prev
            )
          }
        />
        <LabeledInput
          label="Temps drop-off (min)"
          value={takeSettingsField("dropoff_service_min")}
          onChangeText={(value) =>
            setSettings((prev) =>
              prev
                ? {
                    ...prev,
                    service_times: {
                      ...prev.service_times,
                      dropoff_service_min: numberOr(
                        value,
                        prev.service_times.dropoff_service_min
                      ),
                    },
                  }
                : prev
            )
          }
        />
        <LabeledInput
          label="Marge transition (min)"
          value={takeSettingsField("min_transition_margin_min")}
          onChangeText={(value) =>
            setSettings((prev) =>
              prev
                ? {
                    ...prev,
                    service_times: {
                      ...prev.service_times,
                      min_transition_margin_min: numberOr(
                        value,
                        prev.service_times.min_transition_margin_min
                      ),
                    },
                  }
                : prev
            )
          }
        />
      </View>

      <TouchableOpacity
        style={styles.primaryButton}
        onPress={handleSave}
        disabled={loading || !settings}
      >
        <Text style={styles.primaryButtonText}>Enregistrer</Text>
      </TouchableOpacity>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Actions rapides</Text>
        <LabeledInput
          label="Date cible (YYYY-MM-DD)"
          value={dispatchDate}
          onChangeText={setDispatchDate}
        />
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleRunDispatch}
          disabled={loading}
        >
          <Text style={styles.secondaryButtonText}>
            Lancer un dispatch semi-auto
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleRunOptimizer}
          disabled={loading}
        >
          <Text style={styles.secondaryButtonText}>Relancer l’optimiseur</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.secondaryButton, styles.dangerButton]}
          onPress={handleResetAssignments}
          disabled={loading}
        >
          <Text style={styles.secondaryButtonText}>Reset assignations</Text>
        </TouchableOpacity>
      </View>

      {errorMessage && <Text style={styles.error}>{errorMessage}</Text>}
    </ScrollView>
  );
}

const LabeledInput = ({
  label,
  value,
  onChangeText,
}: {
  label: string;
  value: string;
  onChangeText: (text: string) => void;
}) => (
  <View style={styles.inputGroup}>
    <Text style={styles.inputLabel}>{label}</Text>
    <TextInput
      style={styles.input}
      value={value}
      onChangeText={onChangeText}
      keyboardType="numeric"
    />
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B1736",
  },
  content: {
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    color: "#FFFFFF",
    fontSize: 22,
    fontWeight: "700",
  },
  subtitle: {
    color: "#9AA5CC",
    marginTop: 6,
    marginBottom: 20,
  },
  card: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 18,
    padding: 18,
    marginBottom: 18,
  },
  sectionTitle: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "600",
    marginBottom: 12,
  },
  inputGroup: {
    marginBottom: 12,
  },
  inputLabel: {
    color: "#B7C5F5",
    fontSize: 14,
    marginBottom: 6,
  },
  input: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 12,
    padding: 12,
    color: "#FFFFFF",
  },
  primaryButton: {
    backgroundColor: "#4D6BFE",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
    marginBottom: 18,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  secondaryButton: {
    backgroundColor: "rgba(255,255,255,0.12)",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
    marginBottom: 12,
  },
  dangerButton: {
    backgroundColor: "rgba(248,113,113,0.2)",
  },
  secondaryButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  error: {
    color: "#F87171",
    marginTop: 12,
    fontSize: 14,
  },
});
