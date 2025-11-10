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
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
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

const palette = {
  background: "#07130E",
  heroGradient: ["#11412F", "#07130E"] as [string, string],
  heroBorder: "rgba(46,128,94,0.36)",
  heroText: "#E6F2EA",
  heroMeta: "rgba(184,214,198,0.72)",
  cardBg: "rgba(10,34,26,0.9)",
  cardBorder: "rgba(59,143,105,0.28)",
  cardShadow: "#04150F",
  muted: "rgba(184,214,198,0.75)",
  inputBg: "rgba(5,22,16,0.82)",
  inputBorder: "rgba(59,143,105,0.3)",
  inputText: "#F4FFFA",
  divider: "rgba(46,128,94,0.2)",
  primary: "#1EB980",
  primaryText: "#052015",
  secondaryBg: "rgba(10,34,26,0.6)",
  secondaryBorder: "rgba(59,143,105,0.28)",
  dangerBg: "rgba(241,104,104,0.18)",
  dangerBorder: "rgba(241,104,104,0.3)",
  logoutBg: "rgba(10,34,26,0.55)",
  logoutBorder: "rgba(59,143,105,0.2)",
  error: "#F87171",
};

export default function EnterpriseSettingsScreen() {
  const { refreshEnterprise, logoutEnterprise, switchMode } = useAuth();

  const [settings, setSettings] = useState<DispatchSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dispatchDate, setDispatchDate] = useState<string>(() =>
    dayjs().format("YYYY-MM-DD")
  );

  const heroSubtitle = useMemo(() => {
    if (!settings) return "Chargement des paramètres…";
    return `Veille au bon équilibre de l'algorithme et déclenche les actions critiques sans quitter le mobile.`;
  }, [settings]);

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
      <LinearGradient
        colors={palette.heroGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.hero}
      >
        <View style={{ flex: 1 }}>
          <Text style={styles.heroTitle}>Paramètres dispatch</Text>
          <Text style={styles.heroSubtitle}>{heroSubtitle}</Text>
        </View>
        <View style={styles.heroBadge}>
          <Ionicons name="settings-outline" size={18} color={palette.primaryText} />
          <Text style={styles.heroBadgeText}>
            {settings ? "Actifs" : "Initialisation"}
          </Text>
        </View>
      </LinearGradient>

      <Text style={styles.sectionKicker}>Calibration</Text>
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Règles d’équité & urgence</Text>
        <Text style={styles.sectionDescription}>
          Ajuste le comportement du moteur pour respecter les SLA et la rotation
          des chauffeurs.
        </Text>

        <LabeledInput
          label="Fairness – écart maximal"
          help="Limite acceptable entre chauffeurs pour la charge courante."
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
          help="Pénalité appliquée aux chauffeurs déjà sollicités en urgence."
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
        <Text style={styles.sectionDescription}>
          Définit les durées standards utilisées par l’optimiseur pour planifier
          les créneaux.
        </Text>
        <View style={styles.inputGrid}>
          <LabeledInput
            label="Pickup (min)"
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
            label="Drop-off (min)"
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
      </View>

      <TouchableOpacity
        style={styles.primaryButton}
        onPress={handleSave}
        disabled={loading || !settings}
      >
        <Text style={styles.primaryButtonText}>
          {loading ? "Sauvegarde…" : "Enregistrer les ajustements"}
        </Text>
      </TouchableOpacity>

      <Text style={styles.sectionKicker}>Actions ponctuelles</Text>
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Pilotage quotidien</Text>
        <Text style={styles.sectionDescription}>
          Déclenche les opérations essentielles pour la date sélectionnée.
        </Text>

        <LabeledInput
          label="Date cible"
          help="Format ISO attendu : YYYY-MM-DD"
          value={dispatchDate}
          onChangeText={setDispatchDate}
          keyboardType="default"
        />

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleRunDispatch}
          disabled={loading}
        >
          <Ionicons name="flash-outline" size={18} color={palette.primaryText} />
          <Text style={styles.secondaryButtonText}>Lancer un dispatch</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleRunOptimizer}
          disabled={loading}
        >
          <Ionicons name="sparkles-outline" size={18} color={palette.primaryText} />
          <Text style={styles.secondaryButtonText}>Relancer l’optimiseur</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.secondaryButton, styles.dangerButton]}
          onPress={handleResetAssignments}
          disabled={loading}
        >
          <Ionicons name="alert-circle-outline" size={18} color={palette.error} />
          <Text style={styles.dangerButtonText}>Reset assignations</Text>
        </TouchableOpacity>

        <View style={styles.divider} />

        <TouchableOpacity
          style={[styles.secondaryButton, styles.logoutButton]}
          onPress={() =>
            Alert.alert(
              "Déconnexion",
              "Voulez-vous quitter l'espace entreprise ?",
              [
                { text: "Annuler", style: "cancel" },
                {
                  text: "Se déconnecter",
                  style: "destructive",
                  onPress: async () => {
                    await logoutEnterprise();
                    await switchMode("driver");
                  },
                },
              ]
            )
          }
        >
          <Ionicons name="log-out-outline" size={18} color={palette.primaryText} />
          <Text style={styles.secondaryButtonText}>Se déconnecter</Text>
        </TouchableOpacity>
      </View>

      {errorMessage && (
        <View style={styles.errorBanner}>
          <Ionicons name="alert-triangle" size={18} color={palette.error} />
          <Text style={styles.error}>{errorMessage}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const LabeledInput = ({
  label,
  value,
  help,
  onChangeText,
  keyboardType,
}: {
  label: string;
  value: string;
  help?: string;
  onChangeText: (text: string) => void;
  keyboardType?: "numeric" | "default";
}) => (
  <View style={styles.inputGroup}>
    <View style={styles.inputHeader}>
      <Text style={styles.inputLabel}>{label}</Text>
      {help ? <Text style={styles.inputHelp}>{help}</Text> : null}
    </View>
    <TextInput
      style={styles.input}
      value={value}
      onChangeText={onChangeText}
      keyboardType={keyboardType ?? "numeric"}
      placeholderTextColor={palette.muted}
    />
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: palette.background,
  },
  content: {
    padding: 20,
    paddingBottom: 48,
    gap: 22,
  },
  hero: {
    borderRadius: 24,
    padding: 22,
    flexDirection: "row",
    alignItems: "center",
    gap: 18,
    borderWidth: 1,
    borderColor: palette.heroBorder,
  },
  heroTitle: {
    color: palette.heroText,
    fontSize: 26,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  heroSubtitle: {
    color: palette.heroMeta,
    fontSize: 14,
    marginTop: 6,
  },
  heroBadge: {
    backgroundColor: palette.countPillBg,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: palette.cardBorder,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  heroBadgeText: {
    color: palette.primaryText,
    fontWeight: "700",
    fontSize: 13,
  },
  sectionKicker: {
    color: palette.muted,
    textTransform: "uppercase",
    letterSpacing: 3,
    fontSize: 12,
  },
  card: {
    backgroundColor: palette.cardBg,
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: palette.cardBorder,
    shadowColor: palette.cardShadow,
    shadowOpacity: 0.25,
    shadowOffset: { width: 0, height: 8 },
    shadowRadius: 18,
    elevation: 6,
    gap: 14,
  },
  sectionTitle: {
    color: palette.heroText,
    fontSize: 18,
    fontWeight: "600",
  },
  sectionDescription: {
    color: palette.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  inputGroup: {
    gap: 8,
  },
  inputHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
    gap: 12,
  },
  inputLabel: {
    color: palette.heroText,
    fontSize: 14,
    fontWeight: "600",
  },
  inputHelp: {
    color: palette.muted,
    fontSize: 12,
    flex: 1,
    textAlign: "right",
  },
  input: {
    backgroundColor: palette.inputBg,
    borderRadius: 14,
    padding: 14,
    color: palette.inputText,
    borderWidth: 1,
    borderColor: palette.inputBorder,
    fontSize: 15,
  },
  inputGrid: {
    flexDirection: "column",
    gap: 12,
  },
  primaryButton: {
    backgroundColor: palette.primary,
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: palette.primary,
    shadowOpacity: 0.35,
    shadowOffset: { width: 0, height: 10 },
    shadowRadius: 18,
    elevation: 6,
  },
  primaryButtonText: {
    color: palette.primaryText,
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.4,
  },
  secondaryButton: {
    backgroundColor: palette.secondaryBg,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: palette.secondaryBorder,
    paddingVertical: 14,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  secondaryButtonText: {
    color: palette.primaryText,
    fontSize: 14,
    fontWeight: "600",
  },
  dangerButton: {
    backgroundColor: palette.dangerBg,
    borderColor: palette.dangerBorder,
  },
  dangerButtonText: {
    color: palette.error,
    fontSize: 14,
    fontWeight: "600",
  },
  logoutButton: {
    backgroundColor: palette.logoutBg,
    borderColor: palette.logoutBorder,
  },
  divider: {
    height: 1,
    backgroundColor: palette.divider,
    marginVertical: 12,
  },
  errorBanner: {
    flexDirection: "row",
    gap: 12,
    alignItems: "center",
    backgroundColor: "rgba(241,104,104,0.12)",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(241,104,104,0.24)",
  },
  error: {
    color: palette.error,
    flex: 1,
    fontSize: 13,
  },
});
