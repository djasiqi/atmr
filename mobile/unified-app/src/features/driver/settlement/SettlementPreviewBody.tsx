import { Pressable, StyleSheet, Text, View } from "react-native";
import { DriverSettlementPanel } from "./DriverSettlementPanel";
import type { DriverSettlementContract } from "./driverSettlementContract";

export type PreviewScenario = {
  id: string;
  label: string;
  contract: DriverSettlementContract | null;
};

/**
 * Page de prévisualisation. Si l'accès recette est refusé, aucun scénario
 * n'est affiché : ce n'est pas seulement un bouton masqué ailleurs.
 */
export function SettlementPreviewBody({
  allowed,
  scenarios,
  selected,
  loading,
  offline,
  error,
  onSelect,
  onBack,
}: {
  allowed: boolean;
  scenarios: PreviewScenario[];
  selected: DriverSettlementContract | null;
  loading: boolean;
  offline: boolean;
  error: string | null;
  onSelect: (id: string) => void;
  onBack: () => void;
}) {
  if (!allowed) {
    return (
      <View accessibilityLabel="Prévisualisation du règlement">
        <Text>Rubrique indisponible</Text>
      </View>
    );
  }
  return (
    <View accessibilityLabel="Prévisualisation du règlement" style={styles.page}>
      <Text style={styles.banner}>SIMULATION — AUCUN PAIEMENT RÉEL</Text>
      {scenarios.map((scenario) => (
        <Pressable
          key={scenario.id}
          accessibilityRole="button"
          accessibilityLabel={scenario.label}
          onPress={() => onSelect(scenario.id)}
          style={styles.button}
        >
          <Text style={styles.buttonText}>{scenario.label}</Text>
        </Pressable>
      ))}
      <DriverSettlementPanel
        contract={selected}
        loading={loading}
        offline={offline}
        error={error}
        onBack={onBack}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  page: {
    width: "100%",
    padding: 16,
    gap: 8,
  },
  banner: {
    fontSize: 18,
    lineHeight: 26,
    fontWeight: "700",
  },
  button: {
    minHeight: 48,
    justifyContent: "center",
    paddingHorizontal: 12,
  },
  buttonText: {
    fontSize: 17,
    lineHeight: 24,
  },
});
