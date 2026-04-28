import { Text, View } from "react-native";
import { Button } from "../../../components/ui";
import type { DriverMissionStatus, DriverTransitionStatus } from "../types";
import { getDriverStatusUx } from "../statusDictionary";

type Props = {
  mode: "availability" | "mission";
  isAvailable?: boolean;
  onToggleAvailability?: () => void;
  missionStatus?: DriverMissionStatus | null;
  onTransition?: (target: DriverTransitionStatus) => void;
  disabled?: boolean;
  pending?: boolean;
};

export function StatusSwitch(props: Props) {
  if (props.mode === "availability") {
    const available = Boolean(props.isAvailable);
    return (
      <View style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 8 }}>
        <Text style={{ fontWeight: "700" }}>Disponibilite chauffeur</Text>
        <Text>{available ? "Disponible" : "Indisponible"}</Text>
        <Button
          label={
            props.pending
              ? "Mise a jour..."
              : available
                ? "Passer indisponible"
                : "Passer disponible"
          }
          variant="primary"
          onPress={() => props.onToggleAvailability?.()}
          disabled={props.disabled || props.pending}
        />
      </View>
    );
  }

  const ux = getDriverStatusUx(props.missionStatus ?? null);
  return (
    <View style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 8 }}>
      <Text style={{ fontWeight: "700" }}>Transitions mission</Text>
      <Text>Etat courant: {ux.label}</Text>
      {ux.nextTransitions.length === 0 ? (
        <Text style={{ color: "#666" }}>Aucune transition disponible.</Text>
      ) : (
        ux.nextTransitions.map((target) => (
          <Button
            key={target}
            label={props.pending ? "Enregistrement..." : `Passer ${target}`}
            onPress={() => props.onTransition?.(target)}
            disabled={props.disabled || props.pending}
          />
        ))
      )}
    </View>
  );
}
