import { View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { AppButton } from "../../../design/responsive";
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
        <AppText variant="sectionTitle">Disponibilite chauffeur</AppText>
        <AppText variant="body">{available ? "Disponible" : "Indisponible"}</AppText>
        <AppButton
          title={
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
      <AppText variant="sectionTitle">Transitions mission</AppText>
      <AppText variant="body">Etat courant: {ux.label}</AppText>
      {ux.nextTransitions.length === 0 ? (
        <AppText variant="bodyMuted">Aucune transition disponible.</AppText>
      ) : (
        ux.nextTransitions.map((target) => (
          <AppButton
            key={target}
            title={props.pending ? "Enregistrement..." : `Passer ${target}`}
            variant="secondary"
            onPress={() => props.onTransition?.(target)}
            disabled={props.disabled || props.pending}
          />
        ))
      )}
    </View>
  );
}
