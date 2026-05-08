import { View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, Modal } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { E } from "../../company/theme/enterpriseOpsTheme";

type ReleaseConfirmationModalProps = {
  visible: boolean;
  missionId: number | null;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

/**
 * Confirmation de libération de mission (parité `operations-app/MissionCard.tsx`
 * lignes 1070+). Sémantique distincte d'« Annuler » : le chauffeur rend la
 * mission au pool dispatch — réassignée à un autre chauffeur, sans facturation.
 *
 * Côté API : transition `CANCELLED` avec `reason: "RELEASE"`
 * (cf. `useDriverStatusTransition` → `api.ts:180`).
 */
export function ReleaseConfirmationModal(props: ReleaseConfirmationModalProps) {
  return (
    <Modal
      visible={props.visible}
      title="Libérer la course"
      onClose={props.onCancel}
    >
      <View style={{ gap: 12 }}>
        <AppText variant="body">
          Libérer la mission #{props.missionId ?? "n/a"} ? Elle sera réassignée à un autre chauffeur.
        </AppText>

        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            gap: 8,
            paddingVertical: 10,
            paddingHorizontal: 12,
            backgroundColor: "rgba(0, 121, 107, 0.04)",
            borderRadius: 10,
            borderWidth: 1,
            borderColor: "rgba(0, 121, 107, 0.08)",
          }}
        >
          <Ionicons name="information-circle-outline" size={16} color={E.BRAND} />
          <AppText variant="caption" style={{ color: E.TEXT_SEC, flex: 1, lineHeight: 17 }}>
            Sans facturation : opération dispatch, pas une annulation client.
          </AppText>
        </View>

        <View style={{ flexDirection: "row", gap: 8 }}>
          <AppButton
            title="Annuler"
            variant="secondary"
            onPress={props.onCancel}
            disabled={props.pending}
          />
          <AppButton
            title={props.pending ? "Libération..." : "Libérer"}
            variant="primary"
            onPress={props.onConfirm}
            disabled={props.pending}
          />
        </View>
      </View>
    </Modal>
  );
}
