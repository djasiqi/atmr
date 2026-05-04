import { StyleSheet, View } from "react-native";
import { AppButton } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";

const BORDER = "rgba(145, 165, 157, 0.4)";

type TransferCardProps = {
  missionId: number;
  busy?: boolean;
  onPress: () => void;
};

export function TransferCard({ missionId, busy = false, onPress }: TransferCardProps) {
  return (
    <View style={s.root}>
      <AppText variant="label">Transfert de course</AppText>
      <AppText variant="caption" style={s.subSpacing}>
        Course #{missionId}
      </AppText>
      <AppButton
        title={busy ? "Ouverture…" : "Transférer"}
        variant="secondary"
        onPress={onPress}
        disabled={busy}
      />
    </View>
  );
}

const s = StyleSheet.create({
  root: { borderWidth: 1, borderColor: BORDER, borderRadius: 10, paddingVertical: 7, paddingHorizontal: 8, gap: 4, backgroundColor: "#FAFCFB" },
  subSpacing: { marginBottom: 1 },
});
