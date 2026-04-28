import { StyleSheet, Text, View } from "react-native";
import { Button } from "../../../../components/ui";

const BORDER = "rgba(145, 165, 157, 0.4)";

type TransferCardProps = {
  missionId: number;
  busy?: boolean;
  onPress: () => void;
};

export function TransferCard({ missionId, busy = false, onPress }: TransferCardProps) {
  return (
    <View style={s.root}>
      <Text style={s.title}>Transfert de course</Text>
      <Text style={s.sub}>Course #{missionId}</Text>
      <Button
        label={busy ? "Ouverture…" : "Transférer"}
        onPress={onPress}
        disabled={busy}
      />
    </View>
  );
}

const s = StyleSheet.create({
  root: { borderWidth: 1, borderColor: BORDER, borderRadius: 10, paddingVertical: 7, paddingHorizontal: 8, gap: 4, backgroundColor: "#FAFCFB" },
  title: { fontSize: 13, fontWeight: "800", color: "#163A34" },
  sub: { color: "#5F7369", fontSize: 11, marginBottom: 1 },
});
