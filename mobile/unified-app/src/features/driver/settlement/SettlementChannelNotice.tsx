import { StyleSheet, View } from "react-native";
import { LirieText } from "../../../design/foundations/LirieText";
import { lirieColor, lirieRadius, lirieSpace } from "../../../design/foundations/tokens";

/** Distingue la simulation des futurs modules, qui restent fermés. */
export function SettlementChannelNotice() {
  return (
    <View style={styles.box}>
      <LirieText variant="subtitle">Simulation de recette</LirieText>
      <LirieText variant="description">Module espèces réel verrouillé</LirieText>
      <LirieText variant="description">Module facture patient réel verrouillé</LirieText>
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    gap: lirieSpace.xs,
    padding: lirieSpace.md,
    backgroundColor: lirieColor.warningFill,
    borderRadius: lirieRadius.card,
  },
});
