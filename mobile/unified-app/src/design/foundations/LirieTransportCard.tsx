import { StyleSheet, View } from "react-native";
import { LirieButton } from "./LirieButton";
import { LirieLabelBadge, LirieStatusBadge } from "./LirieStatusBadge";
import { LirieText } from "./LirieText";
import { lirieColor, lirieRadius, lirieSpace, type LirieReading, type LirieStatus } from "./tokens";

export type LirieTransportCardProps = {
  reading?: LirieReading;
  time: string;
  status?: LirieStatus;
  statusLabel?: string;
  title: string;
  address: string;
  actionLabel?: string;
  onAction?: () => void;
};

export function LirieTransportCard({
  reading = "comfort",
  time,
  status,
  statusLabel,
  title,
  address,
  actionLabel,
  onAction,
}: LirieTransportCardProps) {
  const operational = reading === "operational";
  return (
    <View
      accessible={false}
      style={[styles.card, operational ? styles.operational : styles.comfort]}
    >
      {operational ? <LirieText variant="data">{time}</LirieText> : null}
      <View style={styles.identity}>
        {operational ? null : <LirieText variant="data">{title}</LirieText>}
        {statusLabel ? <LirieLabelBadge label={statusLabel} /> : status ? <LirieStatusBadge status={status} /> : null}
      </View>
      {operational ? <LirieText variant="description">{title}</LirieText> : <LirieText variant="subtitle">{time}</LirieText>}
      <LirieText variant="description">{address}</LirieText>
      {actionLabel ? (
        <LirieButton label={actionLabel} onPress={onAction} />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: lirieColor.white,
    borderColor: lirieColor.border,
    borderWidth: 1,
    borderRadius: lirieRadius.card,
    gap: lirieSpace.sm,
  },
  comfort: {
    padding: lirieSpace.lg,
  },
  operational: {
    padding: lirieSpace.md,
    borderLeftWidth: 3,
    borderLeftColor: lirieColor.green,
  },
  identity: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: lirieSpace.sm,
  },
});
