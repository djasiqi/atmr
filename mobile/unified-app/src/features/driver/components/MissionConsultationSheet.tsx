import type { ReactNode } from "react";
import { StyleSheet, View } from "react-native";
import { AppCard, AppText } from "../../../design/responsive";
import { DriverCourseStatusBadge } from "./DriverCourseStatusBadge";
import { MissionSettlementRead } from "./MissionSettlementRead";
import type { DriverMission } from "../types";
import { getMissionClientDisplayName } from "../domain/missionDisplay";
import {
  ABSENT,
  ABSENT_TIME,
  GPS_UNAVAILABLE,
  accessLines,
  actualTimeRow,
  assistanceLines,
  hasMissionCoordinates,
  reachedSteps,
  scheduledTimeRow,
  sequenceLabel,
  tripTypeLabel,
} from "../domain/missionConsultation";

type Props = {
  mission: DriverMission;
  etaLabel: string;
  stacked: boolean;
  late: boolean;
  map: ReactNode;
};

export function MissionConsultationSheet({ mission, etaLabel, stacked, late, map }: Props) {
  const scheduled = scheduledTimeRow(mission);
  const pickup = actualTimeRow("Prise en charge réelle", mission.boarded_at);
  const arrival = actualTimeRow("Arrivée réelle", mission.arrived_at);
  const end = actualTimeRow("Fin réelle", mission.completed_at);
  const sequence = sequenceLabel(mission);
  const steps = reachedSteps(mission);

  return (
    <View style={styles.stack} accessibilityLabel="Détails de mission, consultation">
      <Section title="Identification">
        <Fact label="Mission" value={`N° ${mission.id}`} stacked={stacked} />
        <View style={styles.statusRow}>
          <AppText variant="bodyMuted">Statut</AppText>
          <DriverCourseStatusBadge status={String(mission.status ?? "")} late={late} />
        </View>
        <Fact label="Date du transport" value={scheduled.value} stacked={stacked} />
        <Fact label="Type de trajet" value={tripTypeLabel(mission)} stacked={stacked} />
        {sequence ? <Fact label="Tournée" value={sequence} stacked={stacked} /> : null}
        <Fact
          label="Entreprise"
          value={typeof mission.company_name === "string" && mission.company_name.trim()
            ? mission.company_name.trim()
            : ABSENT}
          stacked={stacked}
        />
      </Section>

      <Section title="Bénéficiaire">
        <Fact label="Nom" value={getMissionClientDisplayName(mission)} stacked={stacked} />
      </Section>

      <Section title="Parcours">
        <Fact
          label="Départ"
          value={textOrAbsent(mission.pickup_location)}
          stacked={stacked}
        />
        <Fact
          label="Arrivée"
          value={textOrAbsent(mission.dropoff_location)}
          stacked={stacked}
        />
        {accessLines(mission).map((line) => (
          <AppText key={line} variant="body">
            {line}
          </AppText>
        ))}
      </Section>

      <Section title="Horaires">
        <Fact label={scheduled.label} value={scheduled.value} stacked={stacked} />
        <Fact label={pickup.label} value={pickup.value} stacked={stacked} />
        <Fact label={arrival.label} value={arrival.value} stacked={stacked} />
        <Fact label={end.label} value={end.value} stacked={stacked} />
      </Section>

      <Section title="Assistance">
        {assistanceLines(mission).map((line) => (
          <AppText key={line} variant="body">
            {line}
          </AppText>
        ))}
      </Section>

      <Section title="Transport">
        <Fact label="Chauffeur" value={textOrAbsent(mission.driver_name)} stacked={stacked} />
        <Fact label="Véhicule" value={ABSENT} stacked={stacked} />
        <Fact label="Distance" value={ABSENT_TIME} stacked={stacked} />
        <Fact label="Estimation" value={etaLabel} stacked={stacked} />
        <AppText variant="body">
          {hasMissionCoordinates(mission) ? "Coordonnées GPS disponibles" : GPS_UNAVAILABLE}
        </AppText>
        {map}
      </Section>

      <Section title="Règlement — consultation">
        <MissionSettlementRead missionId={mission.id} stacked={stacked} />
      </Section>

      <Section title="Historique de mission">
        <AppText variant="bodyMuted">{"Étapes atteintes d'après le statut. Sans journal séparé."}</AppText>
        {steps.map((step) => (
          <AppText key={step.label} variant="body">
            {step.at ? `${step.label} · ${step.at}` : step.label}
          </AppText>
        ))}
      </Section>
    </View>
  );
}

function textOrAbsent(value: unknown): string {
  if (typeof value !== "string") return ABSENT;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : ABSENT;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <AppCard style={styles.card} accessibilityLabel={title}>
      <AppText variant="sectionTitle" accessibilityRole="header">
        {title}
      </AppText>
      {children}
    </AppCard>
  );
}

function Fact({ label, value, stacked }: { label: string; value: string; stacked: boolean }) {
  return (
    <View
      style={stacked ? styles.stacked : styles.row}
      accessibilityLabel={`${label}. ${value}`}
    >
      <AppText variant="bodyMuted">{label}</AppText>
      <AppText variant="body" style={stacked ? undefined : styles.value}>
        {value}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  stack: {
    width: "100%",
    gap: 12,
  },
  card: {
    gap: 8,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
  },
  stacked: {
    gap: 2,
  },
  value: {
    flex: 1,
    textAlign: "right",
  },
  statusRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
  },
});
