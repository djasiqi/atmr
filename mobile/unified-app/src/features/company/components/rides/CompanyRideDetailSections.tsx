import type { ReactNode } from "react";
import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import type {
  MissionTimelineView,
  RideBillingSummary,
  RideDetailInfoRow,
  RideDestinationDetails,
  RideTimelineItem,
} from "../../utils/companyRideDetailPresentation";
import { createShadow } from "../../../../styles/shadowStyles";

const cardShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.03,
  shadowRadius: 4,
  elevation: 1,
});

type SectionProps = {
  title: string;
  icon: keyof typeof Ionicons.glyphMap;
  iconTone?: "brand" | "blue" | "warning" | "muted";
  children: ReactNode;
};

function RideDetailSection({ title, icon, iconTone = "brand", children }: SectionProps) {
  const iconBg =
    iconTone === "blue"
      ? "rgba(37, 99, 235, 0.12)"
      : iconTone === "warning"
        ? "rgba(217, 119, 6, 0.12)"
        : iconTone === "muted"
          ? E.BG
          : "rgba(0, 121, 107, 0.12)";
  const iconColor =
    iconTone === "blue"
      ? "#2563EB"
      : iconTone === "warning"
        ? "#D97706"
        : iconTone === "muted"
          ? E.TEXT_MUTED
          : E.BRAND;

  return (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <View style={[styles.sectionIcon, { backgroundColor: iconBg }]}>
          <Ionicons name={icon} size={14} color={iconColor} />
        </View>
        <AppText variant="sectionTitle" style={styles.sectionTitle}>{title}</AppText>
      </View>
      {children}
    </View>
  );
}

export function RideDetailInfoSection({ rows }: { rows: RideDetailInfoRow[] }) {
  return (
    <RideDetailSection title="Informations" icon="person-outline">
      {rows.map((row, idx) => (
        <View
          key={row.label}
          style={[styles.infoRow, idx === rows.length - 1 ? styles.infoRowLast : null]}
        >
          <AppText variant="bodyMuted" style={styles.infoLabel}>{row.label}</AppText>
          {row.pending ? (
            <View
              style={styles.infoPending}
              accessibilityLabel={`${row.label}, actualisation`}
            />
          ) : (
            <AppText
              variant="body"
              style={[
                styles.infoValue,
                row.tone === "danger" ? styles.infoValueDanger : null,
              ]}
            >
              {row.value}
            </AppText>
          )}
        </View>
      ))}
    </RideDetailSection>
  );
}

export function RideDetailRouteSection({
  pickup,
  dropoff,
  clinicalLine,
}: {
  pickup: string;
  dropoff: string;
  clinicalLine: string | null;
}) {
  return (
    <RideDetailSection title="Trajet" icon="location-outline">
      <View style={styles.routeCard}>
        <View style={styles.routeTrackCol}>
          <View style={[styles.routeDot, styles.routeDotStart]} />
          <View style={styles.routeLine} />
          <View style={[styles.routeDot, styles.routeDotEnd]} />
        </View>
        <View style={styles.routeStops}>
          <View style={styles.routeStop}>
            <AppText variant="caption" style={styles.routeStopLabel}>Départ</AppText>
            <AppText variant="body" style={styles.routeStopAddress}>{pickup}</AppText>
          </View>
          <View style={styles.routeStop}>
            <AppText variant="caption" style={styles.routeStopLabel}>Arrivée</AppText>
            <AppText variant="body" style={styles.routeStopAddress}>{dropoff}</AppText>
            {clinicalLine ? (
              <AppText variant="caption" style={styles.routeStopDetails}>{clinicalLine}</AppText>
            ) : null}
          </View>
        </View>
      </View>
    </RideDetailSection>
  );
}

export function RideDetailMissionSection({ view }: { view: MissionTimelineView }) {
  return (
    <RideDetailSection title="Parcours" icon="map-outline">
      <AppText variant="caption" style={styles.missionKicker}>
        {view.missionLabel}
      </AppText>
      <AppText variant="sectionTitle" style={styles.missionSegment}>
        {view.segmentLabel}
      </AppText>
      <View style={styles.routeCard}>
        <View style={styles.missionStops}>
          {view.stops.map((stop, index) => (
            <View key={stop.key}>
              {stop.showCurrentSegmentMarker ? (
                <AppText variant="caption" style={styles.currentSegmentMarker}>
                  Trajet actuel
                </AppText>
              ) : null}
              <View style={styles.missionStopRow}>
                <View style={styles.routeTrackCol}>
                  <View
                    style={[
                      styles.routeDot,
                      stop.onCurrentSegment ? styles.routeDotEnd : styles.routeDotStart,
                    ]}
                  />
                  {index < view.stops.length - 1 ? <View style={styles.routeLine} /> : null}
                </View>
                <View style={[styles.missionStopBody, stop.onCurrentSegment ? styles.missionStopCurrent : null]}>
                  <AppText variant="caption" style={styles.routeStopLabel}>
                    {stop.title}
                  </AppText>
                  {stop.arrivalLabel ? (
                    <AppText variant="caption" style={styles.missionTime}>
                      {stop.arrivalLabel}
                    </AppText>
                  ) : null}
                  {stop.departureLabel ? (
                    <AppText variant="caption" style={styles.missionTime}>
                      {stop.departureLabel}
                    </AppText>
                  ) : null}
                  <AppText variant="body" style={styles.routeStopAddress}>
                    {stop.location}
                  </AppText>
                  {stop.details.map((line) => (
                    <AppText key={line} variant="caption" style={styles.routeStopDetails}>
                      {line}
                    </AppText>
                  ))}
                </View>
              </View>
            </View>
          ))}
        </View>
      </View>
      {view.currentSegment ? (
        <View style={styles.currentSegmentCard}>
          <AppText variant="caption" style={styles.routeStopLabel}>
            Segment actuel
          </AppText>
          <AppText variant="body" style={styles.routeStopAddress}>
            {view.currentSegment.fromTitle} → {view.currentSegment.toTitle}
          </AppText>
          {view.currentSegment.departureLabel ? (
            <AppText variant="caption" style={styles.missionTime}>
              Départ : {view.currentSegment.departureLabel}
            </AppText>
          ) : null}
          {view.currentSegment.arrivalLabel ? (
            <AppText variant="caption" style={styles.missionTime}>
              Arrivée : {view.currentSegment.arrivalLabel}
            </AppText>
          ) : null}
        </View>
      ) : null}
      {view.missionLevel.length > 0 ? (
        <View style={styles.missionLevel}>
          {view.missionLevel.map((row) => (
            <View key={`${row.label}-${row.value}`} style={styles.detailItem}>
              <AppText variant="bodyMuted" style={styles.detailLabel}>
                {row.label}
              </AppText>
              <AppText variant="body" style={styles.detailValue}>
                {row.value}
              </AppText>
            </View>
          ))}
        </View>
      ) : null}
    </RideDetailSection>
  );
}

export function RideDetailDestinationSection({ destination }: { destination: RideDestinationDetails }) {
  if (!destination.establishment && !destination.service && !destination.doctor) return null;
  return (
    <RideDetailSection title="Destination" icon="business-outline" iconTone="blue">
      {destination.establishment ? (
        <DetailItem label="Établissement" value={destination.establishment} />
      ) : null}
      {destination.service ? <DetailItem label="Service" value={destination.service} /> : null}
      {destination.doctor ? <DetailItem label="Médecin" value={destination.doctor} /> : null}
    </RideDetailSection>
  );
}

export function RideDetailBillingSection({ billing }: { billing: RideBillingSummary }) {
  return (
    <RideDetailSection title="Facturation" icon="document-text-outline" iconTone="warning">
      <View style={styles.billingSummary} accessibilityRole="text">
        <AppText variant="sectionTitle" style={styles.billingAmount}>{billing.amountLabel}</AppText>
        <AppText variant="bodyMuted" style={styles.billingRecipient}>
          Destinataire : {billing.recipientLabel}
        </AppText>
      </View>
      {billing.adjustedNote ? (
        <AppText variant="caption" style={styles.billingNote}>{billing.adjustedNote}</AppText>
      ) : null}
      {billing.invoiceStatusLabel ? (
        <AppText variant="caption" style={styles.billingNote}>
          Facture liée : {billing.invoiceStatusLabel}
        </AppText>
      ) : (
        <AppText variant="caption" style={styles.billingNote}>
          Ajustement du montant et du payeur : utilisez l’application web entreprise.
        </AppText>
      )}
    </RideDetailSection>
  );
}

export function RideDetailTimelineSection({ items }: { items: RideTimelineItem[] }) {
  if (items.length === 0) return null;
  return (
    <RideDetailSection title="Historique" icon="time-outline" iconTone="muted">
      {items.map((item, index) => (
        <View
          key={`${item.date}-${item.event}-${index}`}
          style={[styles.timelineItem, index === items.length - 1 ? styles.timelineItemLast : null]}
        >
          <AppText
            variant="body"
            style={item.type === "cancel" ? styles.timelineEventCancel : styles.timelineEvent}
          >
            {item.event}
          </AppText>
          <AppText variant="caption" style={styles.timelineDate}>
            {dayjs(item.date).isValid() ? dayjs(item.date).format("DD.MM HH:mm") : item.date}
          </AppText>
        </View>
      ))}
    </RideDetailSection>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.detailItem}>
      <AppText variant="bodyMuted" style={styles.detailLabel}>{label}</AppText>
      <AppText variant="body" style={styles.detailValue}>{value}</AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  section: {
    backgroundColor: E.CARD,
    borderRadius: 16,
    padding: 16,
    margin: 16,
    marginBottom: 0,
    borderWidth: 1,
    borderColor: E.BORDER,
    ...cardShadow,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 12,
  },
  sectionIcon: {
    width: 28,
    height: 28,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
  },
  sectionTitle: { color: E.TEXT },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: E.BORDER,
    gap: 12,
  },
  infoRowLast: { marginBottom: 0, paddingBottom: 0, borderBottomWidth: 0 },
  infoLabel: { color: E.TEXT_SEC, fontSize: FONT_SIZE.px13, flex: 1 },
  infoValue: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px13,
    flex: 1.2,
    textAlign: "right",
    fontWeight: "600" as const,
  },
  infoValueDanger: { color: E.DANGER },
  infoPending: {
    width: 96,
    height: 12,
    borderRadius: 6,
    alignSelf: "center",
    backgroundColor: "rgba(15, 23, 42, 0.08)",
  },
  routeCard: {
    flexDirection: "row",
    gap: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: E.BORDER,
    backgroundColor: E.BG,
    paddingHorizontal: 12,
    paddingVertical: 12,
  },
  routeTrackCol: {
    width: 14,
    alignItems: "center",
    paddingTop: 4,
  },
  routeDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: E.BRAND,
    backgroundColor: E.CARD,
  },
  routeDotStart: { borderColor: E.BRAND },
  routeDotEnd: { backgroundColor: E.BRAND, borderColor: E.BRAND },
  routeLine: {
    flex: 1,
    width: 2,
    minHeight: 28,
    backgroundColor: "rgba(0, 121, 107, 0.25)",
    marginVertical: 4,
  },
  routeStops: { flex: 1, gap: 16 },
  routeStop: { gap: 4 },
  routeStopLabel: {
    color: E.TEXT_MUTED,
    fontWeight: "700" as const,
    fontSize: FONT_SIZE.px11,
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },
  routeStopAddress: { color: E.TEXT_SEC, fontSize: FONT_SIZE.px13, lineHeight: 19 },
  routeStopDetails: { color: E.TEXT_MUTED, fontSize: FONT_SIZE.px12, lineHeight: 17 },
  missionKicker: { color: E.TEXT_MUTED, marginBottom: 2 },
  missionSegment: { color: E.TEXT, marginBottom: 12 },
  missionStops: { flex: 1, gap: 0 },
  missionStopRow: { flexDirection: "row", gap: 12 },
  missionStopBody: { flex: 1, gap: 2, paddingBottom: 12 },
  missionStopCurrent: {
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  missionTime: { color: E.TEXT_SEC, fontSize: FONT_SIZE.px12 },
  currentSegmentMarker: {
    color: E.BRAND,
    fontWeight: "700" as const,
    marginLeft: 26,
    marginBottom: 4,
  },
  currentSegmentCard: {
    marginTop: 12,
    gap: 4,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: E.BORDER,
  },
  missionLevel: { marginTop: 12 },
  detailItem: {
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: E.BORDER,
  },
  detailLabel: { color: E.TEXT_MUTED, fontSize: FONT_SIZE.px12, marginBottom: 4 },
  detailValue: { color: E.TEXT, fontSize: FONT_SIZE.px14, fontWeight: "600" as const },
  billingSummary: { gap: 4, marginBottom: 8 },
  billingAmount: { color: E.TEXT, fontSize: FONT_SIZE.px18 },
  billingRecipient: { color: E.TEXT_SEC, fontSize: FONT_SIZE.px13 },
  billingNote: { color: E.TEXT_MUTED, fontSize: FONT_SIZE.px12, lineHeight: 17, marginTop: 4 },
  timelineItem: {
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: E.BORDER,
  },
  timelineItemLast: { marginBottom: 0, paddingBottom: 0, borderBottomWidth: 0 },
  timelineEvent: { color: E.TEXT, fontSize: FONT_SIZE.px13, fontWeight: "500" as const },
  timelineEventCancel: { color: E.DANGER, fontSize: FONT_SIZE.px13, fontWeight: "600" as const },
  timelineDate: { color: E.TEXT_MUTED, fontSize: FONT_SIZE.px12, marginTop: 2 },
});
