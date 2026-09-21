import { useMemo } from "react";
import { StyleSheet } from "react-native";
import { useLocalSearchParams } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { useDriverMissionDetailQuery } from "../../../../src/features/driver/hooks";
import { AppCard, AppSpinner, AppText, brandSurfaceSoft, Screen } from "../../../../src/design/responsive";
import { getDriverStatusUx } from "../../../../src/features/driver/statusDictionary";
import {
  DELIVERY_BENEFICIARY_LABEL,
  DELIVERY_CARGO_LABEL,
  formatDeliveryDescriptionDisplay,
  formatMissionTypeLabel,
  isMaterialDeliveryMission,
  MISSION_DELIVERY_BADGE,
} from "../../../../src/features/driver/domain/missionDisplay";
import { useDriverFloatingTabScrollPadding } from "../../../../src/features/driver/navigation/DriverFloatingTabBar";

export default function DriverTripDetailScreen() {
  const params = useLocalSearchParams<{
    tripId?: string;
    pickup?: string;
    dropoff?: string;
    status?: string;
    scheduled?: string;
    source?: string;
    client?: string;
    driver?: string;
  }>();
  const scrollPad = useDriverFloatingTabScrollPadding();

  const tripId = useMemo(() => {
    const parsed = Number.parseInt(String(params.tripId ?? ""), 10);
    return Number.isFinite(parsed) ? parsed : null;
  }, [params.tripId]);

  const missionDetail = useDriverMissionDetailQuery(tripId);
  const mergedStatus = missionDetail.data?.status ?? params.status;
  const ux = getDriverStatusUx(mergedStatus);

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          extraScrollBottomPadding={scrollPad}
          contentContainerStyle={styles.page}
        >
          <AppText variant="sectionTitle" style={styles.title}>
            Détail course
          </AppText>
          {missionDetail.isLoading ? <AppSpinner size="small" /> : null}
          {missionDetail.error ? (
            <AppText variant="error" style={styles.error}>
              {params.source === "company_day"
                ? "Détail API limité sur cette course entreprise : affichage des données du planning."
                : missionDetail.error instanceof Error
                  ? missionDetail.error.message
                  : "Détail indisponible"}
            </AppText>
          ) : null}
          <AppCard>
            <AppText variant="label" style={styles.cardTitle}>
              {isMaterialDeliveryMission(missionDetail.data)
                ? `${MISSION_DELIVERY_BADGE} #${params.tripId ?? "N/A"}`
                : `Course #${params.tripId ?? "N/A"}`}
            </AppText>
            {isMaterialDeliveryMission(missionDetail.data) ? (
              <>
                <AppText variant="body" style={styles.body}>
                  {formatMissionTypeLabel("material_delivery")}
                </AppText>
                <AppText variant="body" style={styles.body}>
                  {DELIVERY_CARGO_LABEL} : {formatDeliveryDescriptionDisplay(missionDetail.data)}
                </AppText>
              </>
            ) : null}
            {String(missionDetail.data?.client_name ?? params.client ?? "").trim().length > 0 ? (
              <AppText variant="body" style={styles.body}>
                {isMaterialDeliveryMission(missionDetail.data) ? DELIVERY_BENEFICIARY_LABEL : "Client"} : {String(missionDetail.data?.client_name ?? params.client ?? "N/A")}
              </AppText>
            ) : null}
            {String(params.driver ?? "").trim().length > 0 ? (
              <AppText variant="body" style={styles.body}>
                Chauffeur assigné : {String(params.driver)}
              </AppText>
            ) : null}
            <AppText variant="body" style={styles.body}>
              Source : {params.source ?? "unknown"}
            </AppText>
            {!isMaterialDeliveryMission(missionDetail.data) ? (
              <AppText variant="body" style={styles.body}>
                Type : {formatMissionTypeLabel(
                  typeof missionDetail.data?.mission_type === "string"
                    ? missionDetail.data.mission_type
                    : null
                )}
              </AppText>
            ) : null}
            <AppText variant="body" style={styles.body}>
              Statut : {ux.label}
            </AppText>
            <AppText variant="body" style={styles.body}>
              Pickup : {String(missionDetail.data?.pickup_location ?? params.pickup ?? "N/A")}
            </AppText>
            <AppText variant="body" style={styles.body}>
              Dropoff : {String(missionDetail.data?.dropoff_location ?? params.dropoff ?? "N/A")}
            </AppText>
            <AppText variant="body" style={styles.body}>
              Planifié : {String(missionDetail.data?.scheduled_time ?? params.scheduled ?? "N/A")}
            </AppText>
            <AppText variant="body" style={styles.body}>
              Dernière mise à jour : {String(missionDetail.data?.updated_at ?? "N/A")}
            </AppText>
          </AppCard>
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    padding: 20,
    gap: 12,
    paddingBottom: 28,
  },
  title: {
    color: "#0f172a",
  },
  cardTitle: {
    fontWeight: "700",
    color: "#0f172a",
    marginBottom: 4,
  },
  body: {
    color: "#334155",
    marginTop: 2,
  },
  error: {
    color: "#B42318",
  },
});
