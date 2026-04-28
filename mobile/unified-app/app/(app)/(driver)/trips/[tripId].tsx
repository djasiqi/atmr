import { useMemo } from "react";
import { ScrollView, Text } from "react-native";
import { useLocalSearchParams } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { useDriverMissionDetailQuery } from "../../../../src/features/driver/hooks";
import { Card, Loader } from "../../../../src/components/ui";
import { getDriverStatusUx } from "../../../../src/features/driver/statusDictionary";

export default function DriverTripDetailScreen() {
  const params = useLocalSearchParams<{
    tripId?: string;
    pickup?: string;
    dropoff?: string;
    status?: string;
    scheduled?: string;
    source?: string;
  }>();

  const tripId = useMemo(() => {
    const parsed = Number.parseInt(String(params.tripId ?? ""), 10);
    return Number.isFinite(parsed) ? parsed : null;
  }, [params.tripId]);

  const missionDetail = useDriverMissionDetailQuery(tripId);
  const mergedStatus = String(
    missionDetail.data?.status ??
      params.status ??
      "UNKNOWN"
  );
  const ux = getDriverStatusUx(mergedStatus);

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <ScrollView contentContainerStyle={{ padding: 20, gap: 12 }}>
          <Text style={{ fontSize: 22, fontWeight: "700" }}>Detail course</Text>
          {missionDetail.isLoading ? <Loader /> : null}
          {missionDetail.error ? (
            <Text style={{ color: "#B00020" }}>
              {missionDetail.error instanceof Error
                ? missionDetail.error.message
                : "Detail indisponible"}
            </Text>
          ) : null}
          <Card>
            <Text style={{ fontWeight: "700" }}>Course #{params.tripId ?? "N/A"}</Text>
            <Text>Source: {params.source ?? "unknown"}</Text>
            <Text>Statut: {ux.label}</Text>
            <Text>Pickup: {String(missionDetail.data?.pickup_location ?? params.pickup ?? "N/A")}</Text>
            <Text>Dropoff: {String(missionDetail.data?.dropoff_location ?? params.dropoff ?? "N/A")}</Text>
            <Text>Planifie: {String(missionDetail.data?.scheduled_time ?? params.scheduled ?? "N/A")}</Text>
            <Text>
              Derniere mise a jour: {String(missionDetail.data?.updated_at ?? "N/A")}
            </Text>
          </Card>
        </ScrollView>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

