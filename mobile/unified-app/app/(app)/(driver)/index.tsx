import { useMemo, useState } from "react";
import { Pressable, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useActiveDriverContextId,
  useDriverAvailabilityMutation,
  useDriverMissionsQuery,
  useDriverRealtimeSync,
} from "../../../src/features/driver/hooks";
import { getDriverStatusUx } from "../../../src/features/driver/statusDictionary";
import { useSession } from "../../../src/core/sessionProvider";
import type { DriverMission } from "../../../src/features/driver/types";
import { useDriverChatMessages, useUnreadMessages } from "../../../src/features/driver/chatHooks";
import { DashboardMissionListSkeleton } from "../../../src/features/driver/components/DashboardMissionListSkeleton";
import { MissionCard } from "../../../src/features/driver/components/MissionCard";
import { StatusSwitch } from "../../../src/features/driver/components/StatusSwitch";
import { DriverStateBanners } from "../../../src/features/driver/components/DriverStateBanners";
import { filterNextMissionsOnly } from "../../../src/features/driver/domain/missionGrouping";

function selectActiveMission(missions: DriverMission[] | undefined): DriverMission | null {
  if (!Array.isArray(missions) || missions.length === 0) return null;
  const nextScope = filterNextMissionsOnly(missions);
  if (nextScope.length > 0) return nextScope[0] ?? null;
  const firstNonTerminal = missions.find((mission) => {
    const ux = getDriverStatusUx(typeof mission.status === "string" ? mission.status : null);
    return !ux.terminal;
  });
  return firstNonTerminal ?? missions[0] ?? null;
}

export default function DriverHomeScreen() {
  const router = useRouter();
  const { status: sessionStatus, activeContext } = useSession();
  const driverContextId = useActiveDriverContextId();
  const missionsQuery = useDriverMissionsQuery();
  useDriverRealtimeSync();
  const companyId = useMemo(() => {
    const fromContext =
      typeof activeContext?.organization_id === "number"
        ? activeContext.organization_id
        : typeof activeContext?.organization_id === "string"
          ? Number.parseInt(activeContext.organization_id, 10)
          : null;
    if (fromContext && Number.isFinite(fromContext)) return fromContext;
    const missionCompany = (missionsQuery.data as DriverMission[] | undefined)
      ?.map((mission) => {
        const candidate = mission["company_id"];
        if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
        if (typeof candidate === "string") {
          const parsed = Number.parseInt(candidate, 10);
          return Number.isFinite(parsed) ? parsed : null;
        }
        return null;
      })
      .find((value): value is number => value != null);
    return missionCompany ?? null;
  }, [activeContext?.organization_id, missionsQuery.data]);
  const chatMessagesQuery = useDriverChatMessages(companyId, driverContextId);
  const unread = useUnreadMessages(companyId, driverContextId, chatMessagesQuery.data);
  const activeMission = selectActiveMission(missionsQuery.data as DriverMission[] | undefined);
  const availabilityMutation = useDriverAvailabilityMutation();
  const [isAvailable, setIsAvailable] = useState(true);
  const bootstrapPending = sessionStatus !== "ready" || missionsQuery.isLoading;

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
          <Text style={{ fontSize: 22, fontWeight: "700" }}>Espace Driver</Text>
          <Text>Missions, transitions, offline replay, realtime et tracking.</Text>
          <DriverStateBanners />

          <StatusSwitch
            mode="availability"
            isAvailable={isAvailable}
            pending={availabilityMutation.isPending}
            onToggleAvailability={() => {
              const next = !isAvailable;
              availabilityMutation.mutate(next, {
                onSuccess: () => setIsAvailable(next),
              });
            }}
          />

          {bootstrapPending ? <Text>Initialisation mission active...</Text> : null}
          {bootstrapPending ? <DashboardMissionListSkeleton /> : null}
          {missionsQuery.isError ? (
            <Text>
              Erreur chargement missions: {(missionsQuery.error as Error)?.message ?? "Erreur"}
            </Text>
          ) : null}

          {!bootstrapPending && activeMission ? (
            <MissionCard
              mission={activeMission}
              onOpen={(missionId) =>
                router.push({
                  pathname: "/(app)/(driver)/missions/[missionId]",
                  params: { missionId: String(missionId) },
                })
              }
            />
          ) : !bootstrapPending ? (
            <Text>Aucune mission active.</Text>
          ) : null}

          <Pressable
            onPress={() => router.push("/(app)/(driver)/missions")}
            style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
            disabled={bootstrapPending}
          >
            <Text style={{ fontWeight: "600" }}>Voir toutes les missions</Text>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/trips")}
            style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
            disabled={bootstrapPending}
          >
            <Text style={{ fontWeight: "600" }}>Ouvrir Courses</Text>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/schedule" as any)}
            style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
            disabled={bootstrapPending}
          >
            <Text style={{ fontWeight: "600" }}>Ouvrir Planning</Text>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/chat")}
            style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
            disabled={bootstrapPending}
          >
            <Text style={{ fontWeight: "600" }}>
              Ouvrir Chat {unread.unreadCount > 0 ? `(${unread.unreadCount})` : ""}
            </Text>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/profile")}
            style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}
            disabled={bootstrapPending}
          >
            <Text style={{ fontWeight: "600" }}>Voir Profil</Text>
          </Pressable>
        </View>
      </PermissionGuard>
    </DriverContextGuard>
  );
}
