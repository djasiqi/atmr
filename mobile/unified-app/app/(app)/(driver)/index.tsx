import { useMemo, useState } from "react";
import { Pressable } from "react-native";
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
import { AppText, Screen, useAppViewport, useResponsiveTokens } from "../../../src/design/responsive";

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
  const t = useResponsiveTokens();
  const { horizontalPadding } = useAppViewport();
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

  const scrollContentStyle = useMemo(
    () => ({
      flexGrow: 1,
      paddingHorizontal: horizontalPadding,
      paddingTop: t.spacingSm + t.spacingXs,
      gap: t.spacingSm + t.spacingXs,
      paddingBottom: t.spacingMd,
    }),
    [horizontalPadding, t.spacingSm, t.spacingXs, t.spacingMd]
  );

  const outlineBtn = useMemo(
    () => ({
      borderWidth: 1,
      borderColor: "#e2e8f0",
      borderRadius: t.radiusSm,
      padding: t.spacingSm,
    }),
    [t.radiusSm, t.spacingSm]
  );

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor="#f8fafc"
          withHorizontalPadding={false}
          contentContainerStyle={scrollContentStyle}
        >
          <AppText variant="screenTitle">Espace Driver</AppText>
          <AppText variant="bodyMuted">
            Missions, transitions, offline replay, realtime et tracking.
          </AppText>
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

          {bootstrapPending ? (
            <AppText variant="caption">Initialisation mission active…</AppText>
          ) : null}
          {bootstrapPending ? <DashboardMissionListSkeleton /> : null}
          {missionsQuery.isError ? (
            <AppText variant="error">
              Erreur chargement missions: {(missionsQuery.error as Error)?.message ?? "Erreur"}
            </AppText>
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
            <AppText variant="bodyMuted">Aucune mission active.</AppText>
          ) : null}

          <Pressable
            onPress={() => router.push("/(app)/(driver)/missions")}
            style={outlineBtn}
            disabled={bootstrapPending}
          >
            <AppText variant="label">Voir toutes les missions</AppText>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/trips")}
            style={outlineBtn}
            disabled={bootstrapPending}
          >
            <AppText variant="label">Ouvrir Courses</AppText>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/schedule" as any)}
            style={outlineBtn}
            disabled={bootstrapPending}
          >
            <AppText variant="label">Ouvrir Planning</AppText>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/chat")}
            style={outlineBtn}
            disabled={bootstrapPending}
          >
            <AppText variant="label">
              Ouvrir Chat {unread.unreadCount > 0 ? `(${unread.unreadCount})` : ""}
            </AppText>
          </Pressable>
          <Pressable
            onPress={() => router.push("/(app)/(driver)/profile")}
            style={outlineBtn}
            disabled={bootstrapPending}
          >
            <AppText variant="label">Voir Profil</AppText>
          </Pressable>
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}
