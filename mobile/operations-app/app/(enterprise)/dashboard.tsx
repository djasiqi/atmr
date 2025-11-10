import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";
import { Ionicons } from "@expo/vector-icons";

import { useAuth } from "@/hooks/useAuth";
import {
  getDispatchRides,
  getDispatchStatus,
  markRideUrgent,
  runDispatch,
} from "@/services/enterpriseDispatch";
import { DispatchStatus, RideSummary } from "@/types/enterpriseDispatch";
import { RideSnippetCard } from "@/components/enterprise/cards/RideSnippetCard";
import { EnterpriseDriversMap } from "@/components/enterprise/EnterpriseDriversMap";
import { useEnterpriseDriverTracking } from "@/hooks/useEnterpriseDriverTracking";
import { useEnterpriseContext } from "@/context/EnterpriseContext";

const enterprisePalette = {
  background: "#07130E",
  heroGradient: ["#11412F", "#07130E"] as [string, string],
  heroKpiSurface: "rgba(22,76,55,0.45)",
  heroKpiBorder: "rgba(59,168,123,0.28)",
  heroKicker: "rgba(226,242,233,0.7)",
  heroTitle: "#E6F2EA",
  heroMeta: "rgba(214,236,224,0.65)",
  heroTick: "#79E0AE",
  surface: "rgba(9,28,21,0.88)",
  surfaceBorder: "rgba(52,143,105,0.18)",
  surfaceMuted: "rgba(184,214,198,0.72)",
  alertSurface: "rgba(241,104,104,0.16)",
  alertBorder: "rgba(241,104,104,0.3)",
  alertText: "rgba(234,246,240,0.85)",
  hintText: "rgba(198,225,211,0.82)",
  dispatchButton: "#1EB980",
  dispatchButtonDisabled: "rgba(30,185,128,0.45)",
  dispatchText: "#052015",
  sectionSurface: "rgba(9,24,18,0.88)",
  sectionBorder: "rgba(59,143,105,0.24)",
  cardOverlay: "rgba(18,58,42,0.65)",
  cardBorder: "rgba(61,147,110,0.26)",
  textStrong: "#F1FFF9",
  textSecondary: "rgba(200,231,213,0.78)",
};

dayjs.extend(utc);
dayjs.extend(timezone);
dayjs.extend(relativeTime);
dayjs.locale("fr");

export default function EnterpriseDashboardScreen() {
  const { enterpriseSession, refreshEnterprise, enterpriseLoading } = useAuth();
  const { selectedDate } = useEnterpriseContext();

  const dispatchMode =
    (enterpriseSession?.company.dispatchMode as
      | "manual"
      | "semi_auto"
      | "fully_auto"
      | undefined) ?? "semi_auto";

  const companyName = enterpriseSession?.company.name ?? "Entreprise";

  const [status, setStatus] = useState<DispatchStatus | null>(null);
  const [urgentRides, setUrgentRides] = useState<RideSummary[]>([]);
  const [unassignedRides, setUnassignedRides] = useState<RideSummary[]>([]);
  const [allRides, setAllRides] = useState<RideSummary[]>([]);
  const [pendingUrgentRide, setPendingUrgentRide] = useState<string | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dispatching, setDispatching] = useState(false);

  const { markers: driverMarkers, refreshLocations } =
    useEnterpriseDriverTracking();

  const currentDate = useMemo(() => selectedDate, [selectedDate]);

  const formattedDay = useMemo(() => {
    const base = dayjs(selectedDate);
    const localized = dayjs.isDayjs(base)
      ? base
      : dayjs(selectedDate, "YYYY-MM-DD");
    const zoned = localized.tz ? localized.tz("Europe/Zurich") : localized;
    return zoned.format("dddd D MMMM");
  }, [selectedDate]);

  const loadData = useCallback(async () => {
    if (!enterpriseSession) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const [statusResponse, urgentResponse, unassignedResponse, allResponse] =
        await Promise.all([
          getDispatchStatus(),
          getDispatchRides({
            date: currentDate,
            status: "urgent",
            page_size: 5,
          }),
          getDispatchRides({
            date: currentDate,
            status: "unassigned",
            page_size: 3,
          }),
          getDispatchRides({
            date: currentDate,
            page_size: 120,
          }),
        ]);
      setStatus(statusResponse);
      setUrgentRides(urgentResponse.items);
      setUnassignedRides(unassignedResponse.items);
      setAllRides(allResponse.items);
      refreshLocations();
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de charger les dernières informations dispatch.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [currentDate, enterpriseSession, refreshLocations]);

  useEffect(() => {
    if (!enterpriseSession) return;
    loadData();
  }, [enterpriseSession, loadData, currentDate]);

  const handleUrgentDelay = useCallback(
    async (rideId: string) => {
      setPendingUrgentRide(rideId);
      try {
        await markRideUrgent(rideId, {
          extra_delay_minutes: 15,
          reason: "Action mobile: urgence +15",
        });
        Alert.alert(
          "Urgence",
          "La course a été marquée urgente avec un délai +15 minutes."
        );
        await loadData();
      } catch (error: any) {
        const message =
          error?.response?.data?.error ??
          error?.message ??
          "Impossible de marquer la course urgente.";
        Alert.alert("Erreur", message);
      } finally {
        setPendingUrgentRide(null);
      }
    },
    [loadData]
  );

  const manualStats = useMemo(() => {
    const total = allRides.length;
    const unassigned = allRides.filter((ride) => ride.status === "unassigned");
    const assigned = allRides.filter((ride) => ride.status === "assigned");
    const completed = allRides.filter((ride) => ride.status === "completed");
    return {
      total,
      unassignedCount: unassigned.length,
      assignedCount: assigned.length,
      completedCount: completed.length,
      assignmentRate:
        total > 0 ? Math.round((assigned.length / total) * 100) : 0,
    };
  }, [allRides]);

  const heroKpis = useMemo(() => {
    if (dispatchMode === "manual") {
      return [
        {
          id: "manual-total",
          label: "Total",
          value: String(manualStats.total),
        },
        {
          id: "manual-assigned",
          label: "En cours",
          value: String(manualStats.assignedCount),
        },
      ];
    }

    const kpis = status?.kpis;
    return [
      {
        id: "auto-total",
        label: "Total",
        value: kpis ? String(kpis.total_bookings) : "—",
      },
      {
        id: "auto-assigned",
        label: "En cours",
        value: kpis ? String(kpis.assigned_bookings) : "—",
      },
    ];
  }, [dispatchMode, manualStats, status?.kpis]);

  const sortedManualRides = useMemo(() => {
    const withTime: RideSummary[] = [];
    const withoutTime: RideSummary[] = [];

    allRides.forEach((ride) => {
      if (ride.time.pickup_at) {
        const moment = dayjs(ride.time.pickup_at);
        if (moment.hour() === 0 && moment.minute() === 0) {
          withoutTime.push(ride);
        } else {
          withTime.push(ride);
        }
      } else {
        withoutTime.push(ride);
      }
    });

    withTime.sort(
      (a, b) =>
        dayjs(a.time.pickup_at!).valueOf() - dayjs(b.time.pickup_at!).valueOf()
    );

    return [...withTime, ...withoutTime];
  }, [allRides]);

  const manualRidesList = (
    <View style={styles.manualListSection}>
      <Text style={styles.sectionTitle}>Courses du jour</Text>
      {sortedManualRides.length === 0 ? (
        <Text style={styles.muted}>
          Aucune course planifiée pour cette date.
        </Text>
      ) : (
        sortedManualRides.map((ride) => {
          let pickupTime: string | null = null;
          if (ride.time.pickup_at) {
            const pickupMoment = dayjs(ride.time.pickup_at);
            pickupTime =
              pickupMoment.hour() === 0 && pickupMoment.minute() === 0
                ? null
                : pickupMoment.format("HH[h]mm");
          }
          const priorityBadge =
            ride.client.priority === "HIGH"
              ? { label: "Priorité", tone: "danger" as const }
              : ride.client.priority === "LOW"
                ? { label: "Basse", tone: "info" as const }
                : undefined;

          return (
            <RideSnippetCard
              key={ride.id}
              ride={{
                id: ride.id,
                time: pickupTime ?? "",
                showUndefinedIcon: pickupTime === null,
                client: ride.client.name,
                pickup: ride.route.pickup_address,
                dropoff: ride.route.dropoff_address,
                assignedTo: ride.driver?.name ?? null,
                badges: priorityBadge ? [priorityBadge] : undefined,
                onPress: () =>
                  router.push({
                    pathname: "/(enterprise)/ride-details",
                    params: { rideId: ride.id },
                  } as any),
                onQuickAction: () => handleUrgentDelay(ride.id),
                onPrimaryAction: () =>
                  router.push({
                    pathname: "/(enterprise)/ride-details",
                    params: { rideId: ride.id },
                  } as any),
              }}
            />
          );
        })
      )}
    </View>
  );

  const manualMapSection = (
    <View style={styles.manualMapSection}>
      <EnterpriseDriversMap
        markers={driverMarkers}
        fallbackMessage="Activez le tracking pour visualiser les chauffeurs en temps réel."
      />
    </View>
  );

  const handleRunDispatch = useCallback(() => {
    if (dispatching) return;
    Alert.alert(
      "Lancer un dispatch ?",
      `Confirme le lancement d'une optimisation pour ${dayjs(
        currentDate
      ).format("dddd D MMMM")}.`,
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Lancer",
          style: "default",
          onPress: async () => {
            setDispatching(true);
            try {
              const response = await runDispatch(currentDate);
              const confirmation = (() => {
                if (!response?.message) {
                  return `Dispatch lancé pour le ${dayjs(currentDate).format(
                    "DD/MM/YYYY"
                  )}`;
                }
                return response.message
                  .replace("Dispatch lancé pour", "Dispatch lancé pour le")
                  .replace(/(\d{4})-(\d{2})-(\d{2})/, "$3/$2/$1");
              })();
              Alert.alert("Dispatch lancé", confirmation);
              await loadData();
            } catch (error: any) {
              const message =
                error?.response?.data?.error ??
                error?.message ??
                "Impossible de lancer le dispatch. Réessaie plus tard.";
              Alert.alert("Erreur dispatch", message);
            } finally {
              setDispatching(false);
            }
          },
        },
      ]
    );
  }, [currentDate, dispatching, formattedDay, loadData]);

  const semiAutoControls = (
    <View style={styles.semiAutoControls}>
      <Text style={styles.sectionTitle}>Mode semi-automatique</Text>
      <Text style={styles.dispatchHint}>
        Laisse l’optimisation préparer les assignations et finalise-les en un
        clic. Relance le dispatch à chaque nouvelle vague de courses.
      </Text>
      <TouchableOpacity
        style={[
          styles.dispatchButton,
          dispatching && styles.dispatchButtonDisabled,
        ]}
        onPress={handleRunDispatch}
        disabled={dispatching}
        activeOpacity={0.85}
      >
        <Ionicons
          name={dispatching ? "time-outline" : "flash-outline"}
          size={18}
          color="#0B1736"
        />
        <Text style={styles.dispatchButtonText}>
          {dispatching ? "Dispatch en cours…" : "Lancer un dispatch"}
        </Text>
      </TouchableOpacity>
    </View>
  );

  const isRefreshing = loading || enterpriseLoading;
  const kpis = status?.kpis;
  const osrm = status?.osrm;
  const agent = status?.agent;
  const optimizer = status?.optimizer;
  const isManual = dispatchMode === "manual";
  const isSemiAuto = dispatchMode === "semi_auto";

  const urgentSection = (
    <Section title="Alertes urgentes">
      {urgentRides.length === 0 ? (
        <Text style={styles.muted}>Aucune urgence en cours.</Text>
      ) : (
        urgentRides.map((ride) => (
          <RideAlert key={ride.id} ride={ride} badge="Urgent" />
        ))
      )}
    </Section>
  );

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={isRefreshing} onRefresh={loadData} />
      }
    >
      <LinearGradient
        colors={enterprisePalette.heroGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.hero}
      >
        <View style={styles.heroHeader}>
          <View style={{ flex: 1 }}>
            <Text style={styles.heroKicker}>Tableau de bord</Text>
            <Text style={styles.heroCompany}>{companyName}</Text>
          </View>
          <View style={styles.heroMeta}>
            <Text style={styles.heroDate}>{formattedDay}</Text>
            <Text style={styles.heroTick}>
              Agent {agent?.last_tick ? dayjs(agent.last_tick).fromNow() : "—"}
            </Text>
          </View>
        </View>

        {heroKpis.length > 0 && (
          <View style={styles.heroKpiRow}>
            {heroKpis.map((kpi) => (
              <View key={kpi.id} style={styles.heroKpiCard}>
                <Text style={styles.heroKpiValue}>{kpi.value}</Text>
                <Text style={styles.heroKpiLabel}>{kpi.label}</Text>
              </View>
            ))}
          </View>
        )}
      </LinearGradient>

      {manualMapSection}

      {isManual && manualRidesList}

      {isSemiAuto && (
        <>
          {semiAutoControls}
          {manualRidesList}
          {urgentSection}
        </>
      )}

      {dispatchMode === "fully_auto" && urgentSection}

      {errorMessage && <Text style={styles.error}>{errorMessage}</Text>}
    </ScrollView>
  );
}

const Section = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) => (
  <View style={styles.section}>
    <Text style={styles.sectionTitle}>{title}</Text>
    {children}
  </View>
);

const StatusCard = ({
  label,
  status,
  detail,
}: {
  label: string;
  status: "OK" | "WARNING" | "DOWN";
  detail: string;
}) => {
  const color =
    status === "OK" ? "#4ADE80" : status === "WARNING" ? "#FACC15" : "#F87171";
  return (
    <View style={styles.statusCard}>
      <Text style={styles.statusLabel}>{label}</Text>
      <Text style={[styles.statusValue, { color }]}>{status}</Text>
      <Text style={styles.statusDetail}>{detail}</Text>
    </View>
  );
};

const RideAlert = ({ ride, badge }: { ride: RideSummary; badge: string }) => (
  <TouchableOpacity
    style={styles.alertCard}
    onPress={() =>
      router.push({
        pathname: "/(enterprise)/ride-details",
        params: { rideId: ride.id },
      } as any)
    }
  >
    <View style={styles.alertHeader}>
      <Text style={styles.alertBadge}>{badge}</Text>
      <Text style={styles.alertTime}>
        {ride.time.pickup_at
          ? dayjs(ride.time.pickup_at).format("HH:mm")
          : "⏱️"}
      </Text>
    </View>
    <Text style={styles.alertClient}>{ride.client.name}</Text>
    <Text style={styles.alertRoute}>{ride.route.pickup_address}</Text>
    <Text style={styles.alertRoute}>→ {ride.route.dropoff_address}</Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: enterprisePalette.background,
  },
  content: {
    padding: 20,
    paddingBottom: 80,
  },
  hero: {
    borderRadius: 24,
    padding: 20,
    marginBottom: 22,
    overflow: "hidden",
  },
  heroHeader: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 10,
  },
  heroKpiRow: {
    flexDirection: "row",
    gap: 8,
    marginTop: 8,
  },
  heroKpiCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: enterprisePalette.heroKpiSurface,
    borderWidth: 1,
    borderColor: enterprisePalette.heroKpiBorder,
  },
  heroKpiValue: {
    color: enterprisePalette.heroTitle,
    fontWeight: "700",
    fontSize: 14,
  },
  heroKpiLabel: {
    color: enterprisePalette.heroMeta,
    fontSize: 12,
  },
  heroKicker: {
    color: enterprisePalette.heroKicker,
    fontSize: 13,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 6,
  },
  heroCompany: {
    color: enterprisePalette.heroTitle,
    fontSize: 28,
    fontWeight: "700",
  },
  heroMeta: {
    alignItems: "flex-end",
    gap: 4,
  },
  heroDate: {
    color: enterprisePalette.heroMeta,
    fontSize: 13,
    textTransform: "capitalize",
  },
  heroTick: {
    color: enterprisePalette.heroTick,
    fontSize: 12,
  },
  modeSwitch: {
    flexDirection: "row",
    backgroundColor: "rgba(10,17,38,0.55)",
    borderRadius: 16,
    padding: 6,
    gap: 6,
  },
  modePill: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 10,
    alignItems: "center",
  },
  modePillActive: {
    backgroundColor: "rgba(255,255,255,0.16)",
  },
  modePillText: {
    color: "rgba(255,255,255,0.65)",
    fontWeight: "600",
  },
  modePillTextActive: {
    color: "#FFFFFF",
  },
  statusRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 18,
  },
  statusCard: {
    flex: 1,
    backgroundColor: enterprisePalette.cardOverlay,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: enterprisePalette.cardBorder,
  },
  statusLabel: {
    color: enterprisePalette.heroMeta,
    fontSize: 13,
  },
  statusValue: {
    fontSize: 16,
    fontWeight: "700",
    marginTop: 6,
  },
  statusDetail: {
    color: enterprisePalette.textSecondary,
    marginTop: 4,
    fontSize: 13,
  },
  section: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 20,
    padding: 18,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
  },
  sectionTitle: {
    color: enterprisePalette.textStrong,
    fontSize: 17,
    fontWeight: "600",
    marginBottom: 12,
  },
  muted: {
    color: enterprisePalette.surfaceMuted,
  },
  alertCard: {
    backgroundColor: enterprisePalette.alertSurface,
    borderRadius: 16,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: enterprisePalette.alertBorder,
  },
  alertHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  alertBadge: {
    color: "#F87171",
    fontWeight: "700",
  },
  alertTime: {
    color: enterprisePalette.alertText,
  },
  alertClient: {
    color: enterprisePalette.textStrong,
    fontWeight: "600",
    fontSize: 15,
    marginBottom: 4,
  },
  alertRoute: {
    color: enterprisePalette.alertText,
    fontSize: 13,
  },
  semiAutoControls: {
    marginTop: 24,
    padding: 20,
    borderRadius: 18,
    backgroundColor: enterprisePalette.surface,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    gap: 12,
  },
  dispatchHint: {
    color: enterprisePalette.hintText,
    fontSize: 13,
    lineHeight: 18,
  },
  dispatchButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    alignSelf: "flex-start",
    backgroundColor: enterprisePalette.dispatchButton,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 14,
    shadowColor: enterprisePalette.dispatchButton,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 6,
  },
  dispatchButtonDisabled: {
    backgroundColor: enterprisePalette.dispatchButtonDisabled,
    shadowOpacity: 0.15,
  },
  dispatchButtonText: {
    color: enterprisePalette.dispatchText,
    fontWeight: "700",
    fontSize: 14,
    letterSpacing: 0.2,
    textTransform: "uppercase",
  },
  rideCard: {
    backgroundColor: enterprisePalette.cardOverlay,
    borderRadius: 16,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: enterprisePalette.cardBorder,
  },
  rideTitle: {
    color: enterprisePalette.textStrong,
    fontWeight: "600",
    marginBottom: 6,
  },
  rideText: {
    color: enterprisePalette.textSecondary,
    fontSize: 13,
  },
  rideTime: {
    color: enterprisePalette.textStrong,
    marginTop: 8,
    fontWeight: "700",
  },
  error: {
    color: "#F87171",
    marginTop: 12,
  },
  manualListSection: {
    marginTop: 24,
    gap: 12,
  },
  manualActionsRow: {
    flexDirection: "row",
    gap: 8,
  },
  manualSecondaryAction: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(148,163,255,0.45)",
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  manualSecondaryText: {
    color: "rgba(214,224,255,0.85)",
    fontWeight: "600",
    fontSize: 12,
  },
  manualPrimaryAction: {
    borderRadius: 999,
    backgroundColor: "#5EEAD4",
    paddingHorizontal: 16,
    paddingVertical: 6,
  },
  manualPrimaryText: {
    color: "#0B1736",
    fontWeight: "700",
    fontSize: 12,
  },
  manualMapSection: {
    marginTop: 14,
    gap: 12,
    marginBottom: 24,
  },
});
