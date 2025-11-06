import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { router } from "expo-router";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";

import { useAuth } from "@/hooks/useAuth";
import {
  getDispatchRides,
  getDispatchStatus,
  switchDispatchMode,
} from "@/services/enterpriseDispatch";
import { DispatchStatus, RideSummary } from "@/types/enterpriseDispatch";

dayjs.extend(utc);
dayjs.extend(timezone);
dayjs.extend(relativeTime);
dayjs.locale("fr");

const QUICK_LINKS = [
  {
    title: "Courses",
    subtitle: "Consulter & agir",
    route: "/(enterprise)/rides",
  },
  {
    title: "Paramètres",
    subtitle: "Modes, relance, reset",
    route: "/(enterprise)/settings",
  },
  {
    title: "Chat",
    subtitle: "Echanger avec l’équipe",
    route: "/(enterprise)/chat",
  },
];

type ModeValue = "manual" | "semi_auto" | "fully_auto";

export default function EnterpriseDashboardScreen() {
  const {
    enterpriseSession,
    refreshEnterprise,
    enterpriseLoading,
    switchMode,
  } = useAuth();

  const companyName = enterpriseSession?.company.name ?? "Entreprise";
  const dispatchMode =
    (enterpriseSession?.company.dispatchMode as ModeValue | undefined) ??
    "semi_auto";

  const [status, setStatus] = useState<DispatchStatus | null>(null);
  const [urgentRides, setUrgentRides] = useState<RideSummary[]>([]);
  const [unassignedRides, setUnassignedRides] = useState<RideSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const currentDate = useMemo(() => {
    const now = dayjs().tz ? dayjs().tz("Europe/Zurich") : dayjs();
    return now.format("YYYY-MM-DD");
  }, []);

  const loadData = useCallback(async () => {
    if (!enterpriseSession) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const [statusResponse, urgentResponse, unassignedResponse] =
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
        ]);
      setStatus(statusResponse);
      setUrgentRides(urgentResponse.items);
      setUnassignedRides(unassignedResponse.items);
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de charger les dernières informations dispatch.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [currentDate, enterpriseSession]);

  useEffect(() => {
    if (!enterpriseSession) return;
    loadData();
  }, [enterpriseSession, loadData]);

  const handleModeChange = useCallback(
    async (target: ModeValue) => {
      if (target === dispatchMode) return;
      setLoading(true);
      try {
        await switchDispatchMode(
          target,
          `Changement de mode via mobile (${dispatchMode} → ${target})`
        );
        await refreshEnterprise();
        await loadData();
      } catch (error: any) {
        const message =
          error?.response?.data?.error ??
          error?.message ??
          "Impossible de changer le mode dispatch.";
        setErrorMessage(message);
      } finally {
        setLoading(false);
      }
    },
    [dispatchMode, loadData, refreshEnterprise]
  );

  const isRefreshing = loading || enterpriseLoading;
  const kpis = status?.kpis;
  const osrm = status?.osrm;
  const agent = status?.agent;
  const optimizer = status?.optimizer;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={isRefreshing} onRefresh={loadData} />
      }
    >
      <Text style={styles.title}>Bienvenue</Text>
      <Text style={styles.company}>{companyName}</Text>

      <View style={styles.modeCard}>
        <Text style={styles.modeHeading}>Mode actif</Text>
        <Text style={styles.modeValue}>{labelForMode(dispatchMode)}</Text>
        <Text style={styles.modeSubtitle}>
          Dernier tick agent{" "}
          {agent?.last_tick ? dayjs(agent.last_tick).fromNow() : "inconnu"}
        </Text>
        <View style={styles.modeButtons}>
          {(["manual", "semi_auto", "fully_auto"] as ModeValue[]).map(
            (mode) => (
              <TouchableOpacity
                key={mode}
                style={[
                  styles.modeButton,
                  mode === dispatchMode && styles.modeButtonActive,
                ]}
                onPress={() => handleModeChange(mode)}
              >
                <Text
                  style={[
                    styles.modeButtonText,
                    mode === dispatchMode && styles.modeButtonTextActive,
                  ]}
                >
                  {labelForMode(mode)}
                </Text>
              </TouchableOpacity>
            )
          )}
        </View>
      </View>

      <View style={styles.quickLinks}>
        {QUICK_LINKS.map((link) => (
          <TouchableOpacity
            key={link.route}
            style={styles.quickCard}
            onPress={() => router.push(link.route as any)}
          >
            <Text style={styles.quickTitle}>{link.title}</Text>
            <Text style={styles.quickSubtitle}>{link.subtitle}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <View style={styles.kpiRow}>
        <KpiTile
          label="Courses assignées"
          value={
            kpis ? `${kpis.assigned_bookings}/${kpis.total_bookings}` : "—"
          }
        />
        <KpiTile
          label="Taux assignation"
          value={kpis ? `${Math.round(kpis.assignment_rate * 100)} %` : "—"}
        />
        <KpiTile
          label="Courses à risque"
          value={kpis ? String(kpis.at_risk) : "—"}
        />
      </View>

      <View style={styles.statusRow}>
        <StatusCard
          label="OSRM"
          status={osrm?.status ?? "WARNING"}
          detail={
            osrm?.latency_ms != null
              ? `${osrm.latency_ms} ms`
              : "Latence inconnue"
          }
        />
        <StatusCard
          label="Agent"
          status={agent?.active ? "OK" : "WARNING"}
          detail={agent ? agent.mode : "Inactif"}
        />
        <StatusCard
          label="Optimiseur"
          status={optimizer?.active ? "OK" : "WARNING"}
          detail={
            optimizer?.next_window_start
              ? dayjs(optimizer.next_window_start).format("HH:mm")
              : "Aucune fenêtre"
          }
        />
      </View>

      <Section title="Alertes urgentes">
        {urgentRides.length === 0 ? (
          <Text style={styles.muted}>Aucune urgence en cours.</Text>
        ) : (
          urgentRides.map((ride) => (
            <RideAlert key={ride.id} ride={ride} badge="Urgent" />
          ))
        )}
      </Section>

      <Section title="Courses à assigner">
        {unassignedRides.length === 0 ? (
          <Text style={styles.muted}>Toutes les courses sont affectées.</Text>
        ) : (
          unassignedRides.map((ride) => (
            <TouchableOpacity
              key={ride.id}
              style={styles.rideCard}
              onPress={() =>
                router.push({
                  pathname: "/(enterprise)/ride-details",
                  params: { rideId: ride.id },
                } as any)
              }
            >
              <Text style={styles.rideTitle}>{ride.client.name}</Text>
              <Text style={styles.rideText}>{ride.route.pickup_address}</Text>
              <Text style={styles.rideText}>
                → {ride.route.dropoff_address}
              </Text>
              <Text style={styles.rideTime}>
                {ride.time.pickup_at
                  ? dayjs(ride.time.pickup_at).format("HH:mm")
                  : "⏱️ À définir"}
              </Text>
            </TouchableOpacity>
          ))
        )}
      </Section>

      {errorMessage && <Text style={styles.error}>{errorMessage}</Text>}

      <TouchableOpacity
        style={styles.refreshButton}
        onPress={loadData}
        disabled={isRefreshing}
      >
        <Text style={styles.refreshButtonText}>
          Rafraîchir les informations
        </Text>
      </TouchableOpacity>
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

const KpiTile = ({ label, value }: { label: string; value: string }) => (
  <View style={styles.kpiCard}>
    <Text style={styles.kpiValue}>{value}</Text>
    <Text style={styles.kpiLabel}>{label}</Text>
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
          : "⏱️ À définir"}
      </Text>
    </View>
    <Text style={styles.alertClient}>{ride.client.name}</Text>
    <Text style={styles.alertRoute}>{ride.route.pickup_address}</Text>
    <Text style={styles.alertRoute}>→ {ride.route.dropoff_address}</Text>
  </TouchableOpacity>
);

const labelForMode = (mode: ModeValue) => {
  switch (mode) {
    case "manual":
      return "Manuel";
    case "semi_auto":
      return "Semi-auto";
    case "fully_auto":
      return "Fully-auto";
    default:
      return mode;
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B1736",
  },
  content: {
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    color: "#6F8BFF",
    textTransform: "uppercase",
    letterSpacing: 1,
    fontSize: 14,
    fontWeight: "600",
  },
  company: {
    color: "#FFFFFF",
    fontSize: 26,
    fontWeight: "700",
    marginBottom: 20,
  },
  modeCard: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 18,
    padding: 18,
    marginBottom: 16,
  },
  modeHeading: {
    color: "#B8C4FF",
    fontSize: 14,
  },
  modeValue: {
    color: "#FFFFFF",
    fontSize: 22,
    fontWeight: "700",
    marginTop: 4,
  },
  modeSubtitle: {
    color: "#9AA5CC",
    marginTop: 4,
  },
  modeButtons: {
    flexDirection: "row",
    gap: 8,
    marginTop: 12,
  },
  modeButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
    alignItems: "center",
  },
  modeButtonActive: {
    backgroundColor: "#4D6BFE",
    borderColor: "#4D6BFE",
  },
  modeButtonText: {
    color: "#D7DFFF",
    fontWeight: "600",
  },
  modeButtonTextActive: {
    color: "#FFFFFF",
  },
  quickLinks: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 18,
  },
  quickCard: {
    flex: 1,
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: 16,
  },
  quickTitle: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  quickSubtitle: {
    color: "#9AA5CC",
    marginTop: 8,
    fontSize: 13,
  },
  kpiRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 18,
  },
  kpiCard: {
    flex: 1,
    backgroundColor: "rgba(255,255,255,0.07)",
    borderRadius: 16,
    paddingVertical: 18,
    alignItems: "center",
  },
  kpiValue: {
    color: "#FFFFFF",
    fontSize: 20,
    fontWeight: "700",
  },
  kpiLabel: {
    color: "#9AA5CC",
    marginTop: 6,
    fontSize: 13,
  },
  statusRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 18,
  },
  statusCard: {
    flex: 1,
    backgroundColor: "rgba(20,28,60,0.9)",
    borderRadius: 14,
    padding: 14,
  },
  statusLabel: {
    color: "#9AA5CC",
    fontSize: 13,
  },
  statusValue: {
    fontSize: 16,
    fontWeight: "700",
    marginTop: 6,
  },
  statusDetail: {
    color: "#E2E8FF",
    marginTop: 4,
    fontSize: 13,
  },
  section: {
    backgroundColor: "rgba(255,255,255,0.05)",
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 12,
  },
  muted: {
    color: "#9AA5CC",
  },
  alertCard: {
    backgroundColor: "rgba(248,113,113,0.1)",
    borderRadius: 14,
    padding: 14,
    marginBottom: 12,
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
    color: "#E2E8FF",
  },
  alertClient: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 15,
    marginBottom: 4,
  },
  alertRoute: {
    color: "#C7D1FF",
    fontSize: 13,
  },
  rideCard: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 14,
    padding: 14,
    marginBottom: 12,
  },
  rideTitle: {
    color: "#FFFFFF",
    fontWeight: "600",
    marginBottom: 6,
  },
  rideText: {
    color: "#C7D1FF",
    fontSize: 13,
  },
  rideTime: {
    color: "#FFFFFF",
    marginTop: 8,
    fontWeight: "700",
  },
  error: {
    color: "#F87171",
    marginTop: 12,
  },
  refreshButton: {
    backgroundColor: "#4D6BFE",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: 8,
  },
  refreshButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
});
