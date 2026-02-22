import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  MissionStateManager,
  type MissionBarStatus,
  type MissionState,
} from "@/services/missionState";
import {
  showMissionNotification,
  dismissMissionNotification,
} from "@/services/missionBarAndroid";
import { openNavigation, safeCall } from "@/services/deepLinks";
import { getAssignedTrips, type Booking } from "@/services/api";
import { normalizeBookingStatus } from "@/utils/bookingStatus";
import { getCallablePhone } from "@/utils/phone";

const MISSIONS_CACHE_KEY = "missions_cache_v2";
const ACTIVE_STATE_KEY = "active_mission_state";

type ScreenPhase = "loading" | "ready" | "confirming" | "done" | "error";

export default function QuickActionScreen() {
  const params = useLocalSearchParams<{
    fromNavigation?: string;
    bookingId?: string;
  }>();
  const router = useRouter();
  const fromNavigation = params.fromNavigation === "true";

  const [phase, setPhase] = useState<ScreenPhase>("loading");
  const [state, setState] = useState<MissionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const autoReturnTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // -- Hydration (cold start safe) ----------------------------------------

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const hydrated = await MissionStateManager.ensureHydrated();
      if (cancelled) return;
      if (hydrated && MissionStateManager.isActive()) {
        setState(MissionStateManager.getState());
        setPhase("ready");
        return;
      }

      // Fallback 1: AsyncStorage
      try {
        const raw = await AsyncStorage.getItem(ACTIVE_STATE_KEY);
        if (raw) {
          const saved = JSON.parse(raw);
          if (saved.activeMission) {
            setState(saved as MissionState);
            setPhase("ready");
            return;
          }
        }
      } catch {}

      // Fallback 2: missions cache
      try {
        const raw = await AsyncStorage.getItem(MISSIONS_CACHE_KEY);
        if (raw) {
          const missions: Booking[] = JSON.parse(raw);
          const active = missions.find((m) => {
            const s = normalizeBookingStatus(m.status);
            return s === "ASSIGNED" || s === "EN_ROUTE" || s === "IN_PROGRESS";
          });
          if (active) {
            const status = normalizeBookingStatus(active.status) as MissionBarStatus;
            setState({
              activeMission: active,
              nextBookingPreview: null,
              currentStatus: status,
              allowedTransitions: getAllowedTransitions(status),
              allowedActions: [],
              isNavigating: fromNavigation,
              lastNavigationDestination: null,
            });
            setPhase("ready");
            return;
          }
        }
      } catch {}

      // Fallback 3: fetch (with timeout)
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 1500);
        const bookings = await getAssignedTrips();
        clearTimeout(timeout);
        if (cancelled) return;
        const active = bookings.find((m) => {
          const s = normalizeBookingStatus(m.status);
          return s === "ASSIGNED" || s === "EN_ROUTE" || s === "IN_PROGRESS";
        });
        if (active) {
          await MissionStateManager.startMission(active);
          setState(MissionStateManager.getState());
          setPhase("ready");
          return;
        }
      } catch {}

      if (!cancelled) setPhase("error");
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // -- Subscribe to state changes -----------------------------------------

  useEffect(() => {
    const unsub = MissionStateManager.subscribe((event, newState) => {
      setState(newState);
      if (event === "mission_stopped") {
        setPhase("done");
      }
    });
    return unsub;
  }, []);

  // -- Actions ------------------------------------------------------------

  const handleTransition = useCallback(
    async (targetStatus: MissionBarStatus) => {
      setPhase("confirming");
      const ok = await MissionStateManager.requestTransition(targetStatus);
      if (!ok) {
        setError("Transition invalide");
        setPhase("error");
        return;
      }
      if (Platform.OS !== "web") {
        if (targetStatus === "COMPLETED") {
          await MissionStateManager.stopMission();
          await dismissMissionNotification();
        } else {
          await showMissionNotification(MissionStateManager.getState());
        }
      }
      setState(MissionStateManager.getState());
      setPhase("done");

      if (fromNavigation && state?.lastNavigationDestination) {
        autoReturnTimer.current = setTimeout(() => {
          openNavigation(state.lastNavigationDestination!);
        }, 2000);
      }
    },
    [fromNavigation, state]
  );

  const handleCall = useCallback(() => {
    const phone = MissionStateManager.getCallablePhone();
    if (phone) safeCall(phone);
  }, []);

  const handleReturnToNav = useCallback(() => {
    if (autoReturnTimer.current) clearTimeout(autoReturnTimer.current);
    const dest =
      state?.lastNavigationDestination ??
      state?.activeMission?.dropoff_location ??
      state?.activeMission?.pickup_location;
    if (dest) openNavigation(dest);
    else router.back();
  }, [state, router]);

  const handleGoBack = useCallback(() => {
    router.replace("/(tabs)/mission");
  }, [router]);

  useEffect(() => {
    return () => {
      if (autoReturnTimer.current) clearTimeout(autoReturnTimer.current);
    };
  }, []);

  // -- Render -------------------------------------------------------------

  if (phase === "loading") {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#fff" />
        <Text style={styles.loadingText}>Chargement...</Text>
      </View>
    );
  }

  if (phase === "error" || !state?.activeMission) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>{error ?? "Mission introuvable"}</Text>
        <TouchableOpacity style={styles.secondaryBtn} onPress={handleGoBack}>
          <Text style={styles.secondaryBtnText}>Revenir à l'app</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (phase === "done") {
    return (
      <View style={styles.container}>
        <Text style={styles.checkmark}>✓</Text>
        <Text style={styles.doneText}>Statut mis à jour</Text>
        <TouchableOpacity style={styles.primaryBtn} onPress={handleReturnToNav}>
          <Text style={styles.primaryBtnText}>Retour à la navigation</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.secondaryBtn} onPress={handleGoBack}>
          <Text style={styles.secondaryBtnText}>Rester dans l'app</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (phase === "confirming") {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#fff" />
        <Text style={styles.loadingText}>Mise à jour en cours...</Text>
      </View>
    );
  }

  const mission = state.activeMission;
  const destination =
    state.currentStatus === "IN_PROGRESS"
      ? mission.dropoff_location
      : mission.pickup_location;

  return (
    <View style={styles.container}>
      <View style={styles.infoBlock}>
        <Text style={styles.statusLabel}>
          {formatStatusLabel(state.currentStatus)} → {shortenAddress(destination)}
        </Text>
        {state.nextBookingPreview && (
          <Text style={styles.nextLabel}>
            Prochaine{" "}
            {state.nextBookingPreview.pickup_at
              ? new Date(state.nextBookingPreview.pickup_at).toLocaleTimeString(
                  "fr-CH",
                  { hour: "2-digit", minute: "2-digit" }
                )
              : ""}{" "}
            ·{" "}
            {state.nextBookingPreview.can_show_identity
              ? state.nextBookingPreview.client_display
              : "Course suivante"}{" "}
            · {state.nextBookingPreview.pickup_short}
          </Text>
        )}
      </View>

      <View style={styles.actionsBlock}>
        {state.currentStatus === "ASSIGNED" && (
          <TouchableOpacity
            style={[styles.actionBtn, styles.actionBtnPrimary]}
            onPress={() => handleTransition("EN_ROUTE")}
          >
            <Text style={styles.actionBtnText}>En route</Text>
          </TouchableOpacity>
        )}
        {state.currentStatus === "EN_ROUTE" && (
          <TouchableOpacity
            style={[styles.actionBtn, styles.actionBtnPrimary]}
            onPress={() => handleTransition("IN_PROGRESS")}
          >
            <Text style={styles.actionBtnText}>À bord</Text>
          </TouchableOpacity>
        )}
        {state.currentStatus === "IN_PROGRESS" && (
          <TouchableOpacity
            style={[styles.actionBtn, styles.actionBtnComplete]}
            onPress={() => handleTransition("COMPLETED")}
          >
            <Text style={styles.actionBtnText}>Terminer</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity
          style={[styles.actionBtn, styles.actionBtnCall]}
          onPress={handleCall}
        >
          <Text style={styles.actionBtnText}>Appeler</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// -- Helpers ---------------------------------------------------------------

function getAllowedTransitions(status: MissionBarStatus): MissionBarStatus[] {
  const map: Record<MissionBarStatus, MissionBarStatus[]> = {
    ASSIGNED: ["EN_ROUTE"],
    EN_ROUTE: ["IN_PROGRESS"],
    IN_PROGRESS: ["COMPLETED"],
    COMPLETED: [],
  };
  return map[status] ?? [];
}

function formatStatusLabel(status: MissionBarStatus): string {
  switch (status) {
    case "ASSIGNED":
      return "ASSIGNÉE";
    case "EN_ROUTE":
      return "EN ROUTE";
    case "IN_PROGRESS":
      return "À BORD";
    case "COMPLETED":
      return "TERMINÉE";
    default:
      return status;
  }
}

function shortenAddress(addr: string | undefined): string {
  if (!addr) return "…";
  const parts = addr.split(",");
  return parts[0]?.trim().substring(0, 30) ?? addr.substring(0, 30);
}

// -- Styles ----------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#1a1a2e",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  infoBlock: {
    width: "100%",
    marginBottom: 32,
    alignItems: "center",
  },
  statusLabel: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "700",
    textAlign: "center",
    marginBottom: 8,
  },
  nextLabel: {
    color: "#aaa",
    fontSize: 15,
    textAlign: "center",
  },
  actionsBlock: {
    width: "100%",
    gap: 12,
  },
  actionBtn: {
    width: "100%",
    paddingVertical: 18,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 56,
  },
  actionBtnPrimary: {
    backgroundColor: "#4361ee",
  },
  actionBtnComplete: {
    backgroundColor: "#2ecc71",
  },
  actionBtnCall: {
    backgroundColor: "#334155",
  },
  actionBtnText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
  primaryBtn: {
    backgroundColor: "#4361ee",
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    marginTop: 20,
    width: "100%",
    alignItems: "center",
  },
  primaryBtnText: {
    color: "#fff",
    fontSize: 17,
    fontWeight: "700",
  },
  secondaryBtn: {
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 12,
    marginTop: 12,
    width: "100%",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#555",
  },
  secondaryBtnText: {
    color: "#ccc",
    fontSize: 16,
    fontWeight: "500",
  },
  checkmark: {
    fontSize: 64,
    color: "#2ecc71",
    marginBottom: 12,
  },
  doneText: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "600",
  },
  loadingText: {
    color: "#ccc",
    fontSize: 16,
    marginTop: 12,
  },
  errorText: {
    color: "#e74c3c",
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 20,
    textAlign: "center",
  },
});
