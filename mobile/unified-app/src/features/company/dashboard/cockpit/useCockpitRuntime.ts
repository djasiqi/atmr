import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CompanyDispatchMission } from "../../api/contracts";
import type { CockpitFsmEvent } from "./cockpitFiniteStateMachine";
import type { CockpitEvent } from "./cockpitEventBus";
import {
  resolveCockpitOrchestration,
  type CockpitOrchestrationDecision,
} from "./cockpitOrchestrator";
import { createCockpitRuntimeStore } from "./cockpitRuntimeStore";
import type { CockpitHumanOverride, CockpitRuntimeSignals, CockpitUiState } from "./cockpitTypes";
import { COCKPIT_INITIAL_OVERRIDE } from "./cockpitTypes";
import { saveCockpitSession } from "./cockpitSession";
import { getCockpitAdvancedFlags } from "./cockpitFeatureFlags";

export type UseCockpitRuntimeOptions = {
  realtimeStatus: string;
  realtimeDataFreshness?: string;
  realtimeLastEventAt?: string | null;
  urgentCount: number;
  delayedCount: number;
  activeDriverCount: number;
  inProgressMissionCount: number;
  driverSheetOpen: boolean;
  opsSheetOpen: boolean;
  filtersOpen?: boolean;
  layersOpen?: boolean;
  searchActive?: boolean;
  navigationActive?: boolean;
  selectedDriverId?: number | null;
  selectedMissionId?: number | null;
  unassignedCount?: number;
  criticalEtaCount?: number;
  missions?: CompanyDispatchMission[];
  zoomLatitudeDelta?: number | null;
};

const POLICY_FAILURE_DECAY_MS = 5 * 60_000;
const POLICY_FAILURE_RESET_MS = 10 * 60_000;
const MAX_RUNTIME_TIMESTAMPS = 200;

export function useCockpitRuntime(options: UseCockpitRuntimeOptions) {
  const bootMsRef = useRef(Date.now());
  const storeRef = useRef(createCockpitRuntimeStore());
  const interactionTimestampsRef = useRef<number[]>([]);
  const policyFailureAtRef = useRef<number[]>([]);
  const [override, setOverride] = useState<CockpitHumanOverride>(COCKPIT_INITIAL_OVERRIDE);
  const [driverSheetSnap, setDriverSheetSnap] = useState<"collapsed" | "medium" | "expanded">("collapsed");
  const [policyFailureCount, setPolicyFailureCount] = useState(0);
  const [sessionHydrated, setSessionHydrated] = useState(false);
  const [orchestrationTick, setOrchestrationTick] = useState(0);
  const [initialSelectedDriverId] = useState<number | null>(null);
  const fsmBootstrappedRef = useRef(false);

  const touchInteraction = useCallback(() => {
    const now = Date.now();
    interactionTimestampsRef.current.push(now);
    if (interactionTimestampsRef.current.length > MAX_RUNTIME_TIMESTAMPS) {
      interactionTimestampsRef.current.splice(
        0,
        interactionTimestampsRef.current.length - MAX_RUNTIME_TIMESTAMPS
      );
    }
    interactionTimestampsRef.current = interactionTimestampsRef.current.filter((t) => now - t < 60_000);
    setOrchestrationTick((n) => n + 1);
  }, []);

  const interactionBurstPerMinute = interactionTimestampsRef.current.length;

  useEffect(() => {
    const store = storeRef.current;
    void store.hydrateSession().then(() => {
      setSessionHydrated(true);
      setOrchestrationTick((n) => n + 1);
    });
  }, []);

  useEffect(() => {
    if (!sessionHydrated || fsmBootstrappedRef.current) return;
    fsmBootstrappedRef.current = true;
    const store = storeRef.current;
    store.bootstrapFsm({
      searchActive: options.searchActive ?? false,
      selectedDriverId:
        options.selectedDriverId ?? store.getSnapshot().initialSelectedDriverId,
      driverSheetOpen: options.driverSheetOpen,
      filtersOpen: options.filtersOpen ?? false,
      layersOpen: options.layersOpen ?? false,
      urgentCount: options.urgentCount,
    });
    setOrchestrationTick((n) => n + 1);
  }, [sessionHydrated, options.searchActive, options.selectedDriverId, options.driverSheetOpen, options.filtersOpen, options.layersOpen, options.urgentCount]);

  useEffect(() => {
    const timer = setInterval(() => {
      const now = Date.now();
      const kept = policyFailureAtRef.current.filter((t) => now - t < POLICY_FAILURE_RESET_MS);
      if (kept.length !== policyFailureAtRef.current.length) {
        policyFailureAtRef.current = kept;
        setPolicyFailureCount(kept.length);
      } else if (kept.length > 0) {
        const decayed = kept.filter((t) => now - t < POLICY_FAILURE_DECAY_MS);
        if (decayed.length < kept.length) {
          policyFailureAtRef.current = decayed;
          setPolicyFailureCount(decayed.length);
        }
      }
    }, POLICY_FAILURE_DECAY_MS);
    return () => clearInterval(timer);
  }, []);

  const signals: CockpitRuntimeSignals = useMemo(
    () => ({
      realtimeStatus: options.realtimeStatus,
      realtimeDataFreshness: options.realtimeDataFreshness,
      realtimeLastEventAt: options.realtimeLastEventAt,
      driverSheetOpen: options.driverSheetOpen,
      driverSheetSnap,
      opsSheetOpen: options.opsSheetOpen,
      filtersOpen: options.filtersOpen ?? false,
      layersOpen: options.layersOpen ?? false,
      searchActive: options.searchActive ?? false,
      navigationActive: options.navigationActive ?? false,
      selectedDriverId: options.selectedDriverId ?? null,
      selectedMissionId: options.selectedMissionId ?? null,
      urgentCount: options.urgentCount,
      delayedCount: options.delayedCount,
      activeDriverCount: options.activeDriverCount,
      inProgressMissionCount: options.inProgressMissionCount,
      policyFailureCount,
      interactionBurstPerMinute,
      healthScore: 0,
      stabilized: true,
    }),
    [
      options.realtimeStatus,
      options.realtimeDataFreshness,
      options.realtimeLastEventAt,
      options.driverSheetOpen,
      options.opsSheetOpen,
      options.filtersOpen,
      options.layersOpen,
      options.searchActive,
      options.navigationActive,
      options.selectedDriverId,
      options.selectedMissionId,
      options.urgentCount,
      options.delayedCount,
      options.activeDriverCount,
      options.inProgressMissionCount,
      driverSheetSnap,
      policyFailureCount,
      interactionBurstPerMinute,
    ]
  );

  const orchestration: CockpitOrchestrationDecision = useMemo(() => {
    // Dépendance volontaire : tick incrémenté après événements store pour forcer un recompute orchestration.
    void orchestrationTick;
    const store = storeRef.current;
    const frameEvents = store.flushFrame();
    const snap = store.getSnapshot();
    const decision = resolveCockpitOrchestration({
      ...signals,
      override,
      bootMs: bootMsRef.current,
      driverCount: options.activeDriverCount,
      unassignedCount: options.unassignedCount,
      criticalEtaCount: options.criticalEtaCount,
      missions: options.missions,
      zoomLatitudeDelta: options.zoomLatitudeDelta,
      manualRecenterOnly: override.manualRecenterOnly,
      fsmSnapshot: snap.fsm,
      stabilityState: snap.stability,
      frameEvents,
    });
    store.setStabilityState(decision.stabilityState);
    return decision;
  }, [signals, override, options.activeDriverCount, options.unassignedCount, options.criticalEtaCount, options.missions, options.zoomLatitudeDelta, orchestrationTick]);

  const uiState: CockpitUiState = orchestration.uiState;

  const dispatch = useCallback((event: CockpitFsmEvent) => {
    storeRef.current.dispatch(event);
    setOrchestrationTick((n) => n + 1);
  }, []);

  const enqueueFrameEvent = useCallback((event: CockpitEvent) => {
    storeRef.current.enqueueFrameEvent(event);
    setOrchestrationTick((n) => n + 1);
  }, []);

  const enqueueGpsUpdate = useCallback((source = "gps") => {
    storeRef.current.enqueueFrameEvent({
      type: "GPS_UPDATE",
      atMs: Date.now(),
      source,
      coalescable: true,
    });
  }, []);

  const setKillSwitch = useCallback((on: boolean) => {
    setOverride((o) => ({ ...o, killSwitch: on }));
  }, []);

  const setAutoFocusOff = useCallback((off: boolean) => {
    setOverride((o) => ({ ...o, autoFocusOff: off }));
  }, []);

  const reportPolicyFailure = useCallback(() => {
    const now = Date.now();
    policyFailureAtRef.current.push(now);
    if (policyFailureAtRef.current.length > MAX_RUNTIME_TIMESTAMPS) {
      policyFailureAtRef.current.splice(
        0,
        policyFailureAtRef.current.length - MAX_RUNTIME_TIMESTAMPS
      );
    }
    setPolicyFailureCount(policyFailureAtRef.current.length);
    setOrchestrationTick((n) => n + 1);
  }, []);

  useEffect(() => {
    void saveCockpitSession({
      selectedDriverId: options.selectedDriverId ?? null,
      selectedMissionId: options.selectedMissionId ?? null,
      driverSheetSnap,
      killSwitch: override.killSwitch,
      autoFocusOff: override.autoFocusOff,
      savedAt: Date.now(),
    });
  }, [
    options.selectedDriverId,
    options.selectedMissionId,
    driverSheetSnap,
    override.killSwitch,
    override.autoFocusOff,
  ]);

  const flags = getCockpitAdvancedFlags();

  return useMemo(
    () => ({
      uiState,
      orchestration,
      flags,
      driverSheetSnap,
      setDriverSheetSnap,
      override,
      setOverride,
      setKillSwitch,
      setAutoFocusOff,
      touchInteraction,
      reportPolicyFailure,
      dispatch,
      enqueueFrameEvent,
      enqueueGpsUpdate,
      sessionHydrated,
      initialSelectedDriverId,
      fsmState: orchestration.fsmState,
    }),
    [
      uiState,
      orchestration,
      flags,
      driverSheetSnap,
      override,
      touchInteraction,
      reportPolicyFailure,
      dispatch,
      enqueueFrameEvent,
      enqueueGpsUpdate,
      sessionHydrated,
      initialSelectedDriverId,
      setKillSwitch,
      setAutoFocusOff,
    ]
  );
}
