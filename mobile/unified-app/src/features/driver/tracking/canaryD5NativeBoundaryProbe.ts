/**
 * Canary D5 — sondes frontières natives (QA panel / production-apk uniquement).
 *
 * Objectif : attribuer Unregister TaskService W1/W2 sans modifier B2 / transient / L1.
 * Logcat : préfixe `[D5-NATIVE]` + telemetry `tracking.lifecycle.canary_*`.
 */
import { AppState, type AppStateStatus } from "react-native";
import * as Location from "expo-location";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";

const TASK_NAME = "background-location-task";

function isCanaryProbeEnabled(): boolean {
  return (
    process.env.EXPO_PUBLIC_TRACKING_QA_PANEL === "1" ||
    process.env.EXPO_PUBLIC_TRACKING_QA_PANEL === "true"
  );
}

export type TaskRegTrigger =
  | "permission_start"
  | "permission_end"
  | "appstate"
  | "watch_restart"
  | "host_mount"
  | "host_unmount"
  | "native_start_entry"
  | "native_start_exit"
  | "native_stop_entry"
  | "native_stop_exit";

type ProbeContext = {
  caller?: string;
  reason?: string;
  generation?: number | string | null;
  missionId?: number | null;
  authority?: string | null;
  appState?: AppStateStatus | string | null;
  permission?: string | null;
  success?: boolean;
  error?: string | null;
  registeredBefore?: boolean | null;
  registeredAfter?: boolean | null;
};

function enabled(): boolean {
  return isCanaryProbeEnabled();
}

function logMark(tag: string, payload: Record<string, unknown>): void {
  console.warn(`[D5-NATIVE] ${tag}`, payload);
  emitDriverTelemetry(`tracking.lifecycle.canary_${tag.toLowerCase()}` as never, {
    source: "driver.canary.d5_native_probe",
    ...payload,
  });
}

async function readRegistered(): Promise<boolean | null> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const TaskManager = require("expo-task-manager") as {
      isTaskRegisteredAsync?: (name: string) => Promise<boolean>;
      isTaskDefined?: (name: string) => boolean;
    };
    if (typeof TaskManager?.isTaskRegisteredAsync !== "function") return null;
    return await TaskManager.isTaskRegisteredAsync(TASK_NAME);
  } catch {
    return null;
  }
}

async function readDefined(): Promise<boolean | null> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const TaskManager = require("expo-task-manager") as {
      isTaskDefined?: (name: string) => boolean;
    };
    if (typeof TaskManager?.isTaskDefined !== "function") return null;
    return Boolean(TaskManager.isTaskDefined(TASK_NAME));
  } catch {
    return null;
  }
}

export async function probeTaskRegState(
  trigger: TaskRegTrigger,
  ctx: ProbeContext = {}
): Promise<{ defined: boolean | null; registered: boolean | null }> {
  if (!enabled()) return { defined: null, registered: null };
  const [defined, registered] = await Promise.all([readDefined(), readRegistered()]);
  logMark("TASK_REG_STATE", {
    trigger,
    defined,
    registered,
    app_state: ctx.appState ?? AppState.currentState,
    mission_id: ctx.missionId ?? null,
    generation: ctx.generation ?? null,
    reason: ctx.reason ?? null,
    caller: ctx.caller ?? null,
    ts: Date.now(),
  });
  return { defined, registered };
}

export async function markNativeStopEntry(ctx: ProbeContext): Promise<boolean | null> {
  if (!enabled()) return null;
  const registered = await readRegistered();
  logMark("NATIVE_STOP_ENTRY", {
    timestamp: Date.now(),
    caller: ctx.caller ?? null,
    reason: ctx.reason ?? null,
    generation: ctx.generation ?? null,
    mission_id: ctx.missionId ?? null,
    authority: ctx.authority ?? null,
    app_state: ctx.appState ?? AppState.currentState,
    isTaskRegisteredAsync_before: registered,
  });
  await probeTaskRegState("native_stop_entry", { ...ctx, registeredBefore: registered });
  return registered;
}

export async function markNativeStopExit(ctx: ProbeContext): Promise<void> {
  if (!enabled()) return;
  const registered = await readRegistered();
  logMark("NATIVE_STOP_EXIT", {
    timestamp: Date.now(),
    caller: ctx.caller ?? null,
    reason: ctx.reason ?? null,
    generation: ctx.generation ?? null,
    mission_id: ctx.missionId ?? null,
    authority: ctx.authority ?? null,
    app_state: ctx.appState ?? AppState.currentState,
    success: ctx.success ?? null,
    error: ctx.error ?? null,
    isTaskRegisteredAsync_after: registered,
  });
  await probeTaskRegState("native_stop_exit", { ...ctx, registeredAfter: registered });
}

export async function markNativeStartEntry(ctx: ProbeContext): Promise<boolean | null> {
  if (!enabled()) return null;
  const registered = await readRegistered();
  logMark("NATIVE_START_ENTRY", {
    timestamp: Date.now(),
    caller: ctx.caller ?? null,
    reason: ctx.reason ?? null,
    generation: ctx.generation ?? null,
    mission_id: ctx.missionId ?? null,
    app_state: ctx.appState ?? AppState.currentState,
    isTaskRegisteredAsync_before: registered,
  });
  await probeTaskRegState("native_start_entry", { ...ctx, registeredBefore: registered });
  return registered;
}

export async function markNativeStartExit(ctx: ProbeContext): Promise<void> {
  if (!enabled()) return;
  const registered = await readRegistered();
  logMark("NATIVE_START_EXIT", {
    timestamp: Date.now(),
    caller: ctx.caller ?? null,
    reason: ctx.reason ?? null,
    generation: ctx.generation ?? null,
    mission_id: ctx.missionId ?? null,
    app_state: ctx.appState ?? AppState.currentState,
    success: ctx.success ?? null,
    error: ctx.error ?? null,
    isTaskRegisteredAsync_after: registered,
  });
  await probeTaskRegState("native_start_exit", { ...ctx, registeredAfter: registered });
}

export async function markPermissionRequestStart(ctx: ProbeContext): Promise<void> {
  if (!enabled()) return;
  logMark("PERMISSION_REQUEST_START", {
    timestamp: Date.now(),
    permission: ctx.permission ?? "foreground_location",
    app_state: ctx.appState ?? AppState.currentState,
    mission_id: ctx.missionId ?? null,
    generation: ctx.generation ?? null,
    caller: ctx.caller ?? null,
    reason: ctx.reason ?? null,
  });
  await probeTaskRegState("permission_start", ctx);
}

export async function markPermissionRequestResult(ctx: ProbeContext): Promise<void> {
  if (!enabled()) return;
  logMark("PERMISSION_REQUEST_RESULT", {
    timestamp: Date.now(),
    permission: ctx.permission ?? "foreground_location",
    app_state: ctx.appState ?? AppState.currentState,
    mission_id: ctx.missionId ?? null,
    generation: ctx.generation ?? null,
    caller: ctx.caller ?? null,
    reason: ctx.reason ?? null,
    success: ctx.success ?? null,
    error: ctx.error ?? null,
  });
  await probeTaskRegState("permission_end", ctx);
}

/**
 * Wrap `Location.requestForegroundPermissionsAsync` avec bornes canary.
 */
export async function requestForegroundPermissionsWithCanaryProbe(opts: {
  caller: string;
  reason: string;
  missionId?: number | null;
  generation?: number | string | null;
}): Promise<Location.LocationPermissionResponse | null> {
  await markPermissionRequestStart({
    caller: opts.caller,
    reason: opts.reason,
    missionId: opts.missionId,
    generation: opts.generation,
    permission: "foreground_location",
  });
  try {
    const result = await Location.requestForegroundPermissionsAsync();
    await markPermissionRequestResult({
      caller: opts.caller,
      reason: opts.reason,
      missionId: opts.missionId,
      generation: opts.generation,
      permission: "foreground_location",
      success: result?.status === "granted",
    });
    return result;
  } catch (error) {
    await markPermissionRequestResult({
      caller: opts.caller,
      reason: opts.reason,
      missionId: opts.missionId,
      generation: opts.generation,
      permission: "foreground_location",
      success: false,
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  }
}

export function installCanaryD5NativeBoundaryProbes(opts: {
  getMissionId: () => number | null;
  getGeneration: () => number | string | null;
}): () => void {
  if (!enabled()) {
    return () => undefined;
  }

  void probeTaskRegState("host_mount", {
    missionId: opts.getMissionId(),
    generation: opts.getGeneration(),
    caller: "DriverTrackingHost",
  });

  const sub = AppState.addEventListener("change", (next) => {
    void probeTaskRegState("appstate", {
      appState: next,
      missionId: opts.getMissionId(),
      generation: opts.getGeneration(),
      caller: "AppState.change",
      reason: `appstate_${next}`,
    });
  });

  return () => {
    void probeTaskRegState("host_unmount", {
      missionId: opts.getMissionId(),
      generation: opts.getGeneration(),
      caller: "DriverTrackingHost",
    });
    sub.remove();
  };
}
