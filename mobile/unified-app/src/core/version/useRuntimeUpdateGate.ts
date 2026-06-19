import { useEffect, useRef, useState } from "react";
import * as Updates from "expo-updates";
import Constants from "expo-constants";
import {
  fetchAndReloadOtaUpdate,
  OTA_ASSET_LOAD_ERROR,
  resolveOtaApplyErrorMessage,
} from "./otaUpdateActions";

type UpdateGateState = {
  checking: boolean;
  applying: boolean;
  updateAvailable: boolean;
  requiresUpdate: boolean;
  recommendedUpdate: boolean;
  blockingScope: "driver" | "company" | "global";
  minimumSupportedVersion: string | null;
  recommendedVersion: string | null;
  killSwitch: boolean;
  error: string | null;
};

function parseVersion(value: string): number[] {
  return value
    .split(".")
    .map((part) => Number.parseInt(part, 10))
    .map((part) => (Number.isFinite(part) ? part : 0));
}

function compareVersions(left: string, right: string): number {
  const leftParts = parseVersion(left);
  const rightParts = parseVersion(right);
  const max = Math.max(leftParts.length, rightParts.length);
  for (let index = 0; index < max; index += 1) {
    const l = leftParts[index] ?? 0;
    const r = rightParts[index] ?? 0;
    if (l > r) return 1;
    if (l < r) return -1;
  }
  return 0;
}

function readPolicy() {
  const minimumSupportedVersion = process.env.EXPO_PUBLIC_MIN_SUPPORTED_VERSION ?? null;
  const recommendedVersion = process.env.EXPO_PUBLIC_RECOMMENDED_VERSION ?? null;
  const blockingScopeRaw = process.env.EXPO_PUBLIC_UPDATE_BLOCKING_SCOPE ?? "global";
  const blockingScope: "driver" | "company" | "global" =
    blockingScopeRaw === "driver" || blockingScopeRaw === "company" ? blockingScopeRaw : "global";
  const killSwitch = process.env.EXPO_PUBLIC_UPDATE_GATE_KILL_SWITCH === "1";
  return { minimumSupportedVersion, recommendedVersion, blockingScope, killSwitch };
}

export function useRuntimeUpdateGate() {
  const policy = readPolicy();
  const appVersion = Constants.expoConfig?.version ?? "0.0.0";

  const [state, setState] = useState<UpdateGateState>({
    checking: true,
    applying: false,
    updateAvailable: false,
    requiresUpdate: false,
    recommendedUpdate: false,
    blockingScope: policy.blockingScope,
    minimumSupportedVersion: policy.minimumSupportedVersion,
    recommendedVersion: policy.recommendedVersion,
    killSwitch: policy.killSwitch,
    error: null,
  });

  const applyingRef = useRef(false);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const result = await Updates.checkForUpdateAsync();
        if (!mounted) return;
        const requiresUpdate =
          !policy.killSwitch &&
          !!policy.minimumSupportedVersion &&
          compareVersions(appVersion, policy.minimumSupportedVersion) < 0;
        const recommendedUpdate =
          !!policy.recommendedVersion &&
          compareVersions(appVersion, policy.recommendedVersion) < 0;
        setState((prev) => ({
          ...prev,
          checking: false,
          updateAvailable: result.isAvailable,
          requiresUpdate,
          recommendedUpdate,
          blockingScope: policy.blockingScope,
          minimumSupportedVersion: policy.minimumSupportedVersion,
          recommendedVersion: policy.recommendedVersion,
          killSwitch: policy.killSwitch,
          error: null,
        }));
      } catch (error) {
        if (!mounted) return;
        setState((prev) => ({
          ...prev,
          checking: false,
          updateAvailable: false,
          requiresUpdate: false,
          recommendedUpdate: false,
          blockingScope: policy.blockingScope,
          minimumSupportedVersion: policy.minimumSupportedVersion,
          recommendedVersion: policy.recommendedVersion,
          killSwitch: policy.killSwitch,
          error: error instanceof Error ? error.message : "update_check_failed",
        }));
      }
    };
    void check();
    return () => {
      mounted = false;
    };
  }, [appVersion, policy.blockingScope, policy.killSwitch, policy.minimumSupportedVersion, policy.recommendedVersion]);

  const applyUpdate = async () => {
    if (applyingRef.current) {
      return;
    }
    applyingRef.current = true;
    setState((prev) => ({ ...prev, applying: true, error: null }));
    try {
      const result = await fetchAndReloadOtaUpdate();
      if (result === "not_new") {
        applyingRef.current = false;
        setState((prev) => ({
          ...prev,
          applying: false,
          updateAvailable: false,
        }));
        return;
      }
      if (result === "failed") {
        throw new Error("update_apply_failed");
      }
    } catch (error) {
      applyingRef.current = false;
      const message =
        error instanceof Error && error.message.includes("Failed to load all assets")
          ? OTA_ASSET_LOAD_ERROR
          : resolveOtaApplyErrorMessage(error);
      setState((prev) => ({ ...prev, applying: false, error: message }));
    }
  };

  return {
    ...state,
    applyUpdate,
  };
}
