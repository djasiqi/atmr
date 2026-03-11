/**
 * Network Debug Card — dev-only
 * Mode forcé diagnostic (QA) : forcer normal / dégradé / ultra-éco.
 * Plan migration 2G/3G — Phase 1.
 */
import React, { useState, useEffect, useCallback } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import {
  getMode,
  setForcedMode,
  subscribeToMode,
  type NetworkMode,
} from "@/services/connectivityPolicy";

const MODES: NetworkMode[] = ["normal", "degraded", "ultra_eco"];

export function NetworkDebugCard() {
  const [mode, setMode] = useState<NetworkMode>(getMode());

  useEffect(() => {
    const unsub = subscribeToMode(setMode);
    return unsub;
  }, []);

  const handleForce = useCallback(async (m: NetworkMode) => {
    await setForcedMode(m);
    setMode(getMode());
  }, []);

  const handleReset = useCallback(async () => {
    await setForcedMode(null);
    setMode(getMode());
  }, []);

  if (!__DEV__) return null;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>📡 Mode réseau (debug)</Text>
      <Text style={styles.current}>Actuel : {mode}</Text>
      <View style={styles.buttons}>
        {MODES.map((m) => (
          <TouchableOpacity
            key={m}
            style={[styles.btn, mode === m && styles.btnActive]}
            onPress={() => handleForce(m)}
          >
            <Text style={styles.btnText}>{m}</Text>
          </TouchableOpacity>
        ))}
      </View>
      <TouchableOpacity style={styles.resetBtn} onPress={handleReset}>
        <Text style={styles.resetText}>Réinitialiser</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 16,
    marginTop: 12,
    padding: 12,
    backgroundColor: "#1E293B",
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: "#3B82F6",
  },
  title: {
    fontSize: 14,
    fontWeight: "600",
    color: "#F1F5F9",
    marginBottom: 6,
  },
  current: {
    fontSize: 12,
    color: "#94A3B8",
    marginBottom: 8,
  },
  buttons: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 8,
  },
  btn: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    backgroundColor: "#334155",
    borderRadius: 6,
  },
  btnActive: {
    backgroundColor: "#3B82F6",
  },
  btnText: {
    fontSize: 12,
    color: "#F1F5F9",
  },
  resetBtn: {
    paddingVertical: 6,
    alignSelf: "flex-start",
  },
  resetText: {
    fontSize: 12,
    color: "#94A3B8",
  },
});
