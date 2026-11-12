import { useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import { useQuery } from "@tanstack/react-query";
import { LirieSheet } from "../../../design/foundations/LirieSheet";
import { lirieSpace } from "../../../design/foundations/tokens";
import { useNetInfo } from "@react-native-community/netinfo";
import { apiClient } from "../../../core/api/client";
import { readPublishedAppVariant } from "../../../core/api/p0RuntimeIdentity";
import { MissionSettlementEntry } from "./MissionSettlementEntry";
import { missionPanelState } from "./missionPanelState";
import {
  settlementCashPrepAllowed,
  settlementLedgerAccessAllowed,
  settlementMomentOpen,
  settlementNeedsHalt,
  settlementRecipeAccessAllowed,
  settlementRecipeCandidate,
} from "./settlementAccess";
import { DriverCashPrepHost } from "./DriverCashPrepHost";
import { DriverPatientPrepHost } from "./DriverPatientPrepHost";
import {
  PersistentSimulationPanel,
  type SimulationDossier,
} from "./PersistentSimulationPanel";
import { SettlementChannelNotice } from "./SettlementChannelNotice";

type RecipeTarget = {
  database?: unknown;
  writable_recipe?: unknown;
  driver_settlement_panel?: unknown;
  settlement_simulation_ledger?: unknown;
  settlement_production_prep?: unknown;
};

async function fetchRecipeTarget(): Promise<{
  database: string | null;
  writableRecipe: boolean;
  serverPanelEnabled: boolean;
  serverLedgerEnabled: boolean;
  serverPrepEnabled: boolean;
}> {
  const { data } = await apiClient.get<RecipeTarget>("/meta/recipe-target");
  return {
    database: typeof data?.database === "string" ? data.database : null,
    writableRecipe: data?.writable_recipe === true,
    serverPanelEnabled: data?.driver_settlement_panel === true,
    serverLedgerEnabled: data?.settlement_simulation_ledger === true,
    serverPrepEnabled: data?.settlement_production_prep === true,
  };
}

async function fetchMissionSettlement(missionId: number): Promise<unknown> {
  const { data } = await apiClient.get(`/driver/me/bookings/${missionId}/settlement-read`);
  return data;
}

async function ensureSimulationCase(missionId: number): Promise<SimulationDossier | null> {
  const path = `/driver/me/bookings/${missionId}/settlement-simulation-case`;
  try {
    const existing = await apiClient.get<SimulationDossier>(path);
    if (existing.data?.persistent_simulation === true) return existing.data;
  } catch {
    // Un dossier absent se crée ensuite. Toute autre erreur n'affiche rien.
  }
  try {
    const created = await apiClient.post<SimulationDossier>(path);
    return created.data?.persistent_simulation === true ? created.data : null;
  } catch {
    return null;
  }
}

export function useSettlementRecipeAccess(): boolean {
  const appVariant = readPublishedAppVariant() ?? process.env.EXPO_PUBLIC_APP_VARIANT;
  const candidate = settlementRecipeCandidate({
    dev: typeof __DEV__ !== "undefined" && __DEV__,
    appVariant,
  });
  const recipe = useQuery({
    queryKey: ["settlement-recipe-target"],
    queryFn: fetchRecipeTarget,
    enabled: candidate,
    retry: false,
    staleTime: 60_000,
  });
  if (!candidate || !recipe.data) return false;
  return settlementRecipeAccessAllowed({
    dev: true,
    appVariant,
    recipeDatabase: recipe.data.database,
    writableRecipe: recipe.data.writableRecipe,
    serverPanelEnabled: recipe.data.serverPanelEnabled,
  });
}

export function useSettlementCashPrepAccess(): boolean {
  const appVariant = readPublishedAppVariant() ?? process.env.EXPO_PUBLIC_APP_VARIANT;
  const candidate = settlementRecipeCandidate({
    dev: typeof __DEV__ !== "undefined" && __DEV__,
    appVariant,
  });
  const recipe = useQuery({
    queryKey: ["settlement-recipe-target"],
    queryFn: fetchRecipeTarget,
    enabled: candidate,
    retry: false,
    staleTime: 60_000,
  });
  if (!candidate || !recipe.data) return false;
  return settlementCashPrepAllowed({
    dev: true,
    appVariant,
    recipeDatabase: recipe.data.database,
    serverPrepEnabled: recipe.data.serverPrepEnabled,
  });
}

export function useSettlementLedgerAccess(): boolean {
  const appVariant = readPublishedAppVariant() ?? process.env.EXPO_PUBLIC_APP_VARIANT;
  const candidate = settlementRecipeCandidate({
    dev: typeof __DEV__ !== "undefined" && __DEV__,
    appVariant,
  });
  const recipe = useQuery({
    queryKey: ["settlement-recipe-target"],
    queryFn: fetchRecipeTarget,
    enabled: candidate,
    retry: false,
    staleTime: 60_000,
  });
  if (!candidate || !recipe.data) return false;
  return settlementLedgerAccessAllowed({
    dev: true,
    appVariant,
    recipeDatabase: recipe.data.database,
    serverLedgerEnabled: recipe.data.serverLedgerEnabled,
  });
}

/**
 * Panneau facultatif. La lecture ne part qu'à l'ouverture, et jamais pendant
 * la conduite : l'arrêt doit d'abord être confirmé. Le statut n'est pas modifié.
 */
export function MissionSettlementSlot({
  missionId,
  operationalStatus,
}: {
  missionId: number;
  operationalStatus: string | null;
}) {
  const recipeAllowed = useSettlementRecipeAccess();
  const ledgerAllowed = useSettlementLedgerAccess();
  const cashPrepAllowed = useSettlementCashPrepAccess();
  const allowed = recipeAllowed || ledgerAllowed || cashPrepAllowed;
  const net = useNetInfo();
  const offline = net.isConnected === false;
  const online = net.isConnected === true;
  const [haltConfirmed, setHaltConfirmed] = useState(false);
  const [opened, setOpened] = useState(false);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    setHaltConfirmed(false);
    setOpened(false);
  }, [missionId, operationalStatus]);

  const momentOpen = settlementMomentOpen(operationalStatus);
  const haltOk = !settlementNeedsHalt(operationalStatus) || haltConfirmed;
  const mayRead =
    recipeAllowed && momentOpen && opened && haltOk && online;
  const mayLoadCase = ledgerAllowed && momentOpen && opened && haltOk && online;
  const reading = useQuery({
    queryKey: ["driver-settlement-read", missionId, operationalStatus, nonce],
    queryFn: () => fetchMissionSettlement(missionId),
    enabled: mayRead,
    retry: false,
    staleTime: 0,
    gcTime: 0,
  });
  const dossier = useQuery({
    queryKey: ["settlement-simulation-case", missionId, nonce],
    queryFn: () => ensureSimulationCase(missionId),
    enabled: mayLoadCase,
    retry: false,
    staleTime: 0,
  });
  const fresh = mayRead && reading.isSuccess && !reading.isFetching;
  const ledgerMiss = ledgerAllowed && dossier.isFetched && !dossier.data;
  const state = missionPanelState({
    allowed,
    operationalStatus,
    haltConfirmed,
    opened: ledgerMiss ? false : opened,
    offline,
    loading: opened && !offline && !reading.isError && !fresh,
    failed: opened && reading.isError,
    payload: fresh ? reading.data : null,
  });
  if (!allowed || !momentOpen) return null;
  const showCash = cashPrepAllowed && opened && haltOk && online;
  const closeSettlement = () => setOpened(false);
  const cashHosts = showCash ? (
    <>
      <SettlementChannelNotice />
      <DriverCashPrepHost missionId={missionId} onBack={closeSettlement} />
      <DriverPatientPrepHost missionId={missionId} onBack={closeSettlement} />
    </>
  ) : null;
  if (dossier.data) {
    return (
      <View style={styles.zone}>
        {cashHosts}
        <PersistentSimulationPanel
          dossier={dossier.data}
          onAction={async (action, body) => {
            const { data } = await apiClient.post<SimulationDossier>(
              `/driver/me/bookings/${missionId}/settlement-simulation-case/transitions`,
              body
            );
            dossier.refetch();
            if (!data?.state) {
              throw new Error("transition");
            }
          }}
          onBack={closeSettlement}
        />
      </View>
    );
  }
  if (cashHosts) {
    return (
      <View style={styles.zone}>
        <LirieSheet visible title="Règlement du transport" onClose={closeSettlement}>
          {cashHosts}
        </LirieSheet>
      </View>
    );
  }
  return (
    <View style={styles.zone}>
      <MissionSettlementEntry
        state={state}
        onConfirmHalt={() => setHaltConfirmed(true)}
        onOpen={() => {
          setNonce((value) => value + 1);
          setOpened(true);
        }}
        onClose={closeSettlement}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  zone: {
    width: "100%",
    marginTop: lirieSpace.lg,
    marginBottom: lirieSpace.md,
  },
});
