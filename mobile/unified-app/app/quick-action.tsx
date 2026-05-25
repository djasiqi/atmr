import { useCallback, useEffect, useMemo, useState } from "react";
import { StyleSheet, Text } from "react-native";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import {
  quickAcceptDriverMission,
  quickCompleteDriverMission,
  quickRejectDriverMission,
  quickStartDriverMission,
} from "../src/features/driver/api";
import {
  AppButton,
  AppCard,
  AppSpinner,
  brandSurfaceSoft,
  ResponsiveContainer,
  Screen,
} from "../src/design/responsive";
import { FONT_SIZE } from "../src/design/responsive/typographyTokens";

type QuickAction = "accept" | "reject" | "start" | "complete";

export default function QuickActionScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ missionId?: string; action?: string }>();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const missionId = useMemo(() => {
    const raw = params.missionId;
    if (!raw) return null;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : null;
  }, [params.missionId]);
  const requestedAction = useMemo(() => {
    const action = String(params.action ?? "").toLowerCase();
    return action === "accept" || action === "reject" || action === "start" || action === "complete"
      ? (action as QuickAction)
      : null;
  }, [params.action]);
  const invalidRequest = !missionId || !requestedAction;

  const executeQuickAction = useCallback(async (action: QuickAction) => {
    if (!missionId) {
      setMessage("Mission invalide.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      if (action === "accept") await quickAcceptDriverMission(missionId);
      if (action === "reject") await quickRejectDriverMission(missionId);
      if (action === "start") await quickStartDriverMission(missionId);
      if (action === "complete") await quickCompleteDriverMission(missionId);
      setMessage(`Action ${action} envoyee pour mission #${missionId}.`);
      router.replace({
        pathname: "/(app)/(driver)/missions/[missionId]",
        params: { missionId: String(missionId) },
      });
    } catch (error) {
      const status = Number((error as { status?: number } | undefined)?.status ?? 0);
      if (status === 401 || status === 403) {
        router.replace({
          pathname: "/(public)/fallback/auth-required",
          params: {
            next: `/quick-action?missionId=${missionId}&action=${action}`,
          },
        } as any);
        return;
      }
      setMessage(error instanceof Error ? error.message : "Erreur pendant l'action rapide.");
    } finally {
      setBusy(false);
    }
  }, [missionId, router]);

  useEffect(() => {
    if (!requestedAction || !missionId || busy || message) return;
    void executeQuickAction(requestedAction);
  }, [busy, executeQuickAction, message, missionId, requestedAction]);

  if (invalidRequest) {
    return <Redirect href={"/(public)/fallback/invalid-link?reason=quick_action_invalid" as any} />;
  }

  return (
    <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <AppCard variant="surface">
          <Text style={styles.title}>Action rapide — mission</Text>
          <Text style={styles.line}>Mission : {missionId ?? "N/A"}</Text>
          <Text style={styles.line}>Action demandée : {params.action ?? "manuel"}</Text>
          {busy ? <AppSpinner size="small" /> : null}
          {message ? <Text style={styles.message}>{message}</Text> : null}
          <AppButton title="Accepter" variant="primary" onPress={() => void executeQuickAction("accept")} />
          <AppButton title="Refuser" variant="secondary" onPress={() => void executeQuickAction("reject")} />
          <AppButton title="Démarrer" variant="secondary" onPress={() => void executeQuickAction("start")} />
          <AppButton title="Terminer" variant="secondary" onPress={() => void executeQuickAction("complete")} />
          <AppButton title="Retour missions" variant="secondary" onPress={() => router.replace("/(app)/(driver)/missions")} />
        </AppCard>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 24,
  },
  title: {
    fontSize: FONT_SIZE.px20,
    fontWeight: "700",
    color: "#163A34",
    marginBottom: 8,
  },
  line: {
    fontSize: FONT_SIZE.px15,
    color: "#475569",
    marginBottom: 4,
  },
  message: {
    fontSize: FONT_SIZE.px14,
    color: "#334155",
    marginBottom: 8,
  },
});
