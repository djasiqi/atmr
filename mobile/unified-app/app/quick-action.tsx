import { useCallback, useEffect, useMemo, useState } from "react";
import { Text, View } from "react-native";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import {
  quickAcceptDriverMission,
  quickCompleteDriverMission,
  quickRejectDriverMission,
  quickStartDriverMission,
} from "../src/features/driver/api";
import { Button, Card, Loader } from "../src/components/ui";

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
    <View style={{ flex: 1, justifyContent: "center", padding: 24 }}>
      <Card>
        <Text style={{ fontSize: 20, fontWeight: "700" }}>Quick action mission</Text>
        <Text>Mission: {missionId ?? "N/A"}</Text>
        <Text>Action demandee: {params.action ?? "manual"}</Text>
        {busy ? <Loader /> : null}
        {message ? <Text>{message}</Text> : null}
        <Button label="Accepter" variant="primary" onPress={() => void executeQuickAction("accept")} />
        <Button label="Refuser" onPress={() => void executeQuickAction("reject")} />
        <Button label="Demarrer" onPress={() => void executeQuickAction("start")} />
        <Button label="Terminer" onPress={() => void executeQuickAction("complete")} />
        <Button label="Retour missions" onPress={() => router.replace("/(app)/(driver)/missions")} />
      </Card>
    </View>
  );
}

