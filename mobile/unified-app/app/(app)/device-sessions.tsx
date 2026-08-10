import { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  StyleSheet,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import {
  listDeviceSessions,
  revokeDeviceSession,
  revokeOtherDeviceSessions,
  type DeviceSessionInfo,
} from "../../src/core/api/client";
import {
  AppButton,
  AppText,
  brandPrimary,
  brandSurfaceSoft,
  ResponsiveContainer,
  Screen,
  useResponsiveTokens,
} from "../../src/design/responsive";

type Feedback = { text: string; tone: "success" | "error" } | null;

function formatRelativeLastSeen(iso?: string | null): string {
  if (!iso) return "Activité inconnue";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Activité inconnue";
  const diffMs = Date.now() - d.getTime();
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return "Vu à l'instant";
  if (mins < 60) return `Vu il y a ${mins} min`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `Vu il y a ${hours} h`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `Vu il y a ${days} j`;
  return `Vu le ${d.toLocaleDateString("fr-FR")}`;
}

function platformLabel(platform?: string | null): string {
  const p = (platform || "").toLowerCase();
  if (p === "ios") return "iOS";
  if (p === "android") return "Android";
  if (p === "web") return "Web";
  return platform?.trim() || "Plateforme inconnue";
}

export default function DeviceSessionsScreen() {
  const router = useRouter();
  const t = useResponsiveTokens();
  const [sessions, setSessions] = useState<DeviceSessionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [busySessionId, setBusySessionId] = useState<string | null>(null);
  const [revokingOthers, setRevokingOthers] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);

  const load = useCallback(async (mode: "initial" | "refresh" = "initial") => {
    if (mode === "refresh") setRefreshing(true);
    else setLoading(true);
    setFeedback(null);
    try {
      const data = await listDeviceSessions();
      setSessions(data.sessions);
    } catch (error) {
      const message =
        error && typeof error === "object" && typeof (error as { message?: unknown }).message === "string"
          ? (error as { message: string }).message
          : "Impossible de charger les appareils connectés.";
      setFeedback({ text: message, tone: "error" });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void load("initial");
  }, [load]);

  const onRevoke = async (sessionId: string) => {
    if (busySessionId || revokingOthers) return;
    setBusySessionId(sessionId);
    setFeedback(null);
    try {
      await revokeDeviceSession(sessionId);
      setFeedback({ text: "Appareil déconnecté.", tone: "success" });
      await load("refresh");
    } catch (error) {
      const message =
        error && typeof error === "object" && typeof (error as { message?: unknown }).message === "string"
          ? (error as { message: string }).message
          : "Impossible de déconnecter cet appareil.";
      setFeedback({ text: message, tone: "error" });
    } finally {
      setBusySessionId(null);
    }
  };

  const onRevokeOthers = async () => {
    if (busySessionId || revokingOthers) return;
    setRevokingOthers(true);
    setFeedback(null);
    try {
      const result = await revokeOtherDeviceSessions();
      const count = result.revoked_sessions;
      setFeedback({
        text:
          count > 0
            ? `${count} autre${count > 1 ? "s" : ""} appareil${count > 1 ? "s" : ""} déconnecté${count > 1 ? "s" : ""}.`
            : "Aucun autre appareil à déconnecter.",
        tone: "success",
      });
      await load("refresh");
    } catch (error) {
      const message =
        error && typeof error === "object" && typeof (error as { message?: unknown }).message === "string"
          ? (error as { message: string }).message
          : "Impossible de déconnecter les autres appareils.";
      setFeedback({ text: message, tone: "error" });
    } finally {
      setRevokingOthers(false);
    }
  };

  const otherSessionsCount = sessions.filter((s) => !s.is_current).length;

  return (
    <Screen
      scroll
      backgroundColor={brandSurfaceSoft}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => void load("refresh")}
          tintColor={brandPrimary}
        />
      }
    >
      <ResponsiveContainer>
        <View style={[styles.header, { paddingTop: t.space.sm }]}>
          <Pressable
            onPress={() => router.back()}
            accessibilityRole="button"
            accessibilityLabel="Retour"
            style={styles.backButton}
            hitSlop={12}
          >
            <Ionicons name="chevron-back" size={22} color="#163A34" />
          </Pressable>
          <AppText variant="title" style={styles.title}>
            Appareils connectés
          </AppText>
        </View>

        <AppText variant="caption" style={styles.subtitle}>
          Gérez les appareils ayant accès à votre compte. Vous pouvez déconnecter un
          appareil individuellement ou tous les autres d’un coup.
        </AppText>

        {feedback ? (
          <View
            style={feedback.tone === "error" ? styles.feedbackError : styles.feedbackSuccess}
          >
            <AppText
              variant={feedback.tone === "error" ? "error" : "body"}
              style={feedback.tone === "success" ? styles.feedbackSuccessText : undefined}
            >
              {feedback.text}
            </AppText>
          </View>
        ) : null}

        {loading ? (
          <View style={styles.loadingWrap}>
            <ActivityIndicator color={brandPrimary} />
            <AppText variant="caption" style={styles.muted}>
              Chargement des appareils…
            </AppText>
          </View>
        ) : sessions.length === 0 ? (
          <View style={styles.emptyCard}>
            <AppText variant="body" style={styles.emptyText}>
              Aucun appareil connecté pour le moment.
            </AppText>
          </View>
        ) : (
          <View style={styles.list}>
            {sessions.map((session) => {
              const sid = session.session_id;
              const busy = busySessionId === sid;
              return (
                <View key={sid} style={styles.card}>
                  <View style={styles.cardHeader}>
                    <AppText variant="body" style={styles.deviceName}>
                      {session.device_name?.trim() || "Appareil"}
                    </AppText>
                    {session.is_current ? (
                      <View style={styles.badge}>
                        <AppText variant="caption" style={styles.badgeText}>
                          Cet appareil
                        </AppText>
                      </View>
                    ) : null}
                  </View>
                  <AppText variant="caption" style={styles.meta}>
                    {platformLabel(session.last_platform)}
                    {session.last_app_version ? ` · v${session.last_app_version}` : ""}
                  </AppText>
                  <AppText variant="caption" style={styles.meta}>
                    {formatRelativeLastSeen(session.last_seen_at)}
                  </AppText>
                  {!session.is_current ? (
                    <AppButton
                      title={busy ? "Déconnexion…" : "Déconnecter"}
                      variant="secondary"
                      disabled={Boolean(busySessionId) || revokingOthers}
                      onPress={() => void onRevoke(sid)}
                      style={styles.rowButton}
                    />
                  ) : null}
                </View>
              );
            })}
          </View>
        )}

        {otherSessionsCount > 0 ? (
          <AppButton
            title={
              revokingOthers
                ? "Déconnexion…"
                : "Déconnecter tous les autres"
            }
            variant="secondary"
            disabled={Boolean(busySessionId) || revokingOthers || loading}
            onPress={() => void onRevokeOthers()}
            style={styles.revokeOthers}
          />
        ) : null}
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  backButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    color: "#163A34",
    fontWeight: "700",
    flex: 1,
  },
  subtitle: {
    color: "#5F7369",
    marginBottom: 16,
    lineHeight: 18,
  },
  loadingWrap: {
    alignItems: "center",
    gap: 10,
    paddingVertical: 40,
  },
  muted: { color: "#5F7369" },
  emptyCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 16,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(145, 165, 157, 0.45)",
  },
  emptyText: { color: "#5F7369" },
  list: { gap: 12 },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 16,
    gap: 6,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(145, 165, 157, 0.45)",
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  deviceName: {
    color: "#163A34",
    fontWeight: "600",
    flex: 1,
  },
  badge: {
    backgroundColor: "rgba(10, 143, 122, 0.12)",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  badgeText: {
    color: brandPrimary,
    fontWeight: "600",
  },
  meta: { color: "#5F7369" },
  rowButton: { marginTop: 8 },
  revokeOthers: { marginTop: 20, marginBottom: 24 },
  feedbackError: {
    backgroundColor: "rgba(185, 28, 28, 0.08)",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  feedbackSuccess: {
    backgroundColor: "rgba(10, 143, 122, 0.1)",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  feedbackSuccessText: { color: "#163A34" },
});
