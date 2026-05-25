import { ScrollView, StyleSheet, View, Pressable, Linking } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../../../src/core/guards";
import { Screen, AppText, useAppViewport } from "../../../../../src/design/responsive";
import { D } from "../../../../../src/features/driver/theme/driverDashboardTheme";
import {
  useDriverCompanyId,
  useMessageHubThreads,
  useThreadMessages,
} from "../../../../../src/features/driver/messages/hooks";
import { SystemMessageRow } from "../../../../../src/features/driver/messages/components/SystemMessageRow";

export default function MessageThreadDetailsScreen() {
  const { threadId: threadIdParam } = useLocalSearchParams<{ threadId: string }>();
  const threadId = typeof threadIdParam === "string" ? threadIdParam : "dispatch";
  const { horizontalPadding } = useAppViewport();
  const companyId = useDriverCompanyId();
  const threadsQuery = useMessageHubThreads(companyId);
  const thread = threadsQuery.data?.threads.find((t) => t.thread_id === threadId);
  const messagesQuery = useThreadMessages(companyId, threadId, thread?.conversation_id);
  const router = useRouter();

  const systemEvents = (messagesQuery.data ?? []).filter((m) => m.message_type === "system");
  const callsPlaceholder = [
    { label: "Patient", phone: null },
    { label: "Dispatch", phone: null },
    { label: "Établissement", phone: null },
  ];

  return (
    <DriverContextGuard>
      <PermissionGuard permission="chat:read">
        <Screen scroll backgroundColor={D.pageBg}>
          <ScrollView contentContainerStyle={{ paddingHorizontal: horizontalPadding, paddingBottom: 32, gap: 16 }}>
            <View style={styles.card}>
              <AppText variant="sectionTitle">Mission</AppText>
              <AppText variant="body">{thread?.title ?? "—"}</AppText>
              <AppText variant="bodyMuted">{thread?.subtitle}</AppText>
              <AppText variant="caption">Statut · {thread?.status ?? "—"}</AppText>
              <AppText variant="caption">Départ · {thread?.pickup_location ?? "—"}</AppText>
              <AppText variant="caption">Arrivée · {thread?.dropoff_location ?? "—"}</AppText>
              {thread?.booking_id ? (
                <Pressable
                  onPress={() => router.push(`/(app)/(driver)/missions/${thread.booking_id}`)}
                  style={styles.cta}
                >
                  <AppText variant="label" style={styles.ctaText}>
                    Voir mission complète
                  </AppText>
                </Pressable>
              ) : null}
            </View>

            <View style={styles.card}>
              <AppText variant="sectionTitle">Patient / Interlocuteur</AppText>
              <AppText variant="body">{thread?.title ?? "Dispatch"}</AppText>
              <AppText variant="bodyMuted">Canal exploitation · disponible 24h/7j</AppText>
            </View>

            <View style={styles.card}>
              <AppText variant="sectionTitle">Historique</AppText>
              {systemEvents.length === 0 ? (
                <AppText variant="bodyMuted">Aucun événement système enregistré.</AppText>
              ) : (
                systemEvents.map((ev) => (
                  <SystemMessageRow
                    key={String(ev.id)}
                    content={ev.content}
                    timestamp={ev.timestamp}
                    senderName={ev.sender_name}
                  />
                ))
              )}
              {(messagesQuery.data ?? [])
                .filter((m) => m.message_type !== "system")
                .slice(-8)
                .map((m) => (
                  <AppText key={String(m.id)} variant="caption" style={styles.historyLine}>
                    {new Date(m.timestamp).toLocaleTimeString("fr-FR", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}{" "}
                    · {m.content.slice(0, 80)}
                  </AppText>
                ))}
            </View>

            <View style={styles.card}>
              <AppText variant="sectionTitle">Appels</AppText>
              {callsPlaceholder.map((row) => (
                <View key={row.label} style={styles.callRow}>
                  <AppText variant="body">{row.label}</AppText>
                  <Pressable
                    onPress={() => {
                      if (row.phone) void Linking.openURL(`tel:${row.phone}`);
                    }}
                    disabled={!row.phone}
                  >
                    <AppText variant="caption" style={row.phone ? styles.link : styles.muted}>
                      {row.phone ?? "Numéro non renseigné"}
                    </AppText>
                  </Pressable>
                </View>
              ))}
            </View>
          </ScrollView>
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 14,
    gap: 6,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#e5e7eb",
  },
  cta: {
    marginTop: 8,
    backgroundColor: "#0A8F7A",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
  },
  ctaText: { color: "#fff" },
  historyLine: { color: "#4b5563", marginTop: 4 },
  callRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#f3f4f6",
  },
  link: { color: "#0A8F7A", fontWeight: "600" },
  muted: { color: "#9ca3af" },
});
