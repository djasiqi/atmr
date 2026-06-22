import { useCallback, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Modal,
  Pressable,
  StyleSheet,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useFocusEffect, useRouter } from "expo-router";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";
import {
  useCompanyInboxQuery,
  useCompanyInboxReadAllMutation,
  useCompanyInboxReadMutation,
} from "../hooks";
import type { CompanyInboxNotification } from "../api/companyInboxApi";
import { resolveCompanyInboxNavigation } from "../utils/companyNotificationNavigation";
import { E } from "../theme/enterpriseOpsTheme";
import { AppButton } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

dayjs.extend(relativeTime);
dayjs.locale("fr");

function formatWhen(iso: string): string {
  const d = dayjs(iso);
  return d.isValid() ? d.fromNow() : iso;
}

/**
 * Cloche + liste des notifications entreprise (`/api/v1/companies/notifications`).
 * Le comptage non lues est indépendant de la pastille « courses affichées » de l’en-tête.
 */
export function CompanyInboxButton() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const { data, isLoading, isError, refetch } = useCompanyInboxQuery();
  const readOne = useCompanyInboxReadMutation();
  const readAll = useCompanyInboxReadAllMutation();

  useFocusEffect(
    useCallback(() => {
      void refetch();
    }, [refetch])
  );

  const unread = data?.unread_count ?? 0;
  const items = data?.notifications ?? [];

  const onPressItem = (n: CompanyInboxNotification) => {
    if (!n.is_read) {
      readOne.mutate(n.id, {
        onError: () => undefined,
      });
    }
    const target = resolveCompanyInboxNavigation(n);
    if (target) {
      setOpen(false);
      router.push({
        pathname: target.pathname,
        params: target.params,
      });
    }
  };

  return (
    <>
      <Pressable
        onPress={() => setOpen(true)}
        style={({ pressed }) => [s.bellWrap, pressed && s.pressed]}
        accessibilityRole="button"
        accessibilityLabel="Ouvrir les notifications"
        hitSlop={12}
        testID="company-inbox-bell"
      >
        <Ionicons name="notifications-outline" size={22} color={E.BRAND} />
        {unread > 0 ? (
          <View style={s.badge} accessibilityLabel={`${unread} non lues`}>
            <AppText variant="caption" style={s.badgeText}>
              {unread > 99 ? "99+" : String(unread)}
            </AppText>
          </View>
        ) : null}
      </Pressable>

      <Modal visible={open} transparent animationType="slide" onRequestClose={() => setOpen(false)}>
        <View style={s.modalBackdrop}>
          <Pressable style={s.modalBackdropTap} onPress={() => setOpen(false)} />
          <View style={s.modalCard}>
            <View style={s.modalHeader}>
              <AppText variant="sectionTitle" style={s.modalTitle}>
                Notifications
              </AppText>
              <View style={s.headerActions}>
                {unread > 0 ? (
                  <AppButton
                    title={readAll.isPending ? "…" : "Tout marquer lu"}
                    onPress={() => readAll.mutate()}
                    disabled={readAll.isPending}
                    variant="secondary"
                  />
                ) : null}
                <Pressable
                  onPress={() => setOpen(false)}
                  accessibilityLabel="Fermer"
                  hitSlop={10}
                  style={s.closeBtn}
                >
                  <Ionicons name="close" size={24} color={E.TEXT_SEC} />
                </Pressable>
              </View>
            </View>
            {isLoading ? (
              <ActivityIndicator color={E.BRAND} style={{ padding: 24 }} />
            ) : items.length === 0 ? (
              <AppText variant="bodyMuted" style={s.empty}>
                Aucune notification récente.
              </AppText>
            ) : (
              <FlatList
                data={items}
                keyExtractor={(n) => String(n.id)}
                contentContainerStyle={s.listContent}
                style={s.list}
                renderItem={({ item: n }) => (
                  <Pressable
                    onPress={() => onPressItem(n)}
                    style={({ pressed }) => [s.row, !n.is_read && s.rowUnread, pressed && s.pressed]}
                  >
                    <View style={s.rowText}>
                      <AppText variant="label" style={s.title} numberOfLines={2}>
                        {n.title}
                      </AppText>
                      <AppText variant="bodyMuted" style={s.message} numberOfLines={3}>
                        {n.message}
                      </AppText>
                      <AppText variant="caption" style={s.when}>
                        {formatWhen(n.created_at)}
                      </AppText>
                    </View>
                    {!n.is_read ? <View style={s.unreadPill} /> : null}
                  </Pressable>
                )}
              />
            )}
            {isError ? (
              <AppText variant="error" style={s.error}>
                Impossible de charger la boîte. Vérifiez le rôle « entreprise ».
              </AppText>
            ) : null}
          </View>
        </View>
      </Modal>
    </>
  );
}

const s = StyleSheet.create({
  bellWrap: { position: "relative", padding: 2, alignItems: "center", justifyContent: "center" },
  pressed: { opacity: 0.8 },
  badge: {
    position: "absolute",
    top: -2,
    right: -2,
    minWidth: 18,
    paddingHorizontal: 5,
    paddingVertical: 1,
    borderRadius: 10,
    backgroundColor: E.DANGER,
    alignItems: "center",
    justifyContent: "center",
  },
  badgeText: { fontSize: FONT_SIZE.px10, fontWeight: "800" as const, color: E.CARD, lineHeight: 12 },
  modalBackdrop: { flex: 1, justifyContent: "flex-end" },
  modalBackdropTap: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(0,0,0,0.4)" },
  modalCard: {
    backgroundColor: E.CARD,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    paddingTop: 12,
    paddingBottom: 24,
    maxHeight: "85%",
  },
  modalHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  modalTitle: { color: E.TEXT },
  headerActions: { flexDirection: "row", alignItems: "center" },
  closeBtn: { padding: 4, marginLeft: 4 },
  list: { maxHeight: 400 },
  listContent: { paddingHorizontal: 12, paddingBottom: 8 },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: E.BORDER,
    marginBottom: 8,
  },
  rowUnread: { backgroundColor: "rgba(0, 121, 107, 0.06)" },
  rowText: { flex: 1, minWidth: 0 },
  title: { marginBottom: 4, color: E.TEXT, fontWeight: "700" as const },
  message: { lineHeight: 18 },
  when: { color: E.BRAND, marginTop: 4, fontWeight: "600" as const },
  unreadPill: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: E.BRAND,
    marginTop: 6,
    marginLeft: 6,
  },
  empty: { padding: 20, textAlign: "center" },
  error: { paddingHorizontal: 16 },
});
