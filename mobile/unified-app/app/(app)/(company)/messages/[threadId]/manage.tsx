import { CompanyContextGuard, PermissionGuard } from "../../../../../src/core/guards";
import { Screen, AppText, useAppViewport } from "../../../../../src/design/responsive";
import { M } from "../../../../../src/features/messaging/messagingTheme";
import { ChannelManageScreen } from "../../../../../src/features/company/messages/components/ChannelManageScreen";
import { resolveConversationId } from "../../../../../src/features/driver/messages/api";
import { useCompanyNumericId } from "../../../../../src/features/company/messages/companyId";
import { useEffect, useMemo, useState } from "react";
import { ActivityIndicator, Pressable, View } from "react-native";
import { useLocalSearchParams } from "expo-router";

export default function CompanyDispatchChannelManageRoute() {
  const { horizontalPadding, topInset } = useAppViewport();
  const companyId = useCompanyNumericId();
  const params = useLocalSearchParams<{ threadId?: string; conversationId?: string }>();
  const threadId = typeof params.threadId === "string" ? params.threadId : "dispatch";
  const paramConversationId = useMemo(() => {
    const raw = params.conversationId;
    const parsed = typeof raw === "string" ? Number.parseInt(raw, 10) : NaN;
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  }, [params.conversationId]);
  const [resolvedId, setResolvedId] = useState<number | null>(paramConversationId);

  const [resolveFailed, setResolveFailed] = useState(false);

  useEffect(() => {
    setResolvedId(paramConversationId);
    setResolveFailed(false);
  }, [paramConversationId]);

  useEffect(() => {
    if (resolvedId != null || !companyId) return;
    void resolveConversationId(companyId, threadId)
      .then((id) => {
        setResolvedId(id);
        setResolveFailed(false);
      })
      .catch(() => {
        setResolvedId(null);
        setResolveFailed(true);
      });
  }, [companyId, resolvedId, threadId]);

  const conversationId = resolvedId ?? paramConversationId;

  return (
    <CompanyContextGuard>
      <PermissionGuard permission="company:dashboard:read">
        <Screen scroll={false} backgroundColor={M.PAGE_BG} withHorizontalPadding={false} safeTop={false}>
          {conversationId != null ? (
            <ChannelManageScreen
              conversationId={conversationId}
              threadId={threadId}
              topInset={topInset}
              horizontalPadding={horizontalPadding}
            />
          ) : resolveFailed ? (
            <View style={{ flex: 1, alignItems: "center", justifyContent: "center", paddingTop: topInset, paddingHorizontal: horizontalPadding }}>
              <AppText variant="body" style={{ fontWeight: "600", marginBottom: 8 }}>
                Impossible de charger le canal
              </AppText>
              <AppText variant="bodyMuted" style={{ textAlign: "center", marginBottom: 16 }}>
                Le canal Dispatch n&apos;a pas pu être identifié.
              </AppText>
              <Pressable
                onPress={() => {
                  setResolveFailed(false);
                  if (companyId) {
                    void resolveConversationId(companyId, threadId)
                      .then((id) => setResolvedId(id))
                      .catch(() => setResolveFailed(true));
                  }
                }}
                style={{ backgroundColor: M.BRAND, paddingHorizontal: 20, paddingVertical: 12, borderRadius: 10 }}
              >
                <AppText variant="label" style={{ color: "#fff", fontWeight: "700" }}>
                  Réessayer
                </AppText>
              </Pressable>
            </View>
          ) : (
            <View style={{ flex: 1, alignItems: "center", justifyContent: "center", paddingTop: topInset }}>
              <ActivityIndicator color={M.BRAND} />
              <AppText variant="bodyMuted" style={{ marginTop: 12 }}>
                Résolution du canal…
              </AppText>
            </View>
          )}
        </Screen>
      </PermissionGuard>
    </CompanyContextGuard>
  );
}
