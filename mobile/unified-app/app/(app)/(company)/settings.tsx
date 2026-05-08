import { useEffect, useMemo, useState } from "react";
import {
  Platform,
  StyleSheet,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useSession } from "../../../src/core/sessionProvider";
import type { AuthContext } from "../../../src/core/contracts/auth";
import {
  AppButton,
  AppText,
  Screen,
} from "../../../src/design/responsive";
import {
  getCompanyBillingSettings,
  getCompanyDispatchModes,
  getDispatchStatus,
  switchCompanyDispatchMode,
} from "../../../src/features/company/api/companyApi";
import { useCompanyRealtimeStatus } from "../../../src/features/company/hooks";
import {
  getResolvedCompanySocketEnvSource,
  getResolvedCompanySocketUrl,
} from "../../../src/features/company/realtime/companyRealtimeBridge";
import { isContextSwitchClientSupported } from "../../../src/core/contextSwitchPolicy";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { createShadow } from "../../../src/styles/shadowStyles";

const cardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

const BORDER_SLATE = "rgba(148, 163, 184, 0.22)";

type PendingAction =
  | "refresh"
  | "switch-driver"
  | "switch-company"
  | "dispatch-mode"
  | null;

function formatDispatchModeFr(mode: string): string {
  if (mode === "manual") return "Manuel";
  if (mode === "semi_auto") return "Semi-auto";
  if (mode === "fully_auto") return "Plein auto";
  return mode;
}

function realtimeStatusLabel(status: string): string {
  const s = status.toLowerCase();
  if (s === "healthy") return "Connecté";
  if (s === "connecting") return "Connexion…";
  if (s === "reconnecting") return "Reconnexion…";
  if (s === "degraded") return "Dégradé";
  if (s === "failed") return "Indisponible";
  if (s === "idle") return "Inactif";
  return status;
}

function realtimePillColors(status: string): { bg: string; fg: string; border: string } {
  const s = status.toLowerCase();
  if (s === "healthy") {
    return {
      bg: "rgba(22, 163, 74, 0.1)",
      fg: "#15803d",
      border: "rgba(22, 163, 74, 0.28)",
    };
  }
  if (s === "connecting" || s === "reconnecting") {
    return {
      bg: "rgba(245, 158, 11, 0.12)",
      fg: "#b45309",
      border: "rgba(245, 158, 11, 0.35)",
    };
  }
  if (s === "degraded") {
    return {
      bg: "rgba(234, 88, 12, 0.1)",
      fg: "#c2410c",
      border: "rgba(234, 88, 12, 0.28)",
    };
  }
  return {
    bg: "rgba(220, 38, 38, 0.1)",
    fg: "#b91c1c",
    border: "rgba(220, 38, 38, 0.28)",
  };
}

type InfoRowProps = { label: string; value: string; mono?: boolean };
function InfoRow({ label, value, mono }: InfoRowProps) {
  return (
    <View style={styles.kvRow}>
      <AppText variant="caption" style={styles.kvLabel} numberOfLines={2}>
        {label}
      </AppText>
      <AppText
        variant="body"
        style={[styles.kvValue, mono && styles.kvValueMono]}
        numberOfLines={4}
        selectable={Platform.OS === "web"}
      >
        {value}
      </AppText>
    </View>
  );
}

export default function CompanySettingsScreen() {
  const router = useRouter();
  const {
    activeContext,
    bootstrap,
    can,
    status,
    error,
    changeContext,
    bootstrapSession,
    logout,
  } = useSession();
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<PendingAction>(null);
  const [dispatchMode, setDispatchMode] = useState<string>("inconnu");
  const [dispatchState, setDispatchState] = useState<string>("inconnu");
  const [billingSummary, setBillingSummary] = useState<string>("n/a");
  const roleGuardsEnabled = isFeatureEnabled("company_mobile_role_guards_enabled");
  const companyRealtime = useCompanyRealtimeStatus();
  const companySocketUrlResolved = getResolvedCompanySocketUrl() || "—";
  const companySocketEnvSource = getResolvedCompanySocketEnvSource();

  const contexts = useMemo<AuthContext[]>(
    () => bootstrap?.available_contexts ?? [],
    [bootstrap?.available_contexts]
  );
  const companyContexts = useMemo(
    () => contexts.filter((ctx: AuthContext) => ctx.context_type === "company"),
    [contexts]
  );
  const driverContexts = useMemo(
    () => contexts.filter((ctx: AuthContext) => ctx.context_type === "driver"),
    [contexts]
  );
  const isCompanyActive = activeContext?.context_type === "company";

  const activeCompanyContext = useMemo(
    () =>
      companyContexts.find(
        (ctx: AuthContext) => ctx.context_id === activeContext?.context_id
      ) ??
      companyContexts[0] ??
      null,
    [activeContext?.context_id, companyContexts]
  );
  const primaryDriverContext = driverContexts[0] ?? null;
  const activeCompanyId =
    activeContext?.context_type === "company" ? activeContext.context_id : activeCompanyContext?.context_id ?? null;
  const contextPermissions = activeContext?.permissions ?? [];
  const canRunSensitiveAction = (permission: string, fallbackPermission: string) => {
    if (!roleGuardsEnabled) return true;
    if (contextPermissions.includes(permission)) return can(permission);
    return can(fallbackPermission);
  };
  const canSwitchContext = canRunSensitiveAction("company:context:switch", "company:dashboard:read");
  const companyAccountForDoubleHat =
    !bootstrap?.user?.role || String(bootstrap.user.role).toUpperCase() === "COMPANY";
  const canTransportMobileRoleSwitch =
    isContextSwitchClientSupported() &&
    companyAccountForDoubleHat &&
    (activeCompanyContext?.allow_mobile_context_switch === true) &&
    (primaryDriverContext?.allow_mobile_context_switch === true);
  const canDispatchManage = canRunSensitiveAction("company:dispatch:manage", "company:dashboard:read");

  const busy = pendingAction !== null;
  const fluxPill = realtimePillColors(companyRealtime.status);

  useEffect(() => {
    if (!activeCompanyId) return;
    let mounted = true;
    void (async () => {
      try {
        const [modesPayload, statusPayload, billingPayload] = await Promise.all([
          getCompanyDispatchModes({ contextId: activeCompanyId }),
          getDispatchStatus({ contextId: activeCompanyId }),
          getCompanyBillingSettings({ contextId: activeCompanyId }),
        ]);
        if (!mounted) return;
        if (modesPayload && typeof modesPayload === "object") {
          const modeCandidate = (modesPayload as Record<string, unknown>).dispatch_mode;
          if (typeof modeCandidate === "string" && modeCandidate.length > 0) {
            setDispatchMode(modeCandidate);
          }
        }
        setDispatchState(statusPayload.dispatch_state);
        if (billingPayload && typeof billingPayload === "object") {
          const payload = billingPayload as Record<string, unknown>;
          const defaultType =
            typeof payload.default_billed_to_type === "string" ? payload.default_billed_to_type : null;
          const contact =
            typeof payload.default_billed_to_contact === "string"
              ? payload.default_billed_to_contact
              : null;
          setBillingSummary(defaultType && contact ? `${defaultType} (${contact})` : "À configurer");
        }
      } catch {
        if (!mounted) return;
        setActionMessage("Impossible de charger les réglages dispatch / facturation.");
      }
    })();
    return () => {
      mounted = false;
    };
  }, [activeCompanyId]);

  async function handleSwitchToDriver() {
    if (!primaryDriverContext) {
      setActionMessage("Aucun contexte chauffeur disponible pour ce compte.");
      return;
    }
    setPendingAction("switch-driver");
    setActionMessage(null);
    try {
      await changeContext(primaryDriverContext.context_id);
      setActionMessage("Contexte chauffeur activé.");
      router.replace("/(app)/(driver)" as any);
    } catch (switchError) {
      setActionMessage(
        switchError instanceof Error
          ? switchError.message
          : "Impossible de basculer vers le contexte chauffeur."
      );
    } finally {
      setPendingAction(null);
    }
  }

  async function handleSwitchBackToCompany() {
    if (!activeCompanyContext) {
      setActionMessage("Aucun contexte entreprise disponible.");
      return;
    }
    setPendingAction("switch-company");
    setActionMessage(null);
    try {
      await changeContext(activeCompanyContext.context_id);
      setActionMessage("Contexte entreprise réactivé.");
      router.replace("/(app)/(company)/settings" as any);
    } catch (switchError) {
      setActionMessage(
        switchError instanceof Error
          ? switchError.message
          : "Impossible de réactiver le contexte entreprise."
      );
    } finally {
      setPendingAction(null);
    }
  }

  async function handleRefreshSession() {
    setPendingAction("refresh");
    setActionMessage(null);
    try {
      await bootstrapSession();
      setActionMessage("Session resynchronisée.");
    } catch (refreshError) {
      setActionMessage(
        refreshError instanceof Error
          ? refreshError.message
          : "Impossible de synchroniser la session."
      );
    } finally {
      setPendingAction(null);
    }
  }

  async function handleToggleDispatchMode() {
    if (!activeCompanyId) return;
    const order: ("manual" | "semi_auto" | "fully_auto")[] = ["manual", "semi_auto", "fully_auto"];
    const current = order.includes(dispatchMode as (typeof order)[number])
      ? (dispatchMode as (typeof order)[number])
      : "manual";
    const next = order[(order.indexOf(current) + 1) % order.length];
    setPendingAction("dispatch-mode");
    setActionMessage(null);
    try {
      await switchCompanyDispatchMode({ contextId: activeCompanyId, mode: next });
      setDispatchMode(next);
      const latest = await getDispatchStatus({ contextId: activeCompanyId });
      setDispatchState(latest.dispatch_state);
      setActionMessage(`Mode dispatch : ${formatDispatchModeFr(next)}.`);
    } catch (switchError) {
      setActionMessage(
        switchError instanceof Error ? switchError.message : "Impossible de changer le mode dispatch."
      );
    } finally {
      setPendingAction(null);
    }
  }

  return (
    <PermissionGuard permission="company:dashboard:read">
      <Screen
        scroll
        backgroundColor={E.BG}
        withHorizontalPadding
        contentContainerStyle={styles.page}
      >
        <View style={styles.hero}>
          <View style={styles.heroIconWell}>
            <Ionicons name="settings-outline" size={26} color={E.BRAND} />
          </View>
          <View style={styles.heroTextCol}>
            <AppText variant="sectionTitle" style={styles.heroTitle}>
              Paramètres entreprise
            </AppText>
            <AppText variant="caption" style={styles.heroSubtitle}>
              Session, contexte actif et connexion temps réel — alignés sur votre espace dispatch.
            </AppText>
          </View>
        </View>

        {error ? (
          <View style={styles.alertStrip} accessibilityRole="alert">
            <Ionicons name="alert-circle" size={20} color="#b91c1c" />
            <AppText variant="body" style={styles.alertText}>
              {error}
            </AppText>
          </View>
        ) : null}

        {companyRealtime.lastError ? (
          <View style={[styles.alertStrip, styles.alertStripFlux]} accessibilityRole="alert">
            <Ionicons name="pulse-outline" size={20} color="#b91c1c" />
            <View style={styles.alertCol}>
              <AppText variant="body" style={styles.alertTitleSmall}>
                Flux temps réel
              </AppText>
              <AppText variant="caption" style={styles.alertTextMuted}>
                {companyRealtime.lastError}
              </AppText>
            </View>
          </View>
        ) : null}

        {actionMessage ? (
          <View style={styles.toastInfo}>
            <Ionicons name="checkmark-circle-outline" size={18} color={E.BRAND} />
            <AppText variant="body" style={styles.toastInfoText}>
              {actionMessage}
            </AppText>
          </View>
        ) : null}

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="person-circle-outline" size={20} color={E.TEXT_SEC} />
            <AppText variant="body" style={styles.cardTitle}>
              Session et contexte
            </AppText>
          </View>
          <InfoRow label="État de la session" value={String(status)} />
          <View style={styles.divider} />
          <InfoRow label="Contexte actif" value={activeContext?.context_id ?? "—"} mono />
          <View style={styles.divider} />
          <InfoRow label="Type de contexte" value={activeContext?.context_type ?? "—"} />
          <View style={styles.divider} />
          <InfoRow label="Rôle compte" value={String(bootstrap?.user?.role ?? "—")} />
          <View style={styles.divider} />
          <InfoRow
            label="Double casquette (entreprise / chauffeur)"
            value={canTransportMobileRoleSwitch ? "Disponible" : "Non disponible"}
          />
          <View style={styles.divider} />
          <InfoRow label="Compte chauffeur lié" value={primaryDriverContext ? "Oui" : "Non"} />
        </View>

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="cloud-outline" size={20} color={E.TEXT_SEC} />
            <AppText variant="body" style={styles.cardTitle}>
              Connectivité
            </AppText>
          </View>
          <View style={styles.fluxRow}>
            <AppText variant="caption" style={styles.kvLabel}>
              Flux Socket.IO
            </AppText>
            <View style={[styles.statusPill, { backgroundColor: fluxPill.bg, borderColor: fluxPill.border }]}>
              <AppText style={[styles.statusPillText, { color: fluxPill.fg }]}>
                {realtimeStatusLabel(companyRealtime.status)}
              </AppText>
            </View>
          </View>
          <AppText variant="caption" style={styles.fluxSub}>
            Branche : {companyRealtime.connected ? "oui" : "non"}
          </AppText>
          <View style={styles.divider} />
          <InfoRow label="URL socket (résolue)" value={companySocketUrlResolved} mono />
          <View style={styles.divider} />
          <InfoRow label="Source (env)" value={companySocketEnvSource} />
        </View>

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="car-outline" size={20} color={E.TEXT_SEC} />
            <AppText variant="body" style={styles.cardTitle}>
              Dispatch et facturation
            </AppText>
          </View>
          <InfoRow label="Mode dispatch" value={formatDispatchModeFr(dispatchMode)} />
          <View style={styles.divider} />
          <InfoRow label="État moteur" value={String(dispatchState)} />
          <View style={styles.divider} />
          <InfoRow label="Référence statut" value="Scheduler runtime" />
          <View style={styles.divider} />
          <InfoRow label="Facturation (défaut)" value={billingSummary} />
        </View>

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="flag-outline" size={20} color={E.TEXT_SEC} />
            <AppText variant="body" style={styles.cardTitle}>
              Fonctionnalités (app)
            </AppText>
          </View>
          <InfoRow
            label="Dispatch entreprise"
            value={isFeatureEnabled("company_dispatch_enabled") ? "Activé" : "Désactivé"}
          />
          <View style={styles.divider} />
          <InfoRow
            label="Temps réel entreprise"
            value={isFeatureEnabled("company_realtime_enabled") ? "Activé" : "Désactivé"}
          />
        </View>

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="flash-outline" size={20} color={E.TEXT_SEC} />
            <AppText variant="body" style={styles.cardTitle}>
              Actions
            </AppText>
          </View>
          <View style={styles.actionsCol}>
            <AppButton
              title={pendingAction === "refresh" ? "Resynchronisation…" : "Resynchroniser la session"}
              disabled={busy}
              onPress={handleRefreshSession}
            />
            <AppButton
              title={
                pendingAction === "switch-driver"
                  ? "Bascule…"
                  : "Passer en contexte chauffeur"
              }
              disabled={
                busy ||
                !isCompanyActive ||
                !primaryDriverContext ||
                !canSwitchContext ||
                !canTransportMobileRoleSwitch
              }
              onPress={handleSwitchToDriver}
            />
            <AppButton
              title={
                pendingAction === "switch-company"
                  ? "Retour…"
                  : "Revenir au contexte entreprise"
              }
              disabled={
                busy ||
                isCompanyActive ||
                !activeCompanyContext ||
                !canSwitchContext ||
                !canTransportMobileRoleSwitch
              }
              onPress={handleSwitchBackToCompany}
            />
            <AppButton
              title={
                pendingAction === "dispatch-mode"
                  ? "Changement de mode…"
                  : "Basculer le mode dispatch"
              }
              disabled={busy || !activeCompanyId || !canDispatchManage}
              onPress={handleToggleDispatchMode}
            />
            <AppButton
              title="Se déconnecter"
              variant="secondary"
              disabled={busy}
              onPress={logout}
            />
          </View>
        </View>

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="layers-outline" size={20} color={E.TEXT_SEC} />
            <AppText variant="body" style={styles.cardTitle}>
              Contextes disponibles
            </AppText>
          </View>
          {contexts.length === 0 ? (
            <AppText variant="bodyMuted" style={styles.emptyContexts}>
              Aucun contexte renvoyé par la session.
            </AppText>
          ) : (
            contexts.map((ctx: AuthContext) => {
              const active = ctx.context_id === activeContext?.context_id;
              return (
                <View
                  key={ctx.context_id}
                  style={[styles.contextRow, active && styles.contextRowActive]}
                >
                  <View
                    style={[
                      styles.contextIconWell,
                      ctx.context_type === "company" ? styles.contextIconCompany : styles.contextIconDriver,
                    ]}
                  >
                    <Ionicons
                      name={ctx.context_type === "company" ? "business-outline" : "car-sport-outline"}
                      size={18}
                      color={ctx.context_type === "company" ? E.BRAND : E.TEXT_SEC}
                    />
                  </View>
                  <View style={styles.contextMeta}>
                    <View style={styles.contextTitleRow}>
                      <AppText variant="body" style={styles.contextTypeLabel}>
                        {ctx.context_type === "company" ? "Entreprise" : "Chauffeur"}
                      </AppText>
                      {ctx.is_default ? (
                        <View style={styles.defaultBadge}>
                          <AppText variant="caption" style={styles.defaultBadgeText}>
                            défaut
                          </AppText>
                        </View>
                      ) : null}
                      {active ? (
                        <View style={styles.activeBadge}>
                          <AppText variant="caption" style={styles.activeBadgeText}>
                            actif
                          </AppText>
                        </View>
                      ) : null}
                    </View>
                    <AppText variant="caption" style={styles.contextId} selectable={Platform.OS === "web"}>
                      {ctx.context_id}
                    </AppText>
                  </View>
                </View>
              );
            })
          )}
        </View>
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingTop: 12,
    paddingBottom: 28,
    gap: 16,
  },
  hero: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    paddingVertical: 4,
  },
  heroIconWell: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: E.CARD,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    alignItems: "center",
    justifyContent: "center",
    ...cardShadow,
  },
  heroTextCol: { flex: 1, minWidth: 0, gap: 6 },
  heroTitle: {
    color: E.TEXT,
    fontSize: 20,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  heroSubtitle: {
    color: E.TEXT_SEC,
    lineHeight: 18,
  },
  card: {
    backgroundColor: E.CARD,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    padding: 16,
    gap: 0,
    ...cardShadow,
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 14,
  },
  cardTitle: {
    fontWeight: "700",
    color: E.TEXT,
    fontSize: 15,
    letterSpacing: 0.15,
  },
  kvRow: {
    paddingVertical: 10,
    gap: 4,
  },
  kvLabel: {
    color: E.TEXT_SEC,
    fontSize: 12,
    fontWeight: "600",
    letterSpacing: 0.2,
    textTransform: "uppercase",
  },
  kvValue: {
    color: E.TEXT,
    fontSize: 15,
    lineHeight: 22,
    fontWeight: "500",
  },
  kvValueMono: {
    fontFamily: Platform.select({ ios: "Menlo", android: "monospace", default: "monospace" }),
    fontSize: 13,
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(148, 163, 184, 0.35)",
    marginVertical: 2,
  },
  fluxRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    paddingVertical: 6,
  },
  fluxSub: {
    color: E.TEXT_MUTED,
    marginTop: 2,
    marginBottom: 8,
  },
  statusPill: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 10,
    borderWidth: 1,
  },
  statusPillText: {
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  alertStrip: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    padding: 14,
    borderRadius: 14,
    backgroundColor: "#FEF2F2",
    borderWidth: 1,
    borderColor: "rgba(220, 38, 38, 0.22)",
    borderLeftWidth: 3,
    borderLeftColor: "#DC2626",
  },
  alertStripFlux: {
    alignItems: "center",
  },
  alertCol: { flex: 1, minWidth: 0, gap: 4 },
  alertTitleSmall: { fontWeight: "700", color: E.TEXT },
  alertText: {
    flex: 1,
    color: "#b91c1c",
    fontWeight: "600",
    fontSize: 14,
    lineHeight: 20,
  },
  alertTextMuted: {
    color: E.TEXT_SEC,
    lineHeight: 18,
  },
  toastInfo: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    padding: 12,
    borderRadius: 12,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.2)",
  },
  toastInfoText: {
    flex: 1,
    color: E.BRAND_DARK,
    fontWeight: "600",
    fontSize: 14,
  },
  actionsCol: { gap: 10, marginTop: 4 },
  emptyContexts: { paddingVertical: 8 },
  contextRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 12,
    paddingHorizontal: 4,
    borderRadius: 12,
  },
  contextRowActive: {
    backgroundColor: "rgba(0, 121, 107, 0.06)",
  },
  contextIconWell: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  contextIconCompany: {
    backgroundColor: "rgba(0, 121, 107, 0.1)",
  },
  contextIconDriver: {
    backgroundColor: "rgba(148, 163, 184, 0.15)",
  },
  contextMeta: { flex: 1, minWidth: 0, gap: 4 },
  contextTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 8,
  },
  contextTypeLabel: { fontWeight: "700", color: E.TEXT },
  contextId: {
    color: E.TEXT_SEC,
    fontFamily: Platform.select({ ios: "Menlo", android: "monospace", default: "monospace" }),
    fontSize: 12,
  },
  defaultBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
    backgroundColor: "rgba(148, 163, 184, 0.2)",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.35)",
  },
  defaultBadgeText: {
    fontSize: 10,
    fontWeight: "700",
    color: E.TEXT_SEC,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  activeBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
    backgroundColor: "rgba(0, 121, 107, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.3)",
  },
  activeBadgeText: {
    fontSize: 10,
    fontWeight: "700",
    color: E.BRAND_DARK,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
});
