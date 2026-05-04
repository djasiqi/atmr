import { useEffect, useMemo, useState } from "react";
import { View } from "react-native";
import { useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useSession } from "../../../src/core/sessionProvider";
import type { AuthContext } from "../../../src/core/contracts/auth";
import {
  AppButton,
  AppText,
  brandSurfaceSoft,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../src/design/responsive";
import {
  getCompanyBillingSettings,
  getCompanyDispatchModes,
  getDispatchStatus,
  switchCompanyDispatchMode,
} from "../../../src/features/company/api/companyApi";
import { useCompanyRealtimeStatus } from "../../../src/features/company/hooks";
import { getResolvedCompanySocketUrl } from "../../../src/features/company/realtime/companyRealtimeBridge";
import { isContextSwitchClientSupported } from "../../../src/core/contextSwitchPolicy";

export default function CompanySettingsScreen() {
  const { horizontalPadding } = useAppViewport();
  const t = useResponsiveTokens();
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
  const [pendingAction, setPendingAction] = useState<"switch-driver" | "switch-company" | "refresh" | null>(null);
  const [dispatchMode, setDispatchMode] = useState<string>("inconnu");
  const [dispatchState, setDispatchState] = useState<string>("inconnu");
  const [billingSummary, setBillingSummary] = useState<string>("n/a");
  const roleGuardsEnabled = isFeatureEnabled("company_mobile_role_guards_enabled");
  const companyRealtime = useCompanyRealtimeStatus();
  const companySocketUrlResolved = getResolvedCompanySocketUrl() || "—";

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
  /** Bascule entreprise ↔ chauffeur : compte `COMPANY` uniquement (jamais un compte chauffeur seul). */
  const canTransportMobileRoleSwitch =
    isContextSwitchClientSupported() &&
    companyAccountForDoubleHat &&
    (activeCompanyContext?.allow_mobile_context_switch === true) &&
    (primaryDriverContext?.allow_mobile_context_switch === true);
  const canDispatchManage = canRunSensitiveAction("company:dispatch:manage", "company:dashboard:read");

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
          setBillingSummary(defaultType && contact ? `${defaultType} (${contact})` : "configure");
        }
      } catch {
        if (!mounted) return;
        setActionMessage("Impossible de charger les settings dispatch/billing.");
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
      setActionMessage("Basculé vers le contexte chauffeur.");
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
      setActionMessage("Session resynchronisée avec succès.");
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
    setPendingAction("refresh");
    setActionMessage(null);
    try {
      await switchCompanyDispatchMode({ contextId: activeCompanyId, mode: next });
      setDispatchMode(next);
      const latest = await getDispatchStatus({ contextId: activeCompanyId });
      setDispatchState(latest.dispatch_state);
      setActionMessage(`Mode dispatch bascule vers ${next}.`);
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
        backgroundColor={brandSurfaceSoft}
        withHorizontalPadding={false}
        contentContainerStyle={{
          paddingHorizontal: horizontalPadding,
          paddingTop: t.spacingMd,
          paddingBottom: t.spacingMd,
          gap: t.spacingSm,
        }}
      >
        <AppText variant="sectionTitle">Parametres entreprise</AppText>
        <AppText variant="body">
          Cette page pilote la session, la bascule de role et le contexte actif.
        </AppText>
        <AppText variant="body">Etat session: {status}</AppText>
        <AppText variant="body">Contexte actif: {activeContext?.context_id ?? "n/a"}</AppText>
        <AppText variant="body">Type contexte actif: {activeContext?.context_type ?? "n/a"}</AppText>
        <AppText variant="body">Compte chauffeur lie: {primaryDriverContext ? "oui" : "non"}</AppText>
        <AppText variant="body">
          Bascule entreprise / chauffeur: {String(canTransportMobileRoleSwitch)} (compte entreprise, web
          / mobile, dispatch) role={String(bootstrap?.user?.role ?? "—")}
        </AppText>
        <AppText variant="body">
          company_dispatch_enabled: {String(isFeatureEnabled("company_dispatch_enabled"))}
        </AppText>
        <AppText variant="body">
          company_realtime_enabled: {String(isFeatureEnabled("company_realtime_enabled"))}
        </AppText>
        <AppText variant="body">URL socket (résolue): {companySocketUrlResolved}</AppText>
        <AppText variant="body">
          Flux company (Socket.IO) : {companyRealtime.status} | branche:{" "}
          {companyRealtime.connected ? "oui" : "non"}
        </AppText>
        {companyRealtime.lastError ? (
          <AppText variant="error">Erreur flux: {companyRealtime.lastError}</AppText>
        ) : null}
        <AppText variant="body">Mode dispatch actif: {dispatchMode}</AppText>
        <AppText variant="body">Etat runtime dispatch: {dispatchState}</AppText>
        <AppText variant="body">Billing party par defaut: {billingSummary}</AppText>
        <AppText variant="caption">Source verite status dispatch: scheduler runtime</AppText>
        {error ? <AppText variant="error">Erreur session: {error}</AppText> : null}
        {actionMessage ? <AppText variant="body">{actionMessage}</AppText> : null}

        <AppButton
          title={pendingAction === "refresh" ? "Resynchronisation..." : "Resynchroniser la session"}
          disabled={pendingAction !== null}
          onPress={handleRefreshSession}
        />
        <AppButton
          title={
            pendingAction === "switch-driver"
              ? "Bascule chauffeur..."
              : "Passer en contexte chauffeur"
          }
          disabled={
            pendingAction !== null ||
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
              ? "Retour entreprise..."
              : "Revenir au contexte entreprise"
          }
          disabled={
            pendingAction !== null ||
            isCompanyActive ||
            !activeCompanyContext ||
            !canSwitchContext ||
            !canTransportMobileRoleSwitch
          }
          onPress={handleSwitchBackToCompany}
        />
        <AppButton
          title={pendingAction === "refresh" ? "Bascule mode..." : "Basculer mode dispatch"}
          disabled={pendingAction !== null || !activeCompanyId || !canDispatchManage}
          onPress={handleToggleDispatchMode}
        />
        <AppButton
          title="Se deconnecter"
          variant="secondary"
          disabled={pendingAction !== null}
          onPress={logout}
        />

        <View style={{ gap: t.spacingXs, marginTop: t.spacingSm }}>
          <AppText variant="sectionTitle">Contextes disponibles</AppText>
          {contexts.map((ctx: AuthContext) => (
            <AppText variant="body" key={ctx.context_id}>
              - {ctx.context_type} | {ctx.context_id} | default: {ctx.is_default ? "yes" : "no"}
            </AppText>
          ))}
          {contexts.length === 0 ? (
            <AppText variant="bodyMuted">Aucun contexte recu du bootstrap.</AppText>
          ) : null}
        </View>
      </Screen>
    </PermissionGuard>
  );
}
