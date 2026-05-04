import { useEffect, useMemo, useState } from "react";
import { ActivityIndicator, RefreshControl, StyleSheet, TouchableOpacity, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import "dayjs/locale/fr";
import { useLocalSearchParams, useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import {
  useActiveCompanyContextId,
  useCompanyRideActions,
  useCompanyRealtimeInvalidation,
  useCompanyRideDetailsQuery,
  useCompanyInvoicesReadonlyQuery,
} from "../../../src/features/company/hooks";
import { emitCompanyDispatchTelemetry } from "../../../src/features/company/telemetry/companyTelemetry";
import { useSession } from "../../../src/core/sessionProvider";
import { contextRealtimeRouter } from "../../../src/core/realtime/contextRealtimeRouter";
import { AppButton, AppText, Modal, Screen } from "../../../src/design/responsive";
import {
  scheduleCompanyRide,
  getCompanyAvailableDrivers,
  getCompanyPartnershipsForTransfer,
} from "../../../src/features/company/api/companyApi";
import { normalizeCompanyEventType } from "../../../src/core/realtime/eventContracts";
import { NotesEditor } from "../../../src/features/company/components/NotesEditor";
import { TransferRideModal } from "../../../src/features/company/components/transfers/TransferRideModal";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { getEnterpriseStatusColors } from "../../../src/features/company/theme/enterpriseStatusColors";
import { createShadow } from "../../../src/styles/shadowStyles";
dayjs.locale("fr");

function resolveMissionIdFromEvent(payload: {
  mission_id?: unknown;
  booking_id?: unknown;
  id?: unknown;
}): number | undefined {
  const candidate = payload.mission_id ?? payload.booking_id ?? payload.id;
  if (typeof candidate === "number" && Number.isFinite(candidate)) {
    return candidate;
  }
  if (typeof candidate === "string") {
    const parsed = Number.parseInt(candidate, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function readClientName(data: Record<string, unknown> | null | undefined): string | null {
  if (!data) return null;
  const c = data.client;
  if (c && typeof c === "object" && "name" in c && typeof (c as { name?: string }).name === "string") {
    const n = (c as { name: string }).name.trim();
    if (n) return n;
  }
  const s = data.client_name;
  return typeof s === "string" && s.trim() ? s.trim() : null;
}

function mapStatusToLabel(status: string) {
  const t = String(status).toLowerCase();
  if (t === "assigned") return "Assignée";
  if (t === "completed") return "Terminée";
  if (t === "cancelled") return "Annulée";
  if (t === "en_route" || t === "arrived") return "En route";
  if (t === "in_progress") return "En cours";
  if (t === "pending") return "En attente";
  return status;
}

const headerShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.04,
  shadowRadius: 6,
  elevation: 2,
});
const cardShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.03,
  shadowRadius: 4,
  elevation: 1,
});
const backBtnShadow = createShadow({
  shadowColor: E.BRAND,
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.2,
  shadowRadius: 6,
  elevation: 3,
});

export default function CompanyRideDetailsScreen() {
  const router = useRouter();
  const { rideId } = useLocalSearchParams<{ rideId?: string }>();
  const { activeContext } = useSession();
  const date = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const contextId = useActiveCompanyContextId();
  const rideIdNumber = rideId ? Number.parseInt(String(rideId), 10) : null;
  const missionQuery = useCompanyRideDetailsQuery({ date, rideId: rideIdNumber });
  const invoicesQuery = useCompanyInvoicesReadonlyQuery({
    q: rideIdNumber != null ? String(rideIdNumber) : undefined,
    limit: 50,
  });
  const { invalidate } = useCompanyRealtimeInvalidation();
  const rideActions = useCompanyRideActions();
  const [drivers, setDrivers] = useState<{ id: number; label: string }[]>([]);
  const [selectedDriverId, setSelectedDriverId] = useState<number | null>(null);
  const [partners, setPartners] = useState<{ id: number; label: string }[]>([]);
  const [selectedPartnerId, setSelectedPartnerId] = useState<number | null>(null);
  const [assignOpen, setAssignOpen] = useState(false);
  const [transferOpen, setTransferOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const linkedInvoice = useMemo(() => {
    const rows = invoicesQuery.data ?? [];
    if (rideIdNumber == null) return null;
    return (
      rows.find((entry) => {
        const bookingId = Number(entry.booking_id ?? entry.mission_id ?? entry.reservation_id);
        return Number.isFinite(bookingId) && bookingId === rideIdNumber;
      }) ?? null
    );
  }, [invoicesQuery.data, rideIdNumber]);
  const hasUnpaidInvoice = useMemo(() => {
    if (!linkedInvoice) return false;
    const status = String(linkedInvoice.status ?? linkedInvoice.payment_status ?? "").toLowerCase();
    return status === "unpaid" || status === "pending" || status === "overdue";
  }, [linkedInvoice]);

  useEffect(() => {
    emitCompanyDispatchTelemetry(
      "company.dispatch.opened",
      {
        source: "company.ride-details.screen",
        context_type: "company",
        context_id: activeContext?.context_id ?? null,
        mission_id: rideId ? Number.parseInt(String(rideId), 10) || null : null,
      },
      { allowWhenDisabled: true }
    );
  }, [activeContext?.context_id, rideId]);

  useEffect(() => {
    if (!activeContext || activeContext.context_type !== "company") return;
    return contextRealtimeRouter.subscribe(activeContext.context_id, (event) => {
      if (!event || typeof event !== "object") return;
      const payload = event as {
        event_type?: string;
        booking_id?: unknown;
        mission_id?: unknown;
        id?: unknown;
      };
      const missionId = resolveMissionIdFromEvent(payload);
      if (!rideIdNumber || missionId !== rideIdNumber) return;
      const eventType = normalizeCompanyEventType(payload.event_type);
      if (eventType === "booking_updated") {
        invalidate("booking_updated", rideIdNumber);
      } else if (eventType === "booking_cancelled") {
        invalidate("booking_cancelled", rideIdNumber);
      }
    });
  }, [activeContext, invalidate, rideIdNumber]);

  const openAssignModal = async () => {
    if (!contextId || !rideIdNumber) return;
    setSaving(true);
    setMutationError(null);
    try {
      const payload = await getCompanyAvailableDrivers({ contextId });
      const rawPayload = payload as { drivers?: unknown[]; items?: unknown[] };
      const rows = [
        ...(Array.isArray(rawPayload.drivers) ? rawPayload.drivers : []),
        ...(Array.isArray(rawPayload.items) ? rawPayload.items : []),
      ].filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object");
      const options = rows
        .map((row) => {
          const idRaw = row.driver_id ?? row.id;
          const id = typeof idRaw === "string" ? Number.parseInt(idRaw, 10) : Number(idRaw);
          if (!Number.isFinite(id)) return null;
          const fromApi =
            typeof row.driver_name === "string" && row.driver_name.trim()
              ? row.driver_name.trim()
              : null;
          const dName =
            fromApi ||
            (typeof row.display_name === "string" && row.display_name.trim() ? row.display_name.trim() : null) ||
            (typeof row.name === "string" && row.name.trim() ? row.name.trim() : null);
          const fn = typeof row.first_name === "string" ? row.first_name.trim() : "";
          const ln = typeof row.last_name === "string" ? row.last_name.trim() : "";
          const fromParts = [fn, ln].filter(Boolean).join(" ").trim();
          const label = dName || (fromParts.length > 0 ? fromParts : `Chauffeur #${id}`);
          return { id, label: String(label) };
        })
        .filter((item): item is { id: number; label: string } => item != null);
      setDrivers(options);
      setSelectedDriverId(options[0]?.id ?? null);
      setAssignOpen(true);
    } catch (error) {
      setMutationError(error instanceof Error ? error.message : "Liste chauffeurs indisponible.");
    } finally {
      setSaving(false);
    }
  };

  const assignDriver = async () => {
    if (!rideIdNumber || !selectedDriverId) return;
    setSaving(true);
    setMutationError(null);
    try {
      await rideActions.assign.mutateAsync({
        missionId: rideIdNumber,
        driverId: selectedDriverId,
      });
      emitCompanyDispatchTelemetry(
        "company.dispatch.driver_selected",
        {
          source: "company.ride-details.assign",
          context_type: "company",
          context_id: activeContext?.context_id ?? null,
          mission_id: rideIdNumber,
          driver_id: selectedDriverId,
        },
        { allowWhenDisabled: true }
      );
      setAssignOpen(false);
      await missionQuery.refetch();
    } catch (error) {
      setMutationError(error instanceof Error ? error.message : "Assignation impossible.");
    } finally {
      setSaving(false);
    }
  };

  const openTransferModal = async () => {
    if (!contextId || !rideIdNumber) return;
    setSaving(true);
    setMutationError(null);
    try {
      const payload = await getCompanyPartnershipsForTransfer({ contextId });
      const rows = Array.isArray((payload as { items?: unknown[] })?.items)
        ? ((payload as { items?: Record<string, unknown>[] }).items ?? [])
        : [];
      const options = rows
        .map((row) => {
          const id = Number(row.company_id ?? row.target_company_id ?? row.id);
          if (!Number.isFinite(id)) return null;
          const label = String(row.company_name ?? row.name ?? `Company #${id}`);
          return { id, label };
        })
        .filter((item): item is { id: number; label: string } => item != null);
      setPartners(options);
      setSelectedPartnerId(options[0]?.id ?? null);
      setTransferOpen(true);
    } catch (error) {
      setMutationError(error instanceof Error ? error.message : "Liste partenaires indisponible.");
    } finally {
      setSaving(false);
    }
  };

  const d = (missionQuery.data as Record<string, unknown> | null | undefined) ?? null;
  const statusStr = d ? String((d as { status?: string }).status ?? "pending") : "pending";
  const statusStyle = getEnterpriseStatusColors(statusStr);
  const clientTitle = d ? readClientName(d) : null;
  const titleLine = clientTitle
    ? `${clientTitle} · ${
        (d as { scheduled_at?: string | null })?.scheduled_at
          ? dayjs(String((d as { scheduled_at?: string }).scheduled_at)).format("DD MMM HH:mm")
          : "⏱️ À définir"
      }`
    : "Course";
  const idLine = d ? String((d as { mission_id?: number }).mission_id ?? rideId ?? "—") : (rideId ?? "—");
  const scheduled = d
    ? String(
        (d as { scheduled_at?: string | null })?.scheduled_at
          ? dayjs(String((d as { scheduled_at: string }).scheduled_at)).format("DD/MM/YYYY HH:mm")
          : "n/a"
      )
    : "n/a";
  const pickup = d ? String((d as { pickup_label?: string | null })?.pickup_label ?? "—") : "—";
  const drop = d ? String((d as { dropoff_label?: string | null })?.dropoff_label ?? "—") : "—";
  const driverId = d && typeof (d as { driver_id?: number | null }).driver_id === "number" ? (d as { driver_id: number }).driver_id : null;

  if (missionQuery.isLoading && !d) {
    return (
      <PermissionGuard permission="company:rides:read">
        <View style={styles.loadWrap}>
          <ActivityIndicator color={E.BRAND} size="large" />
          <AppText variant="bodyMuted" style={styles.loadText}>
            Chargement…
          </AppText>
        </View>
      </PermissionGuard>
    );
  }

  return (
    <PermissionGuard permission="company:rides:read">
      <Screen
        scroll
        backgroundColor={E.BG}
        withHorizontalPadding={false}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={missionQuery.isLoading}
            onRefresh={() => void missionQuery.refetch()}
            tintColor={E.BRAND}
          />
        }
      >
        {d ? (
          <>
            <View style={styles.header}>
              <TouchableOpacity style={styles.headerBack} onPress={() => router.back()} activeOpacity={0.85}>
                <Ionicons name="chevron-back" size={20} color={E.TEXT_SEC} />
              </TouchableOpacity>
              <View style={styles.headerCenter}>
                <AppText variant="sectionTitle" style={styles.headerTitle} numberOfLines={1}>
                  {titleLine}
                </AppText>
                <AppText variant="caption" style={styles.headerSub}>
                  #{idLine}
                </AppText>
              </View>
              <View style={styles.headerStatusBadge}>
                <View style={[styles.statusDot, { backgroundColor: statusStyle.text }]} />
                <AppText
                  variant="caption"
                  style={[styles.statusLabel, { color: statusStyle.text }]}
                  numberOfLines={1}
                >
                  {mapStatusToLabel(statusStr).toUpperCase()}
                </AppText>
              </View>
            </View>
            <View style={styles.section}>
              <AppText variant="sectionTitle" style={styles.sectionTitle}>
                Informations
              </AppText>
              {[
                { label: "Statut", value: mapStatusToLabel(statusStr) },
                { label: "Départ", value: pickup },
                { label: "Arrivée", value: drop },
                { label: "Prévue", value: scheduled },
                { label: "Chauffeur", value: driverId != null ? `#${driverId}` : "Non assigné" },
                {
                  label: "Facturation",
                  value: linkedInvoice
                    ? String(linkedInvoice.status ?? linkedInvoice.payment_status ?? "n/a")
                    : "Aucune facture liée",
                },
                ...(hasUnpaidInvoice ? [{ label: "Alerte", value: "Impayé" }] as const : []),
              ].map((row, idx, arr) => (
                <View
                  key={row.label + String(idx)}
                  style={[
                    styles.infoRow,
                    idx === arr.length - 1 ? styles.infoRowLast : null,
                  ]}
                >
                  <AppText variant="bodyMuted" style={styles.infoLabel}>
                    {row.label}
                  </AppText>
                  <AppText
                    variant="body"
                    style={[
                      styles.infoValue,
                      row.label === "Alerte" ? { color: E.DANGER, fontWeight: "600" } : null,
                    ]}
                  >
                    {row.value}
                  </AppText>
                </View>
              ))}
            </View>
            <View style={styles.section}>
              <AppText variant="sectionTitle" style={styles.sectionTitle}>
                Actions
              </AppText>
              <View style={styles.quickRow}>
                <TouchableOpacity
                  style={styles.quickBtn}
                  onPress={() => void openAssignModal()}
                  disabled={!contextId || saving}
                >
                  {saving ? (
                    <ActivityIndicator color={E.BRAND} size="small" />
                  ) : (
                    <AppText variant="label" style={styles.quickBtnText}>
                      {driverId != null ? "Réassigner" : "Assigner"}
                    </AppText>
                  )}
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.quickBtn}
                  onPress={async () => {
                    if (!contextId || !rideIdNumber) return;
                    setSaving(true);
                    setMutationError(null);
                    try {
                      await scheduleCompanyRide({
                        contextId,
                        missionId: rideIdNumber,
                        payload: {
                          pickup_at: new Date(Date.now() + 20 * 60 * 1000).toISOString(),
                          timezone: "Europe/Zurich",
                          note: "Reschedule from ride-details",
                          force_recompute: true,
                        },
                      });
                      await missionQuery.refetch();
                    } catch (error) {
                      setMutationError(
                        error instanceof Error ? error.message : "Planification impossible."
                      );
                    } finally {
                      setSaving(false);
                    }
                  }}
                  disabled={!contextId || saving}
                >
                  <AppText variant="label" style={styles.quickBtnText}>
                    {saving ? "Planif…" : "Planifier +20 min"}
                  </AppText>
                </TouchableOpacity>
              </View>
              <View style={styles.quickRow}>
                <TouchableOpacity
                  style={styles.quickBtn}
                  onPress={() =>
                    rideActions.urgent.mutate({
                      missionId: rideIdNumber as number,
                      payload: {
                        urgent: true,
                        reason_code: "manual_company_priority",
                        source: "ride_details",
                      },
                    })
                  }
                  disabled={!contextId || rideActions.urgent.isPending}
                >
                  <AppText variant="label" style={styles.quickBtnText}>
                    {rideActions.urgent.isPending ? "Urgent…" : "Urgent"}
                  </AppText>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.quickBtn}
                  onPress={() => void openTransferModal()}
                  disabled={!contextId || saving}
                >
                  <AppText variant="label" style={styles.quickBtnText}>
                    Transférer
                  </AppText>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.quickBtn, { borderColor: "rgba(220,53,69,0.35)" }]}
                  onPress={() =>
                    rideActions.cancel.mutate({
                      missionId: rideIdNumber as number,
                      reasonCode: "cancelled_from_mobile_dispatch",
                      note: "Cancellation requested from company ride-details",
                    })
                  }
                  disabled={!contextId || rideActions.cancel.isPending}
                >
                  <AppText variant="label" style={[styles.quickBtnText, { color: E.DANGER }]}>
                    {rideActions.cancel.isPending ? "…" : "Annuler"}
                  </AppText>
                </TouchableOpacity>
              </View>
            </View>
            <View style={styles.section}>
              <AppText variant="sectionTitle" style={styles.sectionTitle}>
                Notes
              </AppText>
              <NotesEditor
                initialValue={String(d.notes ?? "")}
                onSave={async (notes) => {
                  setMutationError(null);
                  if (notes.length === 0) return;
                }}
                saveLabel="Enregistrer (local)"
              />
            </View>
            {mutationError ? (
              <AppText variant="error" style={styles.mutationErr}>
                {mutationError}
              </AppText>
            ) : null}
            <TouchableOpacity
              style={styles.backCta}
              onPress={() => router.back()}
              activeOpacity={0.85}
            >
              <Ionicons name="arrow-back" size={18} color="#FFFFFF" style={styles.backIcon} />
              <AppText variant="label" style={styles.backCtaText}>
                Retour aux courses
              </AppText>
            </TouchableOpacity>
          </>
        ) : (
          <View style={styles.emptyState}>
            <Ionicons name="alert-circle-outline" size={48} color={E.DANGER} style={{ marginBottom: 12 }} />
            <AppText variant="body" style={styles.emptyText}>
              Mission introuvable pour cette date.
            </AppText>
            <TouchableOpacity onPress={() => void missionQuery.refetch()} style={styles.retryPill}>
              <AppText variant="label" style={styles.retryPillText}>
                Réessayer
              </AppText>
            </TouchableOpacity>
          </View>
        )}
      </Screen>
      <Modal visible={assignOpen} title="Selection chauffeur" onClose={() => !saving && setAssignOpen(false)}>
        {drivers.map((driver) => (
          <AppButton
            key={driver.id}
            title={`${selectedDriverId === driver.id ? "• " : ""}${driver.label}`}
            variant="secondary"
            onPress={() => setSelectedDriverId(driver.id)}
          />
        ))}
        <AppButton
          title={saving ? "Assignation..." : "Confirmer assignation"}
          variant="primary"
          onPress={() => void assignDriver()}
          disabled={saving || selectedDriverId == null}
        />
      </Modal>
      <TransferRideModal
        visible={transferOpen}
        pending={saving}
        options={partners}
        selectedPartnerId={selectedPartnerId}
        error={mutationError}
        onSelect={setSelectedPartnerId}
        onConfirm={() => {
          void (async () => {
            if (!rideIdNumber || !selectedPartnerId) return;
            setSaving(true);
            try {
              await rideActions.transfer.mutateAsync({
                missionId: rideIdNumber,
                targetCompanyId: selectedPartnerId,
              });
              setTransferOpen(false);
              await missionQuery.refetch();
            } catch (error) {
              setMutationError(error instanceof Error ? error.message : "Transfert impossible.");
            } finally {
              setSaving(false);
            }
          })();
        }}
        onClose={() => !saving && setTransferOpen(false)}
      />
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  loadWrap: { flex: 1, backgroundColor: E.BG, alignItems: "center", justifyContent: "center", padding: 24 },
  loadText: { marginTop: 10 },
  content: { paddingBottom: 48 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: E.CARD,
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: E.BORDER,
    ...headerShadow,
  },
  headerBack: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: E.BG,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 10,
  },
  headerCenter: { flex: 1, minWidth: 0 },
  headerTitle: { color: E.TEXT, fontSize: 16, fontWeight: "700" as const },
  headerSub: { color: E.TEXT_MUTED, fontSize: 12, marginTop: 2 },
  headerStatusBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 8,
    backgroundColor: E.BG,
  },
  statusDot: { width: 7, height: 7, borderRadius: 4 },
  statusLabel: { fontWeight: "600" as const, letterSpacing: 0.3, maxWidth: 84 },
  section: {
    backgroundColor: E.CARD,
    borderRadius: 14,
    padding: 16,
    margin: 16,
    marginBottom: 0,
    borderWidth: 1,
    borderColor: E.BORDER,
    ...cardShadow,
  },
  sectionTitle: { color: E.TEXT, marginBottom: 12 },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: E.BORDER,
  },
  infoRowLast: { marginBottom: 0, paddingBottom: 0, borderBottomWidth: 0 },
  infoLabel: { color: E.TEXT_SEC, fontSize: 13, flex: 1, paddingRight: 8 },
  infoValue: { color: E.TEXT, fontSize: 13, flex: 1, textAlign: "right", fontWeight: "500" as const },
  quickRow: { flexDirection: "row", flexWrap: "wrap", gap: 10, marginTop: 4 },
  quickBtn: {
    flex: 1,
    minWidth: 120,
    backgroundColor: E.BG,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: "center",
    borderWidth: 1,
    borderColor: E.BORDER,
  },
  quickBtnText: { color: E.TEXT, fontWeight: "600" as const },
  mutationErr: { marginHorizontal: 16, marginTop: 8, fontWeight: "600" },
  backCta: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: E.BRAND,
    borderRadius: 12,
    paddingVertical: 13,
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 8,
    ...backBtnShadow,
  },
  backIcon: { marginRight: 8 },
  backCtaText: { color: "#FFFFFF", fontSize: 15, fontWeight: "600" as const },
  emptyState: { alignItems: "center", padding: 32 },
  emptyText: { color: E.TEXT_SEC, fontSize: 15, textAlign: "center" },
  retryPill: {
    marginTop: 16,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: "rgba(0,121,107,0.1)",
    borderRadius: 10,
  },
  retryPillText: { color: E.BRAND, fontWeight: "700" as const },
});
