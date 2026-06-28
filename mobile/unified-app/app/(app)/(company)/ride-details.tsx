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
import { AppText, Screen } from "../../../src/design/responsive";
import {
  getCompanyAvailableDrivers,
  getCompanyPartnershipsForTransfer,
  getDispatchApiErrorMessage,
} from "../../../src/features/company/api/companyApi";
import { RideScheduleModal } from "../../../src/features/company/components/rides/RideScheduleModal";
import { AssignDriverModal } from "../../../src/features/company/components/rides/AssignDriverModal";
import { normalizeCompanyEventType } from "../../../src/core/realtime/eventContracts";
import { buildIdentityFromMission } from "../../../src/features/company/utils/bookingIdentity";
import type { CompanyDispatchMission } from "../../../src/features/company/api/contracts";
import { TransferRideModal } from "../../../src/features/company/components/transfers/TransferRideModal";
import {
  RideDetailBillingSection,
  RideDetailDestinationSection,
  RideDetailInfoSection,
  RideDetailRouteSection,
  RideDetailTimelineSection,
} from "../../../src/features/company/components/rides/CompanyRideDetailSections";
import {
  buildRideBillingSummary,
  buildRideDetailInfoRows,
  buildRideTimeline,
  readRideDestinationDetails,
} from "../../../src/features/company/utils/companyRideDetailPresentation";
import {
  EnterpriseActionChip,
  EnterpriseFooterActionRow,
} from "../../../src/features/company/components/EnterpriseActionChip";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { canMarkRideUrgent } from "../../../src/features/company/utils/pickupSentinel";
import { getEnterpriseStatusColors } from "../../../src/features/company/theme/enterpriseStatusColors";
import { createShadow } from "../../../src/styles/shadowStyles";
import { FONT_SIZE } from "../../../src/design/responsive/typographyTokens";
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

function readPassengerLabel(data: Record<string, unknown> | null | undefined): string | null {
  if (!data) return null;
  const identity = data.identity;
  if (identity && typeof identity === "object") {
    const passenger = (identity as { passenger?: { name?: string } }).passenger;
    if (passenger && typeof passenger.name === "string" && passenger.name.trim()) {
      return passenger.name.trim();
    }
  }
  return readClientName(data);
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

function parseNumberishId(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function readDriverLabel(data: Record<string, unknown> | null | undefined): string | null {
  if (!data) return null;
  const direct = data.driver_name;
  if (typeof direct === "string" && direct.trim()) return direct.trim();
  const assigned = data.assigned_driver_name;
  if (typeof assigned === "string" && assigned.trim()) return assigned.trim();
  const driverObj = data.driver;
  if (driverObj && typeof driverObj === "object") {
    const raw = driverObj as Record<string, unknown>;
    const byName = raw.name;
    if (typeof byName === "string" && byName.trim()) return byName.trim();
    const first = typeof raw.first_name === "string" ? raw.first_name.trim() : "";
    const last = typeof raw.last_name === "string" ? raw.last_name.trim() : "";
    const full = [first, last].filter(Boolean).join(" ").trim();
    if (full) return full;
  }
  return null;
}

function readLocationLabel(
  data: Record<string, unknown> | null | undefined,
  kind: "pickup" | "dropoff"
): string | null {
  if (!data) return null;
  const directKeys =
    kind === "pickup"
      ? ["pickup_label", "pickup_location", "pickup_address_label", "from_address"]
      : ["dropoff_label", "dropoff_location", "dropoff_address_label", "to_address"];
  for (const key of directKeys) {
    const raw = data[key];
    if (typeof raw === "string" && raw.trim()) return raw.trim();
  }
  const nested =
    kind === "pickup"
      ? (data.pickup_address as Record<string, unknown> | undefined)
      : (data.dropoff_address as Record<string, unknown> | undefined);
  if (nested) {
    const nestedLabel = nested.label ?? nested.address ?? nested.description;
    if (typeof nestedLabel === "string" && nestedLabel.trim()) return nestedLabel.trim();
  }
  // Fallback deep-scan: certaines réponses détaillées portent les champs
  // dans `booking`, `reservation`, `mission`, `summary`, etc.
  const keys =
    kind === "pickup"
      ? [
          "pickup_label",
          "pickup_location",
          "pickup_address_label",
          "pickup_address",
          "from_address",
        ]
      : [
          "dropoff_label",
          "dropoff_location",
          "dropoff_address_label",
          "dropoff_address",
          "to_address",
        ];
  const queue: unknown[] = [data];
  const seen = new Set<unknown>();
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const raw = current as Record<string, unknown>;
    for (const key of keys) {
      const value = raw[key];
      if (typeof value === "string" && value.trim()) return value.trim();
      if (value && typeof value === "object") {
        const nestedObj = value as Record<string, unknown>;
        const nestedLabel = nestedObj.label ?? nestedObj.address ?? nestedObj.description;
        if (typeof nestedLabel === "string" && nestedLabel.trim()) return nestedLabel.trim();
      }
    }
    for (const value of Object.values(raw)) {
      if (value && typeof value === "object") queue.push(value);
    }
  }
  return null;
}

function readScheduledIso(data: Record<string, unknown> | null | undefined): string | null {
  if (!data) return null;
  const candidates = [data.scheduled_at, data.scheduled_time, data.pickup_at, data.date_time];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  const datePart = typeof data.date === "string" ? data.date.trim() : "";
  const timePart = typeof data.time === "string" ? data.time.trim() : "";
  if (datePart && timePart) return `${datePart}T${timePart}`;
  // Fallback deep-scan pour schémas legacy / imbriqués.
  const queue: unknown[] = [data];
  const seen = new Set<unknown>();
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const raw = current as Record<string, unknown>;
    const candidates = [raw.scheduled_at, raw.scheduled_time, raw.pickup_at, raw.date_time];
    for (const value of candidates) {
      if (typeof value === "string" && value.trim()) return value.trim();
    }
    const d = typeof raw.date === "string" ? raw.date.trim() : "";
    const t = typeof raw.time === "string" ? raw.time.trim() : "";
    if (d && t) return `${d}T${t}`;
    for (const value of Object.values(raw)) {
      if (value && typeof value === "object") queue.push(value);
    }
  }
  return null;
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
  const [scheduleOpen, setScheduleOpen] = useState(false);
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
      const { items } = await getCompanyPartnershipsForTransfer({ contextId });
      setPartners(items);
      setSelectedPartnerId(items[0]?.id ?? null);
      setTransferOpen(true);
    } catch (error) {
      setMutationError(getDispatchApiErrorMessage(error, "Liste partenaires indisponible."));
    } finally {
      setSaving(false);
    }
  };

  const d = (missionQuery.data as Record<string, unknown> | null | undefined) ?? null;
  const statusStr = d ? String((d as { status?: string }).status ?? "pending") : "pending";
  const statusStyle = getEnterpriseStatusColors(statusStr);
  const clientTitle = d ? readPassengerLabel(d) : null;
  const detailIdentity = d
    ? buildIdentityFromMission(d as unknown as CompanyDispatchMission)
    : null;
  const scheduledIso = readScheduledIso(d);
  const titleLine = clientTitle
    ? `${clientTitle} · ${
        scheduledIso
          ? dayjs(String(scheduledIso)).format("DD MMM HH:mm")
          : "⏱️ À définir"
      }`
    : "Course";
  const idLine = d ? String((d as { mission_id?: number }).mission_id ?? rideId ?? "—") : (rideId ?? "—");
  const pickup = readLocationLabel(d, "pickup") ?? "—";
  const drop = readLocationLabel(d, "dropoff") ?? "—";
  const driverId = d
    ? parseNumberishId(
        (d as { driver_id?: unknown; driverId?: unknown; assigned_driver_id?: unknown }).driver_id ??
          (d as { driverId?: unknown }).driverId ??
          (d as { assigned_driver_id?: unknown }).assigned_driver_id
      )
    : null;
  const driverLabel = readDriverLabel(d);
  const hasAssignedDriver = driverId != null || Boolean(driverLabel);
  const driverDisplay = driverLabel ?? (driverId != null ? `#${driverId}` : "Non assigné");
  const showUrgentAction = d ? canMarkRideUrgent(d as unknown as CompanyDispatchMission) : false;
  const billingSummary = d ? buildRideBillingSummary(d, linkedInvoice) : null;
  const destinationDetails = d ? readRideDestinationDetails(d) : null;
  const timelineItems = d ? buildRideTimeline(d, driverLabel) : [];
  const infoRows =
    d && billingSummary
      ? buildRideDetailInfoRows(d, detailIdentity, {
          statusLabel: mapStatusToLabel(statusStr),
          scheduledIso,
          driverDisplay,
          billingSummary,
        })
      : [];

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
            <RideDetailInfoSection rows={infoRows} />
            <RideDetailRouteSection
              pickup={pickup}
              dropoff={drop}
              clinicalLine={destinationDetails?.clinicalLine ?? null}
            />
            {destinationDetails ? (
              <RideDetailDestinationSection destination={destinationDetails} />
            ) : null}
            {billingSummary ? <RideDetailBillingSection billing={billingSummary} /> : null}
            <RideDetailTimelineSection items={timelineItems} />
            <View style={styles.section}>
              <AppText variant="sectionTitle" style={styles.sectionTitle}>
                Actions
              </AppText>
              <View style={styles.actionsWrap}>
                <EnterpriseFooterActionRow>
                  <EnterpriseActionChip
                    icon="person-add-outline"
                    label={saving ? "Affectation…" : hasAssignedDriver ? "Réassigner" : "Assigner"}
                    onPress={() => void openAssignModal()}
                    disabled={!contextId || saving}
                    showSpinner={saving}
                    compact
                  />
                  <EnterpriseActionChip
                    icon="time-outline"
                    label="Planifier"
                    onPress={() => setScheduleOpen(true)}
                    disabled={!contextId || saving}
                    compact
                  />
                  {showUrgentAction ? (
                    <EnterpriseActionChip
                      icon="flash-outline"
                      label={rideActions.urgent.isPending ? "Urgent…" : "Urgent"}
                      tone="urgent"
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
                      compact
                    />
                  ) : null}
                  <EnterpriseActionChip
                    icon="swap-horizontal-outline"
                    label="Transférer"
                    tone="transfer"
                    onPress={() => void openTransferModal()}
                    disabled={!contextId || saving}
                    compact
                  />
                  <EnterpriseActionChip
                    icon="close-circle-outline"
                    label={rideActions.cancel.isPending ? "Annulation…" : "Annuler"}
                    tone="danger"
                    onPress={() =>
                      rideActions.cancel.mutate({
                        missionId: rideIdNumber as number,
                        reasonCode: "cancelled_from_mobile_dispatch",
                        note: "Cancellation requested from company ride-details",
                      })
                    }
                    disabled={!contextId || rideActions.cancel.isPending}
                    compact
                  />
                </EnterpriseFooterActionRow>
              </View>
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
      <AssignDriverModal
        visible={assignOpen}
        pending={saving}
        drivers={drivers}
        selectedDriverId={selectedDriverId}
        error={mutationError}
        onSelect={setSelectedDriverId}
        onConfirm={() => void assignDriver()}
        onClose={() => {
          setAssignOpen(false);
          setMutationError(null);
        }}
        mode={hasAssignedDriver ? "reassign" : "assign"}
      />
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
                partnershipId: selectedPartnerId,
              });
              setTransferOpen(false);
              await missionQuery.refetch();
            } catch (error) {
              setMutationError(getDispatchApiErrorMessage(error, "Transfert impossible."));
            } finally {
              setSaving(false);
            }
          })();
        }}
        onClose={() => !saving && setTransferOpen(false)}
      />
      <RideScheduleModal
        visible={scheduleOpen}
        missionId={rideIdNumber}
        initialScheduledAt={scheduledIso}
        onClose={() => setScheduleOpen(false)}
        onSaved={() => void missionQuery.refetch()}
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
  headerTitle: { color: E.TEXT, fontSize: FONT_SIZE.px16, fontWeight: "700" as const },
  headerSub: { color: E.TEXT_MUTED, fontSize: FONT_SIZE.px12, marginTop: 2 },
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
    borderRadius: 16,
    padding: 16,
    margin: 16,
    marginBottom: 0,
    borderWidth: 1,
    borderColor: E.BORDER,
    ...cardShadow,
  },
  sectionTitle: { color: E.TEXT, marginBottom: 12 },
  actionsWrap: {
    marginTop: 2,
  },
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
  backCtaText: { color: "#FFFFFF", fontSize: FONT_SIZE.px15, fontWeight: "600" as const },
  emptyState: { alignItems: "center", padding: 32 },
  emptyText: { color: E.TEXT_SEC, fontSize: FONT_SIZE.px15, textAlign: "center" },
  retryPill: {
    marginTop: 16,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: "rgba(0,121,107,0.1)",
    borderRadius: 10,
  },
  retryPillText: { color: E.BRAND, fontWeight: "700" as const },
});
