import React, { useEffect, useState, useCallback, useMemo } from "react";
import {
  SectionList,
  RefreshControl,
  Alert,
  TouchableOpacity,
  View,
  Text,
  StyleSheet,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useAuth } from "@/hooks/useAuth";
import { getCompletedTrips, getAssignedTrips, getCompanyTodayTrips, Booking } from "@/services/api";
import { isCompletedStatus, isCanceledStatus } from "@/utils/bookingStatus";
import { Loader } from "@/components/ui/Loader";
import TripHeader from "@/components/dashboard/TripHeader";
import { useNotifications } from "@/hooks/useNotifications";
import TripDetailsModal from "@/components/dashboard/TripDetailsModal";
import {
  onBookingNew,
  onBookingUpdated,
  onBookingCancelled,
  onBookingReassigned,
} from "@/services/socket";
import { filterActiveMissions } from "@/utils/missionGrouping";

const BRAND = "#00796B";
const TXT = "#0f172a";
const TXT_SEC = "#6b7280";
const TXT_MUTED = "#9ca3af";
const BORDER = "#e5e7eb";
const CARD = "#FFFFFF";
const BG = "#f4f7fc";

type Tab = "mine" | "team";

function formatHour(iso: string): string {
  return new Date(iso).toLocaleTimeString("fr-CH", { hour: "2-digit", minute: "2-digit" });
}

function shortenAddress(addr: string | undefined): string {
  if (!addr) return "–";
  const comma = addr.indexOf(",");
  const raw = comma > 0 ? addr.substring(0, comma) : addr;
  return raw.trim().substring(0, 40);
}

type StatusMeta = { label: string; color: string; bg: string };

function statusMeta(raw: string): StatusMeta {
  switch ((raw || "").toUpperCase()) {
    case "PENDING":
      return { label: "En attente", color: "#d97706", bg: "rgba(217,119,6,0.08)" };
    case "ASSIGNED":
      return { label: "Assignée", color: BRAND, bg: "rgba(0,121,107,0.08)" };
    case "EN_ROUTE":
      return { label: "En route", color: "#d97706", bg: "rgba(217,119,6,0.08)" };
    case "IN_PROGRESS":
      return { label: "À bord", color: "#00695C", bg: "rgba(0,105,92,0.08)" };
    case "COMPLETED":
    case "RETURN_COMPLETED":
      return { label: "Terminée", color: "#6b7280", bg: "rgba(107,114,128,0.08)" };
    case "CANCELED":
    case "CANCELLED":
      return { label: "Annulée", color: "#dc2626", bg: "rgba(220,38,38,0.08)" };
    default:
      return { label: raw || "–", color: TXT_SEC, bg: "rgba(107,114,128,0.08)" };
  }
}

function categorizeTripByTime(trip: Booking): string {
  const h = new Date(trip.scheduled_time).getHours();
  if (h < 12) return "Matin";
  if (h < 18) return "Après-midi";
  return "Soirée";
}

export default function TripsScreen() {
  useNotifications();
  const { driver } = useAuth();

  const [activeTab, setActiveTab] = useState<Tab>("mine");
  const [completedTrips, setCompletedTrips] = useState<Booking[]>([]);
  const [assignedTrips, setAssignedTrips] = useState<Booking[]>([]);
  const [companyTrips, setCompanyTrips] = useState<Booking[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTripId, setSelectedTripId] = useState<number | null>(null);
  const [modalVisible, setModalVisible] = useState(false);

  const loadTrips = useCallback(async () => {
    if (!driver) {
      setCompletedTrips([]); setAssignedTrips([]); setCompanyTrips([]);
      setLoading(false); setRefreshing(false);
      return;
    }
    try {
      setLoading(true);
      const [completed, assigned, company] = await Promise.all([
        getCompletedTrips(driver.id), getAssignedTrips(), getCompanyTodayTrips(),
      ]);
      const today = new Date().toDateString();
      setCompletedTrips(completed.filter((t) => new Date(t.scheduled_time).toDateString() === today));
      setAssignedTrips(filterActiveMissions(assigned));
      setCompanyTrips(company);
    } catch {
      Alert.alert("Erreur", "Impossible de charger les trajets.");
    } finally {
      setLoading(false); setRefreshing(false);
    }
  }, [driver]);

  useEffect(() => { loadTrips(); }, [loadTrips]);

  useEffect(() => {
    if (!driver) return;
    const unsubs = [
      onBookingNew((booking: Booking) => {
        setAssignedTrips((prev) => {
          const exists = prev.find((b) => b.id === booking.id);
          if (exists) return prev.map((b) => (b.id === booking.id ? booking : b));
          return [...prev, booking].sort((a, b) => new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime());
        });
      }),
      onBookingUpdated((booking: Booking) => {
        const bst = (booking.status || "").toLowerCase();
        setAssignedTrips((prev) => prev.map((b) => (b.id === booking.id ? booking : b)).filter((b) => !isCanceledStatus((b.status || "").toLowerCase())));
        if (isCompletedStatus(bst)) {
          setCompletedTrips((prev) => {
            if (prev.find((b) => b.id === booking.id)) return prev.map((b) => (b.id === booking.id ? booking : b));
            const today = new Date().toDateString();
            if (new Date(booking.scheduled_time).toDateString() === today)
              return [...prev, booking].sort((a, b) => new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime());
            return prev;
          });
        }
        setCompanyTrips((prev) => prev.map((b) => (b.id === booking.id ? booking : b)));
      }),
      onBookingCancelled((data) => {
        const id = typeof data === "object" && "id" in data ? data.id : null;
        if (!id) return;
        setAssignedTrips((prev) => prev.filter((b) => b.id !== id));
        setCompletedTrips((prev) => prev.filter((b) => b.id !== id));
        setCompanyTrips((prev) => prev.filter((b) => b.id !== id));
      }),
      onBookingReassigned(() => loadTrips()),
    ];
    return () => unsubs.forEach((u) => u());
  }, [driver, loadTrips]);

  const onRefresh = useCallback(() => { setRefreshing(true); loadTrips(); }, [loadTrips]);

  const myIds = useMemo(() => new Set([...assignedTrips.map((t) => t.id), ...completedTrips.map((t) => t.id)]), [assignedTrips, completedTrips]);

  const teamTrips = useMemo(() =>
    companyTrips
      .filter((t) => !isCanceledStatus((t.status || "").toUpperCase()))
      .sort((a, b) => new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()),
    [companyTrips]);

  const completedSections = useMemo(() => {
    const grouped = completedTrips.reduce((acc, t) => { const k = categorizeTripByTime(t); (acc[k] ??= []).push(t); return acc; }, {} as Record<string, Booking[]>);
    return Object.entries(grouped).map(([t, d]) => ({ title: `${t} — terminé`, data: d }));
  }, [completedTrips]);

  const mineSections = useMemo(() => {
    const ph = { id: -1, pickup_location: "", dropoff_location: "", scheduled_time: new Date().toISOString(), status: "assigned", client_name: "", client_phone: "", company_id: 0, driver_id: 0, is_return: false, isPlaceholder: true } as Booking & { isPlaceholder: boolean };
    return [
      { title: `À faire${assignedTrips.length > 0 ? ` (${assignedTrips.length})` : ""}`, data: assignedTrips.length > 0 ? assignedTrips : [ph] },
      ...completedSections,
    ];
  }, [assignedTrips, completedSections]);

  const teamSections = useMemo(() => {
    if (teamTrips.length === 0) return [{ title: "Équipe", data: [{ id: -2, isPlaceholder: true } as any] }];
    const grouped = teamTrips.reduce((acc, t) => { const k = categorizeTripByTime(t); (acc[k] ??= []).push(t); return acc; }, {} as Record<string, Booking[]>);
    return Object.entries(grouped).map(([t, d]) => ({ title: t, data: d }));
  }, [teamTrips]);

  const summary = useMemo(() => ({
    total: assignedTrips.length + completedTrips.length,
    done: completedTrips.length,
    remaining: assignedTrips.length,
  }), [assignedTrips, completedTrips]);

  const sections = activeTab === "mine" ? mineSections : teamSections;

  const renderCard = useCallback((trip: Booking & { isPlaceholder?: boolean }, isMine: boolean) => {
    if (trip.isPlaceholder) {
      return (
        <View style={c.emptyCard}>
          <Ionicons name={isMine ? "checkmark-circle-outline" : "people-outline"} size={24} color={TXT_MUTED} />
          <Text style={c.emptyTitle}>{isMine ? "Aucune course à faire" : "Aucun transport collègue"}</Text>
          <Text style={c.emptySub}>{isMine ? "Vous serez notifié dès qu'une mission vous sera attribuée." : "Aucun transport en cours pour votre équipe."}</Text>
        </View>
      );
    }

    const meta = statusMeta(trip.status);
    const client = trip.client?.full_name || trip.client_name || trip.customer_name || "Client";
    const driverName = (trip as any).driver_name;
    const isCompleted = isCompletedStatus(trip.status);

    return (
      <TouchableOpacity
        style={[c.card, isCompleted && { opacity: 0.6 }]}
        activeOpacity={0.65}
        onPress={() => { setSelectedTripId(trip.id); setModalVisible(true); }}
      >
        <View style={[c.bar, { backgroundColor: meta.color }]} />
        <View style={c.body}>
          {/* Row 1 */}
          <View style={c.row1}>
            <Text style={c.time}>{formatHour(trip.scheduled_time)}</Text>
            <Text style={c.client} numberOfLines={1}>{client}</Text>
            <View style={[c.statusPill, { backgroundColor: meta.bg }]}>
              <Text style={[c.statusLabel, { color: meta.color }]}>{meta.label}</Text>
            </View>
          </View>

          {/* Route */}
          <View style={c.route}>
            <View style={c.timeline}>
              <View style={c.dotA} />
              <View style={c.connector} />
              <View style={c.dotB} />
            </View>
            <View style={c.addresses}>
              <Text style={c.addr} numberOfLines={1}>{shortenAddress(trip.pickup_location)}</Text>
              <Text style={c.addr} numberOfLines={1}>{shortenAddress(trip.dropoff_location)}</Text>
            </View>
          </View>

          {/* Badges — only show if there are any */}
          {((!isMine) || trip.is_return || trip.wheelchair_client_has || trip.wheelchair_need || (trip.distance_meters != null && trip.distance_meters > 0)) && (
            <View style={c.badges}>
              {!isMine && driverName && (
                <View style={c.badge}>
                  <Ionicons name="person-outline" size={9} color={BRAND} />
                  <Text style={[c.badgeLabel, { color: BRAND }]}>{driverName}</Text>
                </View>
              )}
              {!isMine && !driverName && (
                <View style={[c.badge, { borderColor: "rgba(217,119,6,0.2)", backgroundColor: "rgba(217,119,6,0.04)" }]}>
                  <Text style={[c.badgeLabel, { color: "#92400E" }]}>Non assigné</Text>
                </View>
              )}
              {trip.is_return && (
                <View style={[c.badge, { borderColor: "rgba(217,119,6,0.2)", backgroundColor: "rgba(217,119,6,0.04)" }]}>
                  <Ionicons name="repeat-outline" size={9} color="#92400E" />
                  <Text style={[c.badgeLabel, { color: "#92400E" }]}>Retour</Text>
                </View>
              )}
              {(trip.wheelchair_client_has || trip.wheelchair_need) && (
                <View style={[c.badge, { borderColor: "rgba(217,119,6,0.2)", backgroundColor: "rgba(217,119,6,0.04)" }]}>
                  <Text style={[c.badgeLabel, { color: "#92400E" }]}>PMR</Text>
                </View>
              )}
              {trip.distance_meters != null && trip.distance_meters > 0 && (
                <Text style={c.dist}>{(trip.distance_meters / 1000).toFixed(1)} km</Text>
              )}
            </View>
          )}
        </View>
        <View style={c.chevron}>
          <Ionicons name="chevron-forward" size={14} color={TXT_MUTED} />
        </View>
      </TouchableOpacity>
    );
  }, []);

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: BG }}>
        <Loader />
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: BG }}>
      <TripHeader
        date={new Date().toLocaleDateString("fr-CH", { weekday: "long", day: "numeric", month: "long" })}
        totalTrips={summary.total}
        doneTrips={summary.done}
        remainingTrips={summary.remaining}
      />

      {/* Tabs */}
      <View style={c.tabBar}>
        {(["mine", "team"] as Tab[]).map((tab) => {
          const active = activeTab === tab;
          const count = tab === "mine" ? assignedTrips.length : teamTrips.length;
          const label = tab === "mine" ? "Mes courses" : "Équipe";
          return (
            <TouchableOpacity
              key={tab}
              style={[c.tab, active && c.tabActive]}
              onPress={() => setActiveTab(tab)}
              activeOpacity={0.7}
            >
              <Text style={[c.tabLabel, active && c.tabLabelActive]}>{label}</Text>
              {count > 0 && (
                <View style={[c.tabBadge, active && c.tabBadgeActive]}>
                  <Text style={[c.tabBadgeText, active && c.tabBadgeTextActive]}>{count}</Text>
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </View>

      <SectionList
        sections={sections}
        keyExtractor={(item) => item?.id != null ? String(item.id) : `ph-${Math.random().toString(36).substr(2, 9)}`}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[BRAND]} tintColor={BRAND} />}
        renderSectionHeader={({ section }) => (
          <View style={c.secWrap}>
            <Text style={c.secTitle}>{section.title}</Text>
          </View>
        )}
        renderItem={({ item }) => renderCard(item as any, activeTab === "mine")}
        contentContainerStyle={{ paddingBottom: 90, paddingTop: 2 }}
        stickySectionHeadersEnabled={false}
        ListEmptyComponent={() => (
          <View style={{ marginTop: 40, alignItems: "center" }}>
            <Text style={{ color: TXT_SEC, fontSize: 14 }}>Aucun trajet prévu.</Text>
          </View>
        )}
      />

      <TripDetailsModal visible={modalVisible} tripId={selectedTripId} onClose={() => { setModalVisible(false); setSelectedTripId(null); }} />
    </View>
  );
}

const shadow = Platform.OS === "web"
  ? { boxShadow: "0 1px 4px rgba(0,0,0,0.04)" }
  : { shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 4, elevation: 1 };

const c = StyleSheet.create({
  /* Tabs */
  tabBar: { flexDirection: "row", paddingHorizontal: 20, paddingTop: 10, paddingBottom: 4, gap: 8, backgroundColor: BG },
  tab: {
    flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6,
    paddingVertical: 9, borderRadius: 8,
    backgroundColor: CARD, borderWidth: 1, borderColor: BORDER,
  },
  tabActive: { backgroundColor: BRAND, borderColor: BRAND },
  tabLabel: { fontSize: 13, fontWeight: "600", color: TXT_SEC },
  tabLabelActive: { color: "#fff" },
  tabBadge: { backgroundColor: "rgba(0,0,0,0.06)", borderRadius: 6, minWidth: 18, paddingHorizontal: 4, paddingVertical: 1, alignItems: "center" },
  tabBadgeActive: { backgroundColor: "rgba(255,255,255,0.25)" },
  tabBadgeText: { fontSize: 10, fontWeight: "700", color: TXT_SEC },
  tabBadgeTextActive: { color: "#fff" },

  /* Section */
  secWrap: { paddingHorizontal: 22, paddingTop: 12, paddingBottom: 4 },
  secTitle: { fontSize: 11, fontWeight: "700", color: TXT_SEC, letterSpacing: 0.4, textTransform: "uppercase" },

  /* Card */
  card: {
    backgroundColor: CARD, borderRadius: 12, marginHorizontal: 16, marginVertical: 3,
    borderWidth: 1, borderColor: BORDER, overflow: "hidden", flexDirection: "row", ...shadow,
  },
  bar: { width: 3 },
  body: { flex: 1, paddingVertical: 10, paddingLeft: 12, paddingRight: 4 },
  row1: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 6 },
  time: { fontSize: 13, fontWeight: "700", color: TXT, minWidth: 40, letterSpacing: -0.2 },
  client: { fontSize: 13, fontWeight: "600", color: TXT, flex: 1 },
  statusPill: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4 },
  statusLabel: { fontSize: 10, fontWeight: "700", letterSpacing: 0.2, textTransform: "uppercase" },

  /* Route */
  route: { flexDirection: "row", gap: 8, marginBottom: 4 },
  timeline: { alignItems: "center", width: 10, paddingTop: 3 },
  dotA: { width: 6, height: 6, borderRadius: 3, backgroundColor: BRAND },
  connector: { width: 1.5, flex: 1, backgroundColor: "#e5e7eb", marginVertical: 2, minHeight: 6 },
  dotB: { width: 6, height: 6, borderRadius: 1.5, backgroundColor: TXT },
  addresses: { flex: 1, justifyContent: "space-between", gap: 2 },
  addr: { fontSize: 12, fontWeight: "500", color: TXT_SEC, lineHeight: 16 },

  /* Badges */
  badges: { flexDirection: "row", alignItems: "center", flexWrap: "wrap", gap: 4, marginTop: 4 },
  badge: { flexDirection: "row", alignItems: "center", paddingVertical: 1, paddingHorizontal: 5, borderRadius: 4, borderWidth: 1, borderColor: BORDER, backgroundColor: "rgba(0,121,107,0.04)", gap: 3 },
  badgeLabel: { fontSize: 9, fontWeight: "600", letterSpacing: 0.1 },
  dist: { fontSize: 10, fontWeight: "500", color: TXT_MUTED, marginLeft: "auto" },

  /* Chevron */
  chevron: { justifyContent: "center", paddingRight: 8 },

  /* Empty */
  emptyCard: { backgroundColor: CARD, borderRadius: 12, paddingVertical: 24, paddingHorizontal: 20, marginHorizontal: 16, marginVertical: 3, alignItems: "center", borderWidth: 1, borderColor: BORDER, borderStyle: "dashed", gap: 6 },
  emptyTitle: { fontSize: 14, fontWeight: "600", color: TXT, textAlign: "center" },
  emptySub: { fontSize: 12, color: TXT_SEC, textAlign: "center", lineHeight: 17, maxWidth: 260 },
});
