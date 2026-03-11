import React, { useEffect, useState } from "react";
import {
  View,
  ScrollView,
  Text,
  Alert,
  Platform,
  TouchableOpacity,
  StyleSheet,
  Linking,
} from "react-native";
import { useLocalSearchParams, router } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import {
  getTripDetails,
  Booking,
  BookingStatus,
} from "@/services/api";
import { Loader } from "@/components/ui/Loader";
import { MissionStateManager, type MissionBarStatus } from "@/services/missionState";
import { dismissMissionNotification, showMissionNotification } from "@/services/missionBarAndroid";
import {
  buildBgTrackingInputs,
  refreshBackgroundTrackingNotification,
} from "@/services/locationTracker";
import { openNavigation } from "@/services/deepLinks";
import { createShadow } from "@/styles/shadowStyles";
import { getLogger } from "@/utils/logger";
import { formatTimeLocal } from "@/utils/formatTimeLocal";

const log = getLogger("TripDetail");

const BRAND = "#00796B";
const TXT = "#1E293B";
const TXT_SEC = "#64748B";
const TXT_MUTED = "#94A3B8";
const BORDER = "#E2E8F0";
const BG = "#F8FAFC";
const CARD = "#FFFFFF";
const DANGER = "#EF4444";
const WARNING_BG = "rgba(245,158,11,0.06)";
const WARNING_BORDER = "rgba(245,158,11,0.15)";
const WARNING_TXT = "#92400E";

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("fr-CH", {
    weekday: "short", day: "numeric", month: "long", year: "numeric",
  });
}
function fmtTime(iso: string): string {
  return formatTimeLocal(iso);
}
function fmtDuration(sec: number): string {
  const m = Math.ceil(sec / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm > 0 ? `${h}h${String(rm).padStart(2, "0")}` : `${h}h`;
}
function civility(gender?: string): string {
  if (!gender) return "";
  const g = gender.toUpperCase();
  if (g === "FEMME" || g === "FEMALE") return "Madame";
  if (g === "HOMME" || g === "MALE") return "Monsieur";
  return "";
}
function statusInfo(raw: string): { label: string; color: string } {
  switch ((raw || "").toUpperCase()) {
    case "ASSIGNED": return { label: "Assignée", color: "#2563EB" };
    case "EN_ROUTE": return { label: "En route", color: "#7C3AED" };
    case "IN_PROGRESS": return { label: "En cours", color: BRAND };
    case "COMPLETED":
    case "RETURN_COMPLETED": return { label: "Terminée", color: "#16A34A" };
    case "CANCELED":
    case "CANCELLED": return { label: "Annulée", color: DANGER };
    default: return { label: raw || "—", color: TXT_MUTED };
  }
}

function callPhone(phone: string) {
  if (Platform.OS === "web") {
    (window as any).open(`tel:${phone}`);
  } else {
    Linking.openURL(`tel:${phone}`);
  }
}

export default function TripDetailsScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [trip, setTrip] = useState<Booking | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchTripDetails = async () => {
    setLoading(true);
    try {
      const details = await getTripDetails(Number(id));
      setTrip(details);
    } catch {
      Alert.alert("Erreur", "Impossible de charger les détails du trajet.");
      router.back();
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) fetchTripDetails();
  }, [id]);

  const handleUpdateStatus = async (targetStatus: MissionBarStatus) => {
    if (!trip) return;
    setLoading(true);
    try {
      const isManagedByMission =
        MissionStateManager.isActive() &&
        MissionStateManager.getState().activeMission?.id === trip.id;

      if (isManagedByMission) {
        const ok = await MissionStateManager.requestTransition(targetStatus);
        if (!ok) {
          Alert.alert("Erreur", "Transition de statut non autorisée.");
          return;
        }
        if (Platform.OS !== "web") {
          const inputs = await buildBgTrackingInputs({
            isAuthenticated: true,
            role: "driver",
            hasActiveMission: MissionStateManager.isActive(),
          });
          const refreshed = await refreshBackgroundTrackingNotification(inputs);
          if (!refreshed) {
            await showMissionNotification(MissionStateManager.getState());
          }
        }
        if (targetStatus === "COMPLETED") {
          await MissionStateManager.stopMission();
          if (Platform.OS !== "web") await dismissMissionNotification();
        }
      } else {
        const { updateTripStatus } = await import("@/services/api");
        await updateTripStatus(trip.id, targetStatus as BookingStatus);
      }
      await fetchTripDetails();
      Alert.alert("Succès", "Statut mis à jour.");
    } catch {
      Alert.alert("Erreur", "Impossible de mettre à jour le statut.");
    } finally {
      setLoading(false);
    }
  };

  if (loading || !trip) {
    return (
      <View style={s.loadingWrap}>
        <Loader />
      </View>
    );
  }

  const st = statusInfo(trip.status);
  const civ = civility(trip.client?.gender);
  const clientDisplay = trip.client?.full_name || trip.client_name || trip.customer_name || "Client";
  const clientPhone = trip.client?.contact_phone || trip.client?.phone || trip.client_phone;
  const isDelivery = trip.mission_type === "material_delivery";
  const normalized = (trip.status || "").toUpperCase();

  const hasPickupAccess = !!(trip.pickup_access_notes || trip.client?.door_code || trip.client?.floor || trip.client?.access_notes);
  const hasDropoffAccess = !!trip.dropoff_access_notes;
  const hasMedical = !!(trip.medical_facility || trip.doctor_name || trip.hospital_service);
  const hasWheelchair = !!(trip.wheelchair_client_has || trip.wheelchair_need);
  const hasNotes = !!(trip.notes_medical || trip.notes);

  return (
    <View style={s.container}>
      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={() => router.back()} style={s.backBtn}>
          <Ionicons name="arrow-back" size={22} color={TXT} />
        </TouchableOpacity>
        <View style={{ flex: 1 }}>
          <Text style={s.headerTitle}>Détails de la course</Text>
          <Text style={s.headerSub}>#{trip.id}</Text>
        </View>
        <View style={[s.statusBadge, { backgroundColor: `${st.color}14` }]}>
          <View style={[s.statusDot, { backgroundColor: st.color }]} />
          <Text style={[s.statusLabel, { color: st.color }]}>{st.label}</Text>
        </View>
      </View>

      <ScrollView style={s.scroll} contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Statut + Horaire */}
        <View style={s.card}>
          <View style={s.cardHeader}>
            <Ionicons name="time-outline" size={16} color={BRAND} />
            <Text style={s.cardTitle}>Horaire</Text>
          </View>
          <Text style={s.rowValue}>{fmtDate(trip.scheduled_time)} — {fmtTime(trip.scheduled_time)}</Text>
          <View style={s.metaRow}>
            {trip.distance_meters != null && trip.distance_meters > 0 && (
              <Text style={s.metaChip}>{(trip.distance_meters / 1000).toFixed(1)} km</Text>
            )}
            {trip.duration_seconds != null && trip.duration_seconds > 0 && (
              <Text style={s.metaChip}>{fmtDuration(trip.duration_seconds)}</Text>
            )}
            {trip.is_return && <Text style={[s.metaChip, { color: WARNING_TXT }]}>Retour</Text>}
          </View>
          {trip.return_time && (
            <Text style={s.rowSec}>Retour prévu : {fmtDate(trip.return_time)} — {fmtTime(trip.return_time)}</Text>
          )}
        </View>

        {/* Livraison */}
        {isDelivery && (
          <View style={s.card}>
            <View style={s.cardHeader}>
              <Ionicons name="cube-outline" size={16} color="#B45309" />
              <Text style={s.cardTitle}>Livraison</Text>
            </View>
            <Text style={s.rowValue}>{trip.delivery_description || "Livraison de matériel"}</Text>
          </View>
        )}

        {/* Client */}
        {!isDelivery && (
          <View style={s.card}>
            <View style={s.cardHeader}>
              <Ionicons name="person-outline" size={16} color={BRAND} />
              <Text style={s.cardTitle}>Client</Text>
            </View>
            {civ !== "" && <Text style={s.rowSec}>{civ}</Text>}
            <Text style={s.rowValue}>{clientDisplay}</Text>
            {trip.client?.birth_date && (
              <Text style={s.rowSec}>
                Né(e) le {new Date(trip.client.birth_date).toLocaleDateString("fr-CH", { day: "2-digit", month: "2-digit", year: "numeric" })}
              </Text>
            )}
            {clientPhone && (
              <TouchableOpacity style={s.phoneRow} onPress={() => callPhone(clientPhone)}>
                <Ionicons name="call-outline" size={14} color={BRAND} />
                <Text style={s.phoneText}>{clientPhone}</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {/* Trajet */}
        <View style={s.card}>
          <View style={s.cardHeader}>
            <Ionicons name="navigate-outline" size={16} color={BRAND} />
            <Text style={s.cardTitle}>Trajet</Text>
          </View>

          {/* Pickup */}
          <View style={s.routeRow}>
            <View style={s.routeIndicator}>
              <View style={[s.routeDot, { backgroundColor: BRAND }]} />
              <View style={s.routeLine} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={s.routeLabel}>Prise en charge</Text>
              <Text style={s.rowValue}>{trip.pickup_location || "—"}</Text>
            </View>
          </View>

          {hasPickupAccess && (
            <View style={s.accessBlock}>
              <Ionicons name="key-outline" size={12} color={BRAND} style={{ marginTop: 2 }} />
              <View style={{ flex: 1 }}>
                {trip.client?.floor && <Text style={s.accessText}>Étage {trip.client.floor}</Text>}
                {trip.client?.door_code && <Text style={s.accessText}>Code : {trip.client.door_code}</Text>}
                {trip.client?.access_notes && <Text style={s.accessText}>{trip.client.access_notes}</Text>}
                {trip.pickup_access_notes && <Text style={s.accessText}>{trip.pickup_access_notes}</Text>}
              </View>
            </View>
          )}

          {/* Dropoff */}
          <View style={s.routeRow}>
            <View style={s.routeIndicator}>
              <View style={[s.routeDotSquare, { backgroundColor: TXT }]} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={s.routeLabel}>Destination</Text>
              <Text style={s.rowValue}>{trip.dropoff_location || "—"}</Text>
            </View>
          </View>

          {hasDropoffAccess && (
            <View style={s.accessBlock}>
              <Ionicons name="key-outline" size={12} color={BRAND} style={{ marginTop: 2 }} />
              <Text style={[s.accessText, { flex: 1 }]}>{trip.dropoff_access_notes}</Text>
            </View>
          )}

          {/* Navigate button */}
          <TouchableOpacity
            style={s.navBtn}
            onPress={() => {
              const dest = normalized === "IN_PROGRESS" ? trip.dropoff_location : trip.pickup_location;
              if (dest) openNavigation(dest);
            }}
          >
            <Ionicons name="navigate-outline" size={16} color={BRAND} />
            <Text style={s.navBtnText}>Ouvrir dans la navigation</Text>
          </TouchableOpacity>
        </View>

        {/* Informations médicales */}
        {hasMedical && (
          <View style={s.card}>
            <View style={s.cardHeader}>
              <Ionicons name="medkit-outline" size={16} color={BRAND} />
              <Text style={s.cardTitle}>Informations médicales</Text>
            </View>
            {trip.medical_facility && (
              <View style={s.infoRow}>
                <Ionicons name="business-outline" size={14} color={TXT_SEC} />
                <View style={{ flex: 1 }}>
                  <Text style={s.infoLabel}>Établissement</Text>
                  <Text style={s.rowValue}>{trip.medical_facility}</Text>
                </View>
              </View>
            )}
            {trip.hospital_service && (
              <View style={s.infoRow}>
                <Ionicons name="medkit-outline" size={14} color={TXT_SEC} />
                <View style={{ flex: 1 }}>
                  <Text style={s.infoLabel}>Service</Text>
                  <Text style={s.rowValue}>{trip.hospital_service}</Text>
                </View>
              </View>
            )}
            {trip.doctor_name && (
              <View style={s.infoRow}>
                <Ionicons name="person-outline" size={14} color={TXT_SEC} />
                <View style={{ flex: 1 }}>
                  <Text style={s.infoLabel}>Médecin</Text>
                  <Text style={s.rowValue}>Dr {trip.doctor_name}</Text>
                </View>
              </View>
            )}
            {trip.client?.gp_phone && (
              <TouchableOpacity style={s.infoRow} onPress={() => callPhone(trip.client!.gp_phone!)}>
                <Ionicons name="call-outline" size={14} color={BRAND} />
                <View style={{ flex: 1 }}>
                  <Text style={s.infoLabel}>Médecin traitant</Text>
                  <Text style={[s.rowValue, { color: BRAND, fontSize: 13 }]}>{trip.client.gp_phone}</Text>
                </View>
              </TouchableOpacity>
            )}
          </View>
        )}

        {/* Chaise roulante */}
        {hasWheelchair && (
          <View style={s.card}>
            {trip.wheelchair_client_has && (
              <View style={s.alertBlock}>
                <Ionicons name="accessibility-outline" size={16} color={WARNING_TXT} />
                <Text style={s.alertText}>Client en fauteuil roulant</Text>
              </View>
            )}
            {trip.wheelchair_need && (
              <View style={s.alertBlock}>
                <Ionicons name="alert-circle-outline" size={16} color={WARNING_TXT} />
                <Text style={s.alertText}>Fauteuil roulant à prévoir</Text>
              </View>
            )}
          </View>
        )}

        {/* Notes */}
        {hasNotes && (
          <View style={s.card}>
            <View style={s.cardHeader}>
              <Ionicons name="document-text-outline" size={16} color={BRAND} />
              <Text style={s.cardTitle}>Notes</Text>
            </View>
            {trip.notes_medical && <Text style={s.notesText}>{trip.notes_medical}</Text>}
            {trip.notes && trip.notes !== trip.notes_medical && (
              <Text style={[s.notesText, trip.notes_medical ? { marginTop: 6 } : undefined]}>{trip.notes}</Text>
            )}
          </View>
        )}

        {/* Entreprise */}
        {trip.company_name && (
          <View style={s.card}>
            <View style={s.cardHeader}>
              <Ionicons name="briefcase-outline" size={16} color={BRAND} />
              <Text style={s.cardTitle}>Entreprise</Text>
            </View>
            <Text style={s.rowSec}>{trip.company_name}</Text>
          </View>
        )}

        {/* Actions */}
        <View style={s.actionsCard}>
          {normalized === "ASSIGNED" && (
            <TouchableOpacity style={s.actionBtnPrimary} onPress={() => handleUpdateStatus("EN_ROUTE")}>
              <Ionicons name="walk-outline" size={18} color="#FFF" />
              <Text style={s.actionBtnPrimaryText}>En route</Text>
            </TouchableOpacity>
          )}
          {normalized === "EN_ROUTE" && (
            <TouchableOpacity style={s.actionBtnPrimary} onPress={() => handleUpdateStatus("IN_PROGRESS")}>
              <Ionicons name={isDelivery ? "cube-outline" : "person-outline"} size={18} color="#FFF" />
              <Text style={s.actionBtnPrimaryText}>{isDelivery ? "Colis récupéré" : "Client à bord"}</Text>
            </TouchableOpacity>
          )}
          {normalized === "IN_PROGRESS" && (
            <TouchableOpacity style={s.actionBtnPrimary} onPress={() => handleUpdateStatus("COMPLETED")}>
              <Ionicons name="checkmark-done-outline" size={18} color="#FFF" />
              <Text style={s.actionBtnPrimaryText}>{trip.is_return ? "Terminer retour" : "Terminer"}</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity style={s.actionBtnSecondary} onPress={() => router.back()}>
            <Ionicons name="arrow-back-outline" size={18} color={TXT_SEC} />
            <Text style={s.actionBtnSecondaryText}>Retour</Text>
          </TouchableOpacity>
        </View>

        <View style={{ height: 30 }} />
      </ScrollView>
    </View>
  );
}

const cardShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.06,
  shadowRadius: 8,
  elevation: 2,
});

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: BG },
  loadingWrap: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: BG },

  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingHorizontal: 20,
    paddingTop: Platform.OS === "ios" ? 56 : 16,
    paddingBottom: 14,
    backgroundColor: CARD,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  backBtn: { padding: 4 },
  headerTitle: { fontSize: 16, fontWeight: "700", color: TXT },
  headerSub: { fontSize: 12, color: TXT_SEC, marginTop: 2 },
  statusBadge: { flexDirection: "row", alignItems: "center", gap: 5, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8 },
  statusDot: { width: 7, height: 7, borderRadius: 4 },
  statusLabel: { fontSize: 12, fontWeight: "600" },

  scroll: { flex: 1 },
  scrollContent: { padding: 16, gap: 12 },

  card: {
    backgroundColor: CARD,
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: BORDER,
    ...cardShadow,
  },
  cardHeader: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 10 },
  cardTitle: { fontSize: 14, fontWeight: "700", color: TXT },

  rowValue: { fontSize: 14, fontWeight: "500", color: TXT },
  rowSec: { fontSize: 13, color: TXT_SEC, marginTop: 2 },

  metaRow: { flexDirection: "row", gap: 12, marginTop: 6 },
  metaChip: { fontSize: 12, fontWeight: "600", color: TXT_SEC },

  phoneRow: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 8 },
  phoneText: { fontSize: 13, fontWeight: "600", color: BRAND },

  // Route
  routeRow: { flexDirection: "row", gap: 10, marginBottom: 2 },
  routeIndicator: { alignItems: "center", width: 10, paddingTop: 4 },
  routeDot: { width: 8, height: 8, borderRadius: 4 },
  routeDotSquare: { width: 8, height: 8, borderRadius: 2 },
  routeLine: { width: 2, flex: 1, backgroundColor: BORDER, marginVertical: 4, minHeight: 20 },
  routeLabel: { fontSize: 11, fontWeight: "600", color: TXT_MUTED, textTransform: "uppercase", letterSpacing: 0.4, marginBottom: 2 },

  accessBlock: { flexDirection: "row", gap: 6, marginLeft: 20, marginBottom: 8, paddingVertical: 6, paddingHorizontal: 10, backgroundColor: "rgba(0,121,107,0.04)", borderRadius: 8, borderWidth: 1, borderColor: "rgba(0,121,107,0.08)" },
  accessText: { fontSize: 12, color: TXT_SEC, lineHeight: 17 },

  navBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, marginTop: 12, paddingVertical: 10, backgroundColor: "rgba(0,121,107,0.06)", borderRadius: 10, borderWidth: 1, borderColor: "rgba(0,121,107,0.12)" },
  navBtnText: { fontSize: 13, fontWeight: "600", color: BRAND },

  // Info rows
  infoRow: { flexDirection: "row", alignItems: "flex-start", gap: 10, paddingVertical: 6 },
  infoLabel: { fontSize: 11, fontWeight: "600", color: TXT_MUTED, textTransform: "uppercase", letterSpacing: 0.3, marginBottom: 1 },

  // Alert (wheelchair)
  alertBlock: { flexDirection: "row", alignItems: "center", gap: 8, paddingVertical: 10, paddingHorizontal: 12, backgroundColor: WARNING_BG, borderRadius: 10, borderWidth: 1, borderColor: WARNING_BORDER, marginBottom: 6 },
  alertText: { fontSize: 13, fontWeight: "600", color: WARNING_TXT },

  // Notes
  notesText: { fontSize: 13, color: TXT_SEC, lineHeight: 19 },

  // Actions
  actionsCard: { gap: 10 },
  actionBtnPrimary: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: BRAND,
    paddingVertical: 14,
    borderRadius: 12,
    ...createShadow({ shadowColor: BRAND, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6, elevation: 3 }),
  },
  actionBtnPrimaryText: { fontSize: 15, fontWeight: "700", color: "#FFF" },
  actionBtnSecondary: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: CARD,
    paddingVertical: 13,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: BORDER,
  },
  actionBtnSecondaryText: { fontSize: 14, fontWeight: "600", color: TXT_SEC },
});
