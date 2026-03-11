import React, { useEffect, useState, useRef } from "react";
import {
  View,
  ScrollView,
  Text,
  Alert,
  Modal,
  TouchableOpacity,
  PanResponder,
  Animated,
  Platform,
  Linking,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { getTripDetails, Booking } from "@/services/api";
import { Loader } from "@/components/ui/Loader";
import { styles as s, palette } from "@/styles/tripDetailsStyles";
import { openNavigation } from "@/services/deepLinks";
import { formatTimeLocal } from "@/utils/formatTimeLocal";

type Props = {
  visible: boolean;
  tripId: number | null;
  onClose: () => void;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("fr-CH", {
    weekday: "short",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

function formatTime(iso: string): string {
  return formatTimeLocal(iso);
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds} sec`;
  const m = Math.ceil(seconds / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm > 0 ? `${h}h${String(rm).padStart(2, "0")}` : `${h}h`;
}

function civilityLabel(gender?: string): string {
  if (!gender) return "";
  switch (gender.toUpperCase()) {
    case "HOMME": return "Monsieur";
    case "FEMME": return "Madame";
    default: return "";
  }
}

type StatusInfo = { label: string; color: string };
function statusInfo(raw: string): StatusInfo {
  switch ((raw || "").toUpperCase()) {
    case "ASSIGNED": return { label: "Assignée", color: "#0A7F59" };
    case "EN_ROUTE": return { label: "En route", color: "#D97706" };
    case "IN_PROGRESS": return { label: "Client à bord", color: "#2563EB" };
    case "COMPLETED":
    case "RETURN_COMPLETED": return { label: "Terminée", color: palette.secondary };
    case "CANCELED":
    case "CANCELLED": return { label: "Annulée", color: "#DC2626" };
    default: return { label: raw || "–", color: palette.secondary };
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TripDetailsModal({ visible, tripId, onClose }: Props) {
  const [trip, setTrip] = useState<Booking | null>(null);
  const [loading, setLoading] = useState(true);
  const pan = useRef(new Animated.Value(0)).current;
  const overlayOpacity = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (visible && tripId) {
      setLoading(true);
      pan.setValue(0);
      overlayOpacity.setValue(1);
      getTripDetails(tripId)
        .then(setTrip)
        .catch(() => {
          Alert.alert("Erreur", "Impossible de charger les détails.");
          onClose();
        })
        .finally(() => setLoading(false));
    }
  }, [visible, tripId]);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => false,
      onMoveShouldSetPanResponder: (_, g) => g.dy > 10,
      onPanResponderMove: (_, g) => {
        if (g.dy > 0) {
          pan.setValue(g.dy);
          overlayOpacity.setValue(Math.max(0, 1 - g.dy / 150));
        }
      },
      onPanResponderRelease: (_, g) => {
        if (g.dy > 100) {
          Animated.parallel([
            Animated.timing(pan, { toValue: 500, duration: 250, useNativeDriver: true }),
            Animated.timing(overlayOpacity, { toValue: 0, duration: 250, useNativeDriver: true }),
          ]).start(onClose);
        } else {
          Animated.parallel([
            Animated.spring(pan, { toValue: 0, useNativeDriver: true }),
            Animated.spring(overlayOpacity, { toValue: 1, useNativeDriver: true }),
          ]).start();
        }
      },
    }),
  ).current;

  const callPhone = (phone: string) => {
    if (Platform.OS === "web") {
      (window as any).open(`tel:${phone}`);
    } else {
      Linking.openURL(`tel:${phone}`);
    }
  };

  const st = trip ? statusInfo(trip.status) : null;
  const civility = trip?.client?.gender ? civilityLabel(trip.client.gender) : "";
  const clientDisplay = trip?.client?.full_name || trip?.client_name || trip?.customer_name || "Client";
  const clientPhone = trip?.client?.contact_phone || trip?.client?.phone || trip?.client_phone;

  const hasPickupAccess = !!(trip?.pickup_access_notes || trip?.client?.door_code || trip?.client?.floor || trip?.client?.access_notes);
  const hasDropoffAccess = !!trip?.dropoff_access_notes;
  const hasMedical = !!(trip?.medical_facility || trip?.doctor_name || trip?.hospital_service);
  const hasWheelchair = !!(trip?.wheelchair_client_has || trip?.wheelchair_need);
  const hasNotes = !!(trip?.notes_medical || trip?.notes);
  const isDelivery = trip?.mission_type === "material_delivery";

  return (
    <Modal transparent animationType="slide" visible={visible} onRequestClose={onClose}>
      <View style={{ flex: 1, justifyContent: "flex-end" }}>
        <Animated.View style={[s.overlay, { opacity: overlayOpacity }]}>
          <TouchableOpacity activeOpacity={1} onPress={onClose} style={{ flex: 1 }} />
        </Animated.View>

        <Animated.View style={[s.sheet, { transform: [{ translateY: pan }] }]}>
          {/* Handle — swipe-down zone */}
          <View style={s.handle} {...panResponder.panHandlers}>
            <View style={s.handleBar} />
          </View>

          {/* Header */}
          <View style={s.header}>
            <Text style={s.headerTitle}>
              Détails de la course
            </Text>
            <TouchableOpacity onPress={onClose} style={s.headerClose}>
              <Ionicons name="close" size={22} color={palette.secondary} />
            </TouchableOpacity>
          </View>

          {loading || !trip ? (
            <View style={{ flex: 1, justifyContent: "center", alignItems: "center", minHeight: 200 }}>
              <Loader />
            </View>
          ) : (
            <ScrollView contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>

              {/* ══════ STATUT + HORAIRE ══════ */}
              <View style={s.section}>
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
                  <View>
                    <Text style={s.rowLabel}>Horaire prévu</Text>
                    <Text style={s.rowValue}>
                      {formatDate(trip.scheduled_time)} — {formatTime(trip.scheduled_time)}
                    </Text>
                  </View>
                  <Text style={[s.statusInline, { color: st!.color }]}>{st!.label}</Text>
                </View>

                {(trip.distance_meters || trip.duration_seconds) && (
                  <View style={{ flexDirection: "row", gap: 16, marginTop: 8 }}>
                    {trip.distance_meters != null && trip.distance_meters > 0 && (
                      <Text style={s.rowValueSecondary}>
                        {(trip.distance_meters / 1000).toFixed(1)} km
                      </Text>
                    )}
                    {trip.duration_seconds != null && trip.duration_seconds > 0 && (
                      <Text style={s.rowValueSecondary}>
                        {formatDuration(trip.duration_seconds)}
                      </Text>
                    )}
                    {trip.is_return && (
                      <Text style={[s.rowValueSecondary, { color: palette.alertText }]}>Retour</Text>
                    )}
                  </View>
                )}
              </View>

              {/* ══════ LIVRAISON ══════ */}
              {isDelivery && (
                <View style={s.section}>
                  <Text style={s.sectionTitle}>Livraison</Text>
                  <View style={s.sectionCard}>
                    <Text style={s.rowValue}>
                      {trip.delivery_description || "Livraison de matériel"}
                    </Text>
                  </View>
                </View>
              )}

              {/* ══════ CLIENT ══════ */}
              {!isDelivery && (
                <View style={s.section}>
                  <Text style={s.sectionTitle}>Client</Text>
                  <View style={s.sectionCard}>
                    {civility !== "" && (
                      <Text style={s.rowLabel}>{civility}</Text>
                    )}
                    <Text style={s.rowValue}>{clientDisplay}</Text>

                    {trip.client?.birth_date && (
                      <Text style={s.rowValueSecondary}>
                        Né(e) le {new Date(trip.client.birth_date).toLocaleDateString("fr-CH", {
                          day: "2-digit", month: "2-digit", year: "numeric",
                        })}
                      </Text>
                    )}

                    {clientPhone && (
                      <TouchableOpacity
                        style={{ flexDirection: "row", alignItems: "center", marginTop: 6, gap: 6 }}
                        onPress={() => callPhone(clientPhone)}
                      >
                        <Ionicons name="call-outline" size={14} color={palette.accent} />
                        <Text style={[s.rowValue, { color: palette.accent, fontSize: 13 }]}>
                          {clientPhone}
                        </Text>
                      </TouchableOpacity>
                    )}
                  </View>
                </View>
              )}

              {/* ══════ TRAJET ══════ */}
              <View style={s.section}>
                <Text style={s.sectionTitle}>Trajet</Text>
                <View style={s.routeBlock}>
                  {/* Pickup */}
                  <View style={s.routeRow}>
                    <View style={s.routeIndicator}>
                      <View style={[s.routeDot, { backgroundColor: palette.accent }]} />
                    </View>
                    <View style={s.routeTextWrap}>
                      <Text style={s.rowLabel}>Prise en charge</Text>
                      <Text style={s.rowValue}>{trip.pickup_location || "–"}</Text>
                    </View>
                  </View>

                  {/* Pickup access */}
                  {hasPickupAccess && (
                    <View style={s.accessRow}>
                      <Ionicons name="key-outline" size={12} color={palette.accent} style={{ marginRight: 6, marginTop: 1 }} />
                      <View style={{ flex: 1 }}>
                        {trip.client?.floor && (
                          <Text style={s.accessText}>Étage {trip.client.floor}</Text>
                        )}
                        {trip.client?.door_code && (
                          <Text style={s.accessText}>Code : {trip.client.door_code}</Text>
                        )}
                        {trip.client?.access_notes && (
                          <Text style={s.accessText}>{trip.client.access_notes}</Text>
                        )}
                        {trip.pickup_access_notes && (
                          <Text style={s.accessText}>{trip.pickup_access_notes}</Text>
                        )}
                      </View>
                    </View>
                  )}

                  {/* Line */}
                  <View style={{ alignItems: "flex-start", paddingLeft: 3.25 }}>
                    <View style={s.routeLine} />
                  </View>

                  {/* Dropoff */}
                  <View style={s.routeRow}>
                    <View style={s.routeIndicator}>
                      <View style={[s.routeDotSquare, { backgroundColor: palette.text }]} />
                    </View>
                    <View style={s.routeTextWrap}>
                      <Text style={s.rowLabel}>Destination</Text>
                      <Text style={s.rowValue}>{trip.dropoff_location || "–"}</Text>
                    </View>
                  </View>

                  {/* Dropoff access */}
                  {hasDropoffAccess && (
                    <View style={s.accessRow}>
                      <Ionicons name="key-outline" size={12} color={palette.accent} style={{ marginRight: 6, marginTop: 1 }} />
                      <Text style={[s.accessText, { flex: 1 }]}>{trip.dropoff_access_notes}</Text>
                    </View>
                  )}
                </View>

                {/* Navigate button */}
                <TouchableOpacity
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "center",
                    marginTop: 10,
                    paddingVertical: 10,
                    backgroundColor: "rgba(10,127,89,0.06)",
                    borderRadius: 10,
                    borderWidth: 1,
                    borderColor: "rgba(10,127,89,0.15)",
                    gap: 6,
                  }}
                  onPress={() => {
                    const upper = (trip.status || "").toUpperCase();
                    const dest = upper === "IN_PROGRESS"
                      ? trip.dropoff_location
                      : trip.pickup_location;
                    if (dest) openNavigation(dest);
                  }}
                >
                  <Ionicons name="navigate-outline" size={16} color={palette.accent} />
                  <Text style={{ fontSize: 13, fontWeight: "600", color: palette.accent }}>
                    Ouvrir dans la navigation
                  </Text>
                </TouchableOpacity>
              </View>

              {/* ══════ INFORMATIONS MÉDICALES ══════ */}
              {hasMedical && (
                <View style={s.section}>
                  <Text style={s.sectionTitle}>Informations médicales</Text>
                  <View style={s.sectionCard}>
                    {trip.medical_facility && (
                      <View style={s.row}>
                        <View style={s.rowIcon}>
                          <Ionicons name="business-outline" size={14} color={palette.secondary} />
                        </View>
                        <View style={s.rowContent}>
                          <Text style={s.rowLabel}>Établissement</Text>
                          <Text style={s.rowValue}>{trip.medical_facility}</Text>
                        </View>
                      </View>
                    )}
                    {trip.doctor_name && (
                      <View style={s.row}>
                        <View style={s.rowIcon}>
                          <Ionicons name="person-outline" size={14} color={palette.secondary} />
                        </View>
                        <View style={s.rowContent}>
                          <Text style={s.rowLabel}>Médecin</Text>
                          <Text style={s.rowValue}>Dr {trip.doctor_name}</Text>
                        </View>
                      </View>
                    )}
                    {trip.hospital_service && (
                      <View style={s.row}>
                        <View style={s.rowIcon}>
                          <Ionicons name="medkit-outline" size={14} color={palette.secondary} />
                        </View>
                        <View style={s.rowContent}>
                          <Text style={s.rowLabel}>Service</Text>
                          <Text style={s.rowValue}>{trip.hospital_service}</Text>
                        </View>
                      </View>
                    )}
                    {trip.client?.gp_phone && (
                      <TouchableOpacity
                        style={[s.row, { marginTop: 2 }]}
                        onPress={() => callPhone(trip.client!.gp_phone!)}
                      >
                        <View style={s.rowIcon}>
                          <Ionicons name="call-outline" size={14} color={palette.accent} />
                        </View>
                        <View style={s.rowContent}>
                          <Text style={s.rowLabel}>Médecin traitant</Text>
                          <Text style={[s.rowValue, { color: palette.accent, fontSize: 13 }]}>
                            {trip.client.gp_phone}
                          </Text>
                        </View>
                      </TouchableOpacity>
                    )}
                  </View>
                </View>
              )}

              {/* ══════ PMR / FAUTEUIL ROULANT ══════ */}
              {hasWheelchair && (
                <View style={s.section}>
                  {trip.wheelchair_client_has && (
                    <View style={s.alertBlock}>
                      <Ionicons name="alert-circle-outline" size={16} color={palette.alertText} />
                      <Text style={s.alertText}>Client en fauteuil roulant</Text>
                    </View>
                  )}
                  {trip.wheelchair_need && (
                    <View style={s.alertBlock}>
                      <Ionicons name="alert-circle-outline" size={16} color={palette.alertText} />
                      <Text style={s.alertText}>Fauteuil roulant à prévoir</Text>
                    </View>
                  )}
                </View>
              )}

              {/* ══════ NOTES ══════ */}
              {hasNotes && (
                <View style={s.section}>
                  <Text style={s.sectionTitle}>Notes</Text>
                  <View style={s.sectionCard}>
                    {trip.notes_medical && (
                      <Text style={s.notesText}>{trip.notes_medical}</Text>
                    )}
                    {trip.notes && trip.notes !== trip.notes_medical && (
                      <Text style={[s.notesText, trip.notes_medical ? { marginTop: 6 } : undefined]}>
                        {trip.notes}
                      </Text>
                    )}
                  </View>
                </View>
              )}

              {/* ══════ ENTREPRISE ══════ */}
              {trip.company_name && (
                <View style={s.section}>
                  <Text style={s.sectionTitle}>Entreprise</Text>
                  <Text style={s.rowValueSecondary}>{trip.company_name}</Text>
                </View>
              )}

            </ScrollView>
          )}
        </Animated.View>
      </View>
    </Modal>
  );
}
