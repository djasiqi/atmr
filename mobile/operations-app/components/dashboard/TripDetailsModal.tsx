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
} from "react-native";
import { getTripDetails, Booking } from "@/services/api";
import { Loader } from "@/components/ui/Loader";
import { styles } from "@/styles/tripDetailsStyles";
import { Ionicons } from "@expo/vector-icons";

type Props = {
  visible: boolean;
  tripId: number | null;
  onClose: () => void;
};

export default function TripDetailsModal({ visible, tripId, onClose }: Props) {
  const [trip, setTrip] = useState<Booking | null>(null);
  const [loading, setLoading] = useState(true);
  const pan = useRef(new Animated.Value(0)).current;
  const overlayOpacity = useRef(new Animated.Value(1)).current;

  const fetchTripDetails = async () => {
    if (!tripId) return;

    setLoading(true);
    try {
      const details = await getTripDetails(tripId);
      setTrip(details);
    } catch {
      Alert.alert("Erreur", "Impossible de charger les détails du trajet.");
      onClose();
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (visible && tripId) {
      fetchTripDetails();
      // Réinitialiser la position du modal et l'opacité de l'overlay
      pan.setValue(0);
      overlayOpacity.setValue(1);
    }
  }, [visible, tripId]);

  // Gestionnaire de swipe down pour fermer
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_, gestureState) => {
        // Activer seulement si on glisse vers le bas (dy > 0)
        return gestureState.dy > 5;
      },
      onPanResponderMove: (_, gestureState) => {
        // Permettre seulement le glissement vers le bas
        if (gestureState.dy > 0) {
          pan.setValue(gestureState.dy);
          // Fade out progressif de l'overlay : transparent à 30% (150px)
          const opacity = Math.max(0, 1 - gestureState.dy / 150);
          overlayOpacity.setValue(opacity);
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        // Si on a glissé plus de 100px vers le bas, fermer le modal
        if (gestureState.dy > 100) {
          // Animation de fermeture smooth avec fade out de l'overlay
          Animated.parallel([
            Animated.timing(pan, {
              toValue: 500,
              duration: 300,
              useNativeDriver: true,
            }),
            Animated.timing(overlayOpacity, {
              toValue: 0,
              duration: 300,
              useNativeDriver: true,
            }),
          ]).start(() => {
            onClose();
          });
        } else {
          // Sinon, revenir à la position initiale
          Animated.parallel([
            Animated.spring(pan, {
              toValue: 0,
              useNativeDriver: true,
            }),
            Animated.spring(overlayOpacity, {
              toValue: 1,
              useNativeDriver: true,
            }),
          ]).start();
        }
      },
    })
  ).current;

  return (
    <Modal
      transparent
      animationType="slide"
      visible={visible}
      onRequestClose={onClose}
    >
      <View style={{ flex: 1, justifyContent: "flex-end" }}>
        {/* Overlay sombre avec animation */}
        <Animated.View
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.5)",
            opacity: overlayOpacity,
          }}
        >
          <TouchableOpacity
            activeOpacity={1}
            onPress={onClose}
            style={{ flex: 1 }}
          />
        </Animated.View>

        {/* Modal qui vient du bas */}
        <Animated.View
          style={{
            backgroundColor: "#FFFFFF",
            borderTopLeftRadius: 20,
            borderTopRightRadius: 20,
            height: "80%",
            maxHeight: "85%",
            shadowColor: "#000",
            shadowOffset: { width: 0, height: -3 },
            shadowOpacity: 0.3,
            shadowRadius: 10,
            elevation: 10,
            transform: [{ translateY: pan }],
          }}
          {...panResponder.panHandlers}
        >
          {/* Indicateur de swipe (petite barre en haut) */}
          <View
            style={{
              alignItems: "center",
              paddingVertical: 8,
            }}
          >
            <View
              style={{
                width: 40,
                height: 5,
                backgroundColor: "#D0D0D0",
                borderRadius: 3,
              }}
            />
          </View>

          {/* Header avec bouton fermer */}
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-between",
              alignItems: "center",
              paddingHorizontal: 16,
              paddingBottom: 16,
              borderBottomWidth: 1,
              borderBottomColor: "#E0E0E0",
            }}
          >
            <Text style={{ fontSize: 18, fontWeight: "700", color: "#104F55" }}>
              Détails du trajet #{tripId}
            </Text>
            <TouchableOpacity onPress={onClose} style={{ padding: 8 }}>
              <Ionicons name="close" size={28} color="#666" />
            </TouchableOpacity>
          </View>

          {loading || !trip ? (
            <View
              style={{
                flex: 1,
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <Loader />
            </View>
          ) : (
            <ScrollView
              style={{ flex: 1 }}
              contentContainerStyle={{ padding: 16 }}
            >
              {/* Client */}
              <View style={styles.section}>
                <View style={styles.rowBetween}>
                  <Text style={styles.label}>Client :</Text>
                  <Text style={styles.value}>{trip.client_name}</Text>
                </View>
              </View>

              {/* Lieux */}
              <View style={styles.section}>
                <View style={styles.rowBetween}>
                  <Text style={styles.label}>De :</Text>
                  <Text style={styles.value}>{trip.pickup_location}</Text>
                </View>
              </View>
              <View style={styles.section}>
                <View style={styles.rowBetween}>
                  <Text style={styles.label}>Vers :</Text>
                  <Text style={styles.value}>{trip.dropoff_location}</Text>
                </View>
              </View>

              {/* Horaires */}
              <View style={styles.section}>
                <View style={styles.rowBetween}>
                  <Text style={styles.label}>Heure prévue :</Text>
                  <Text style={styles.value}>
                    {new Date(trip.scheduled_time).toLocaleString()}
                  </Text>
                </View>
              </View>

              {/* Montant / Distance / Durée */}
              <View style={styles.section}>
                <View style={styles.rowBetween}>
                  <Text style={styles.label}>Montant :</Text>
                  <Text style={styles.value}>
                    {trip.amount?.toFixed(2)} CHF
                  </Text>
                </View>
              </View>

              {trip.distance_meters && (
                <View style={styles.section}>
                  <View style={styles.rowBetween}>
                    <Text style={styles.label}>Distance :</Text>
                    <Text style={styles.value}>
                      {(trip.distance_meters / 1000).toFixed(1)} km
                    </Text>
                  </View>
                </View>
              )}

              {trip.duration_seconds && (
                <View style={styles.section}>
                  <View style={styles.rowBetween}>
                    <Text style={styles.label}>Durée :</Text>
                    <Text style={styles.value}>
                      {Math.ceil(trip.duration_seconds / 60)} min
                    </Text>
                  </View>
                </View>
              )}

              {/* Infos médicales */}
              {(trip.medical_facility ||
                trip.doctor_name ||
                trip.hospital_service) && (
                <View
                  style={{
                    marginTop: 16,
                    padding: 12,
                    backgroundColor: "#E3F2FD",
                    borderRadius: 8,
                  }}
                >
                  <Text
                    style={{
                      fontSize: 14,
                      fontWeight: "700",
                      color: "#004085",
                      marginBottom: 8,
                    }}
                  >
                    🏥 Informations médicales
                  </Text>
                  {trip.medical_facility && (
                    <Text
                      style={{
                        fontSize: 13,
                        color: "#004085",
                        marginBottom: 4,
                      }}
                    >
                      📍 {trip.medical_facility}
                    </Text>
                  )}
                  {trip.doctor_name && (
                    <Text
                      style={{
                        fontSize: 13,
                        color: "#004085",
                        marginBottom: 4,
                      }}
                    >
                      👨‍⚕️ Dr {trip.doctor_name}
                    </Text>
                  )}
                  {trip.hospital_service && (
                    <Text
                      style={{
                        fontSize: 13,
                        color: "#004085",
                        marginBottom: 4,
                      }}
                    >
                      🚪 {trip.hospital_service}
                    </Text>
                  )}
                </View>
              )}

              {/* Options chaise roulante */}
              {(trip.wheelchair_client_has || trip.wheelchair_need) && (
                <View
                  style={{
                    marginTop: 12,
                    padding: 12,
                    backgroundColor: "#FFF3CD",
                    borderRadius: 8,
                  }}
                >
                  {trip.wheelchair_client_has && (
                    <Text
                      style={{
                        fontSize: 14,
                        fontWeight: "700",
                        color: "#856404",
                        marginBottom: 4,
                      }}
                    >
                      ♿ Client en chaise roulante
                    </Text>
                  )}
                  {trip.wheelchair_need && (
                    <Text
                      style={{
                        fontSize: 14,
                        fontWeight: "700",
                        color: "#856404",
                      }}
                    >
                      🏥 Prendre une chaise roulante
                    </Text>
                  )}
                </View>
              )}

              {trip.notes_medical && (
                <View style={{ marginTop: 12 }}>
                  <Text
                    style={{
                      fontSize: 13,
                      color: "#616161",
                      fontStyle: "italic",
                    }}
                  >
                    📝 Notes : {trip.notes_medical}
                  </Text>
                </View>
              )}

              {/* Statut */}
              <View style={[styles.section, { marginTop: 16 }]}>
                <View style={styles.rowBetween}>
                  <Text style={styles.label}>Statut :</Text>
                  <Text style={[styles.value, { color: "#00796B" }]}>
                    {trip.status}
                  </Text>
                </View>
              </View>
            </ScrollView>
          )}
        </Animated.View>
      </View>
    </Modal>
  );
}
