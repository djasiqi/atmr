import React, { useState, useEffect } from "react";
import {
  View,
  ScrollView,
  Alert,
  TouchableOpacity,
  Image,
  StyleSheet,
  Modal,
} from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { updateDriverProfile, updateDriverPhoto } from "@/services/api";
import { InputField } from "@/components/ui/InputField";
import { Button } from "@/components/ui/Button";
import { Loader } from "@/components/ui/Loader";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";

// Import direct de l'image par défaut
import DefaultDriver from "../../assets/images/default-driver.png";

export default function ProfileScreen() {
  const { driver, refreshProfile, logout } = useAuth();
  const [form, setForm] = useState({
    vehicle_assigned: "",
    brand: "",
    license_plate: "",
    phone: "",
    address: "",
    email: "",
    photo: "",
  });
  const [profileLoading, setProfileLoading] = useState(false);
  const [photoLoading, setPhotoLoading] = useState(false);
  const [photoModalVisible, setPhotoModalVisible] = useState(false);
  const [logoutModalVisible, setLogoutModalVisible] = useState(false);

  useEffect(() => {
    if (driver) {
      setForm({
        vehicle_assigned: driver.vehicle_assigned,
        brand: driver.brand,
        license_plate: driver.license_plate,
        phone: driver.phone,
        address: (driver as any).address || "",
        email: (driver as any).email || "",
        photo: driver.photo,
      });
    }
  }, [driver]);

  const handleSaveProfile = async () => {
    setProfileLoading(true);
    try {
      await updateDriverProfile({
        vehicle_assigned: form.vehicle_assigned,
        brand: form.brand,
        license_plate: form.license_plate,
        phone: form.phone,
        // Note: address et email peuvent être ajoutés plus tard si l'API les supporte
      });
      await refreshProfile();
      Alert.alert("Succès", "Votre profil a été mis à jour.");
    } catch (error) {
      Alert.alert("Erreur", "Échec de la mise à jour du profil.");
    } finally {
      setProfileLoading(false);
    }
  };

  const pickImageFromGallery = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.7,
      base64: true,
    });

    if (!result.canceled && result.assets.length > 0) {
      const base64Img = `data:image/jpeg;base64,${result.assets[0].base64}`;
      setForm((prev) => ({ ...prev, photo: base64Img }));
    }
  };

  const takePhotoWithCamera = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission requise",
        "Permission caméra nécessaire pour prendre une photo"
      );
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.7,
      base64: true,
    });

    if (!result.canceled && result.assets.length > 0) {
      const base64Img = `data:image/jpeg;base64,${result.assets[0].base64}`;
      setForm((prev) => ({ ...prev, photo: base64Img }));
    }
  };

  const showPhotoOptions = () => {
    setPhotoModalVisible(true);
  };

  const handlePhotoSelection = async (type: "camera" | "gallery") => {
    setPhotoModalVisible(false);
    if (type === "camera") {
      await takePhotoWithCamera();
    } else {
      await pickImageFromGallery();
    }
  };

  const handleSavePhoto = async () => {
    if (!form.photo) {
      Alert.alert("Erreur", "Aucune photo sélectionnée.");
      return;
    }
    setPhotoLoading(true);
    try {
      await updateDriverPhoto(form.photo);
      await refreshProfile();
      Alert.alert("Succès", "Photo mise à jour.");
    } catch (error) {
      Alert.alert("Erreur", "Impossible de mettre à jour la photo.");
    } finally {
      setPhotoLoading(false);
    }
  };

  const handleLogout = () => {
    setLogoutModalVisible(true);
  };

  const confirmLogout = async () => {
    setLogoutModalVisible(false);
    try {
      await logout();
      Alert.alert("Succès", "Vous avez été déconnecté.");
    } catch (error) {
      Alert.alert("Erreur", "Impossible de se déconnecter.");
    }
  };

  if (!driver) {
    return (
      <ThemedView className="flex-1 justify-center items-center">
        <Loader />
      </ThemedView>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        style={styles.scrollContainer}
      >
        {/* Header avec photo intégrée */}
        <View style={styles.headerGradient}>
          <View style={styles.headerContent}>
            <View style={styles.headerText}>
              <ThemedText style={styles.headerTitle}>
                {driver?.first_name} {driver?.last_name}
              </ThemedText>
            </View>
            <TouchableOpacity
              style={styles.headerPhotoContainer}
              onPress={showPhotoOptions}
            >
              <Image
                source={form.photo ? { uri: form.photo } : DefaultDriver}
                style={styles.headerPhoto}
              />
              <View style={styles.headerPhotoOverlay}>
                <Ionicons name="camera" size={16} color="white" />
              </View>
            </TouchableOpacity>
          </View>
        </View>

        {/* Section Informations Personnelles */}
        <View style={styles.cardContainer}>
          <View style={styles.cardHeader}>
            <Ionicons name="person-outline" size={20} color="#00796B" />
            <ThemedText style={styles.cardTitle}>
              Informations Personnelles
            </ThemedText>
          </View>

          <InputField
            label="Téléphone"
            value={form.phone}
            keyboardType="phone-pad"
            onChangeText={(phone) => setForm({ ...form, phone })}
            showToggle={false}
          />

          <InputField
            label="Adresse"
            value={form.address}
            onChangeText={(address) => setForm({ ...form, address })}
            placeholder="Votre adresse complète"
            showToggle={false}
          />

          <InputField
            label="Email"
            value={form.email}
            keyboardType="email-address"
            onChangeText={(email) => setForm({ ...form, email })}
            placeholder="votre.email@exemple.com"
            showToggle={false}
          />
        </View>

        {/* Section Véhicule */}
        <View style={styles.cardContainer}>
          <View style={styles.cardHeader}>
            <Ionicons name="car-outline" size={20} color="#00796B" />
            <ThemedText style={styles.cardTitle}>Mon Véhicule</ThemedText>
          </View>

          <InputField
            label="Véhicule assigné"
            value={form.vehicle_assigned}
            onChangeText={(vehicle_assigned) =>
              setForm({ ...form, vehicle_assigned })
            }
            placeholder="Type de véhicule"
            showToggle={false}
          />

          <InputField
            label="Marque du véhicule"
            value={form.brand}
            onChangeText={(brand) => setForm({ ...form, brand })}
            placeholder="Marque du véhicule"
            showToggle={false}
          />

          <InputField
            label="Plaque d'immatriculation"
            value={form.license_plate}
            onChangeText={(license_plate) =>
              setForm({ ...form, license_plate })
            }
            placeholder="ABC-123"
            showToggle={false}
          />
        </View>

        {/* Actions */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity
            style={styles.saveButton}
            onPress={handleSaveProfile}
            disabled={profileLoading}
          >
            <Ionicons name="save-outline" size={20} color="white" />
            <ThemedText style={styles.saveButtonText}>
              {profileLoading
                ? "Enregistrement..."
                : "Enregistrer les modifications"}
            </ThemedText>
          </TouchableOpacity>

          {/* Bouton de déconnexion */}
          <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
            <Ionicons name="log-out-outline" size={20} color="white" />
            <ThemedText style={styles.logoutButtonText}>
              Se déconnecter
            </ThemedText>
          </TouchableOpacity>
        </View>

        {/* Espacement final */}
        <View style={styles.bottomSpacing} />
      </ScrollView>

      {/* Modal de sélection photo */}
      <Modal
        visible={photoModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setPhotoModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContainer}>
            <View style={styles.modalHeader}>
              <ThemedText style={styles.modalTitle}>
                Modifier la photo
              </ThemedText>
              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={() => setPhotoModalVisible(false)}
              >
                <Ionicons name="close" size={24} color="#666" />
              </TouchableOpacity>
            </View>

            <View style={styles.modalContent}>
              <TouchableOpacity
                style={styles.modalOption}
                onPress={() => handlePhotoSelection("camera")}
              >
                <View style={styles.modalOptionIcon}>
                  <Ionicons name="camera" size={32} color="#00796B" />
                </View>
                <ThemedText style={styles.modalOptionText}>
                  Prendre une photo
                </ThemedText>
                <ThemedText style={styles.modalOptionSubtext}>
                  Utiliser la caméra
                </ThemedText>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.modalOption}
                onPress={() => handlePhotoSelection("gallery")}
              >
                <View style={styles.modalOptionIcon}>
                  <Ionicons name="images" size={32} color="#00796B" />
                </View>
                <ThemedText style={styles.modalOptionText}>
                  Choisir depuis la galerie
                </ThemedText>
                <ThemedText style={styles.modalOptionSubtext}>
                  Sélectionner une image existante
                </ThemedText>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Modal de déconnexion */}
      <Modal
        visible={logoutModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setLogoutModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.logoutModalContainer}>
            <View style={styles.logoutModalHeader}>
              <View style={styles.logoutIconContainer}>
                <Ionicons name="log-out-outline" size={32} color="#D32F2F" />
              </View>
              <ThemedText style={styles.logoutModalTitle}>
                Déconnexion
              </ThemedText>
              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={() => setLogoutModalVisible(false)}
              >
                <Ionicons name="close" size={24} color="#666" />
              </TouchableOpacity>
            </View>

            <View style={styles.logoutModalContent}>
              <ThemedText style={styles.logoutModalMessage}>
                Êtes-vous sûr de vouloir vous déconnecter ?
              </ThemedText>
              <ThemedText style={styles.logoutModalSubtext}>
                Vous devrez vous reconnecter pour accéder à l'application
              </ThemedText>

              <View style={styles.logoutModalActions}>
                <TouchableOpacity
                  style={styles.logoutCancelButton}
                  onPress={() => setLogoutModalVisible(false)}
                >
                  <ThemedText style={styles.logoutCancelButtonText}>
                    Annuler
                  </ThemedText>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.logoutConfirmButton}
                  onPress={confirmLogout}
                >
                  <Ionicons name="log-out-outline" size={20} color="white" />
                  <ThemedText style={styles.logoutConfirmButtonText}>
                    Se déconnecter
                  </ThemedText>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

// Styles modernes avec photo dans le header et modal amélioré
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F7F9FB",
  },

  scrollContainer: {
    flex: 1,
  },

  // Header avec photo intégrée
  headerGradient: {
    backgroundColor: "#004D40",
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: 16,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },

  headerContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  headerText: {
    flex: 1,
  },

  headerTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#FFFFFF",
  },

  headerPhotoContainer: {
    position: "relative",
    marginLeft: 16,
  },

  headerPhoto: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#E0E0E0",
    borderWidth: 3,
    borderColor: "#FFFFFF",
  },

  headerPhotoOverlay: {
    position: "absolute",
    bottom: -2,
    right: -2,
    backgroundColor: "#00796B",
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#FFFFFF",
  },

  // Cartes de contenu
  cardContainer: {
    backgroundColor: "#FFFFFF",
    marginHorizontal: 16,
    marginVertical: 8,
    borderRadius: 16,
    padding: 18,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
    borderWidth: 1,
    borderColor: "#E0E0E0",
  },

  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#E0E0E0",
  },

  cardTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#004D40",
    marginLeft: 12,
    flex: 1,
  },

  // Actions
  actionsContainer: {
    paddingHorizontal: 16,
    marginVertical: 12,
  },

  saveButton: {
    backgroundColor: "#00796B",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 12,
    shadowColor: "#00796B",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 4,
    marginBottom: 12,
  },

  saveButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
    marginLeft: 8,
  },

  // Bouton de déconnexion
  logoutButton: {
    backgroundColor: "#D32F2F",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 12,
    shadowColor: "#D32F2F",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 4,
  },

  logoutButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
    marginLeft: 8,
  },

  // Espacement final
  bottomSpacing: {
    height: 80,
  },

  // Modal de sélection photo
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },

  modalContainer: {
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    width: "100%",
    maxWidth: 400,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 10,
  },

  modalHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#E0E0E0",
  },

  modalTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#004D40",
  },

  modalCloseButton: {
    padding: 4,
  },

  modalContent: {
    padding: 20,
  },

  modalOption: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 16,
    paddingHorizontal: 12,
    borderRadius: 12,
    marginBottom: 8,
    backgroundColor: "#F8F9FA",
    borderWidth: 1,
    borderColor: "#E9ECEF",
  },

  modalOptionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "#E8F5E8",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 16,
  },

  modalOptionText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#004D40",
    flex: 1,
  },

  modalOptionSubtext: {
    fontSize: 12,
    color: "#666",
    marginTop: 2,
  },

  // Modal de déconnexion
  logoutModalContainer: {
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    width: "90%",
    maxWidth: 400,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 10,
  },

  logoutModalHeader: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#E0E0E0",
  },

  logoutIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "#FFEBEE",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },

  logoutModalTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#004D40",
    flex: 1,
  },

  logoutModalContent: {
    padding: 20,
  },

  logoutModalMessage: {
    fontSize: 16,
    color: "#374151",
    textAlign: "center",
    marginBottom: 8,
    fontWeight: "500",
  },

  logoutModalSubtext: {
    fontSize: 14,
    color: "#6B7280",
    textAlign: "center",
    marginBottom: 24,
    lineHeight: 20,
  },

  logoutModalActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
  },

  logoutCancelButton: {
    flex: 1,
    backgroundColor: "#F3F4F6",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },

  logoutCancelButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#374151",
  },

  logoutConfirmButton: {
    flex: 1,
    backgroundColor: "#D32F2F",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    shadowColor: "#D32F2F",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },

  logoutConfirmButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#FFFFFF",
    marginLeft: 8,
  },
});
