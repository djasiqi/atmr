import React, { useState, useEffect } from "react";
import {
  View,
  ScrollView,
  Alert,
  TouchableOpacity,
  Image,
  Text,
  Modal,
} from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { updateDriverProfile, updateDriverPhoto } from "@/services/api";
import { InputField } from "@/components/ui/InputField";
import { Loader } from "@/components/ui/Loader";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { profileStyles } from "@/styles/profileStyles";

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
      <View style={profileStyles.container}>
        <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
          <Loader />
        </View>
      </View>
    );
  }

  return (
    <View style={profileStyles.container}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        style={profileStyles.scrollContainer}
      >
        {/* Header avec photo intégrée */}
        <View style={profileStyles.headerGradient}>
          <View style={profileStyles.headerContent}>
            <View style={profileStyles.headerText}>
              <Text style={profileStyles.headerTitle}>
                {driver?.first_name} {driver?.last_name}
              </Text>
            </View>
            <TouchableOpacity
              style={profileStyles.headerPhotoContainer}
              onPress={showPhotoOptions}
            >
              <Image
                source={form.photo ? { uri: form.photo } : DefaultDriver}
                style={profileStyles.headerPhoto}
              />
              <View style={profileStyles.headerPhotoOverlay}>
                <Ionicons name="camera" size={16} color="#FFFFFF" />
              </View>
            </TouchableOpacity>
          </View>
        </View>

        {/* Section Informations Personnelles */}
        <View style={profileStyles.cardContainer}>
          <View style={profileStyles.cardHeader}>
            <Ionicons name="person-outline" size={22} color="#0A7F59" />
            <Text style={profileStyles.cardTitle}>
              Informations Personnelles
            </Text>
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
        <View style={profileStyles.cardContainer}>
          <View style={profileStyles.cardHeader}>
            <Ionicons name="car-outline" size={22} color="#0A7F59" />
            <Text style={profileStyles.cardTitle}>Mon Véhicule</Text>
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
        <View style={profileStyles.actionsContainer}>
          <TouchableOpacity
            style={profileStyles.saveButton}
            onPress={handleSaveProfile}
            disabled={profileLoading}
          >
            <Ionicons name="save-outline" size={20} color="#FFFFFF" />
            <Text style={profileStyles.saveButtonText}>
              {profileLoading
                ? "Enregistrement..."
                : "Enregistrer les modifications"}
            </Text>
          </TouchableOpacity>

          {/* Bouton de déconnexion */}
          <TouchableOpacity style={profileStyles.logoutButton} onPress={handleLogout}>
            <Ionicons name="log-out-outline" size={20} color="#FFFFFF" />
            <Text style={profileStyles.logoutButtonText}>
              Se déconnecter
            </Text>
          </TouchableOpacity>
        </View>

        {/* Espacement final */}
        <View style={profileStyles.bottomSpacing} />
      </ScrollView>

      {/* Modal de sélection photo */}
      <Modal
        visible={photoModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setPhotoModalVisible(false)}
      >
        <View style={profileStyles.modalOverlay}>
          <View style={profileStyles.modalContainer}>
            <View style={profileStyles.modalHeader}>
              <Text style={profileStyles.modalTitle}>
                Modifier la photo
              </Text>
              <TouchableOpacity
                style={profileStyles.modalCloseButton}
                onPress={() => setPhotoModalVisible(false)}
              >
                <Ionicons name="close" size={24} color="#5F7369" />
              </TouchableOpacity>
            </View>

            <View style={profileStyles.modalContent}>
              <TouchableOpacity
                style={profileStyles.modalOption}
                onPress={() => handlePhotoSelection("camera")}
              >
                <View style={profileStyles.modalOptionIcon}>
                  <Ionicons name="camera" size={28} color="#0A7F59" />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={profileStyles.modalOptionText}>
                    Prendre une photo
                  </Text>
                  <Text style={profileStyles.modalOptionSubtext}>
                    Utiliser la caméra
                  </Text>
                </View>
              </TouchableOpacity>

              <TouchableOpacity
                style={profileStyles.modalOption}
                onPress={() => handlePhotoSelection("gallery")}
              >
                <View style={profileStyles.modalOptionIcon}>
                  <Ionicons name="images" size={28} color="#0A7F59" />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={profileStyles.modalOptionText}>
                    Choisir depuis la galerie
                  </Text>
                  <Text style={profileStyles.modalOptionSubtext}>
                    Sélectionner une image existante
                  </Text>
                </View>
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
        <View style={profileStyles.modalOverlay}>
          <View style={profileStyles.logoutModalContainer}>
            <View style={profileStyles.logoutIconContainer}>
              <Ionicons name="log-out-outline" size={32} color="#D32F2F" />
            </View>
            <Text style={profileStyles.logoutModalTitle}>
              Déconnexion
            </Text>
            <Text style={profileStyles.logoutModalMessage}>
              Êtes-vous sûr de vouloir vous déconnecter ?
            </Text>
            <View style={profileStyles.logoutModalActions}>
              <TouchableOpacity
                style={profileStyles.logoutCancelButton}
                onPress={() => setLogoutModalVisible(false)}
              >
                <Text style={profileStyles.logoutCancelButtonText}>
                  Annuler
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={profileStyles.logoutConfirmButton}
                onPress={confirmLogout}
              >
                <Text style={profileStyles.logoutConfirmButtonText}>
                  Se déconnecter
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}
