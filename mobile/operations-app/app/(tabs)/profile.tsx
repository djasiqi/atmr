import React, { useState, useEffect, useCallback } from "react";
import {
  View,
  ScrollView,
  Alert,
  TouchableOpacity,
  Image,
  Text,
  Modal,
  ActivityIndicator,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import { useAuth } from "@/hooks/useAuth";
import {
  getLastSessionEvent,
  getLastSessionEventFromStorage,
  subscribeSessionJournal,
} from "@/services/sessionJournal";
import {
  updateDriverProfile,
  updateDriverPhoto,
  switchToEnterpriseToken,
  invalidateInterceptorCache,
} from "@/services/api";
import { sendIngestEvent } from "@/src/config/telemetry";
import { secureStorage, asyncStorage } from "@/services/storage";
import {
  ENTERPRISE_SESSION_KEY,
  fetchEnterpriseSession,
  EnterpriseTokenPayload,
  invalidateEnterpriseInterceptorCache,
} from "@/services/enterpriseAuth";
import { InputField } from "@/components/ui/InputField";
import { Loader } from "@/components/ui/Loader";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { profileStyles } from "@/styles/profileStyles";

// Import direct de l'image par défaut
import DefaultDriver from "../../assets/images/default-driver.png";

export default function ProfileScreen() {
  const { driver, refreshProfile, logout, switchMode, loadEnterpriseSession } = useAuth();
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
  const [switchingToEnterprise, setSwitchingToEnterprise] = useState(false);
  const [showSwitchModal, setShowSwitchModal] = useState(false);
  const [lastSessionDiag, setLastSessionDiag] = useState<{
    event: string;
    at: number;
  } | null>(() => getLastSessionEvent());

  useEffect(() => {
    const unsub = subscribeSessionJournal((event, at) => {
      setLastSessionDiag({ event, at });
    });
    setLastSessionDiag(getLastSessionEvent());
    getLastSessionEventFromStorage().then((v) => {
      if (v) setLastSessionDiag(v);
    });
    return unsub;
  }, []);

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

  // Fonction pour basculer vers le compte entreprise
  const handleSwitchToEnterprise = useCallback(async () => {
    setShowSwitchModal(false);
    setSwitchingToEnterprise(true);
    try {
      // 1. Obtenir un token entreprise à partir du token driver
      const enterpriseTokenResponse = await switchToEnterpriseToken();
      console.log("[Profile] Tokens entreprise reçus:", {
        hasToken: !!enterpriseTokenResponse.token,
        hasRefreshToken: !!enterpriseTokenResponse.refresh_token,
        userPublicId: enterpriseTokenResponse.user.public_id,
      });

      // 2. ✅ CORRECTION: Stocker les tokens Enterprise dans SecureStore (source of truth)
      // (enterpriseApi lit SecureStore, pas AsyncStorage)
      await secureStorage.setEnterpriseToken(enterpriseTokenResponse.token);
      if (enterpriseTokenResponse.refresh_token) {
        await secureStorage.setEnterpriseRefreshToken(enterpriseTokenResponse.refresh_token);
      } else {
        await secureStorage.removeEnterpriseRefreshToken();
      }
      // Invalider le cache de l'intercepteur Enterprise pour forcer l'utilisation du nouveau token
      invalidateEnterpriseInterceptorCache();
      console.log("[Profile] Tokens entreprise stockés dans SecureStore + cache Enterprise invalidé");

      // 3. Récupérer la session complète depuis le backend
      // Cela nous donne toutes les informations (user.id, dispatch_mode, etc.)
      let enterprisePayload: EnterpriseTokenPayload;
      try {
        const session = await fetchEnterpriseSession(enterpriseTokenResponse.token);
        enterprisePayload = {
          token: enterpriseTokenResponse.token,
          refresh_token: enterpriseTokenResponse.refresh_token || null,
          user: session.user,
          company: {
            id: session.company.id,
            name: session.company.name,
            dispatch_mode: session.company.dispatch_mode,
          },
          scopes: session.scopes || [],
          session_id: session.session_id || `switch_${Date.now()}_${enterpriseTokenResponse.user.public_id}`,
          mfa_required: false as const,
        };
        console.log("[Profile] Session entreprise récupérée depuis le backend");
      } catch (sessionError) {
        console.warn("[Profile] Erreur lors de la récupération de la session, utilisation des données du switch:", sessionError);
        // Fallback: utiliser les données du switch si fetchEnterpriseSession échoue
        enterprisePayload = {
          token: enterpriseTokenResponse.token,
          refresh_token: enterpriseTokenResponse.refresh_token || null,
          user: {
            id: 0,
            public_id: enterpriseTokenResponse.user.public_id,
            email: enterpriseTokenResponse.user.email || "",
            first_name: enterpriseTokenResponse.user.first_name || "",
            last_name: enterpriseTokenResponse.user.last_name || "",
            role: "company",
          },
          company: {
            id: enterpriseTokenResponse.company.id,
            name: enterpriseTokenResponse.company.name,
            dispatch_mode: "MANUAL" as const,
          },
          scopes: [],
          session_id: `switch_${Date.now()}_${enterpriseTokenResponse.user.public_id}`,
          mfa_required: false as const,
        };
      }

      // 4. Stocker la session complète dans AsyncStorage
      await AsyncStorage.multiSet([
        [
          ENTERPRISE_SESSION_KEY,
          JSON.stringify({
            token: enterprisePayload.token,
            refreshToken: enterprisePayload.refresh_token,
            user: enterprisePayload.user,
            company: {
              id: enterprisePayload.company.id,
              name: enterprisePayload.company.name,
              dispatchMode: enterprisePayload.company.dispatch_mode,
            },
            scopes: enterprisePayload.scopes,
            sessionId: enterprisePayload.session_id,
          }),
        ],
        // Marquer que la session vient d'être créée pour éviter la vérification immédiate
        ["enterprise_session_just_created", "true"],
      ]);
      console.log("[Profile] Session entreprise complète stockée dans AsyncStorage");

      // 5. Basculer vers le mode entreprise
      // #region agent log
      sendIngestEvent({ location: 'profile.tsx:handleSwitchToEnterprise', message: 'Avant switchMode', data: { hasToken: !!enterpriseTokenResponse.token, hasRefreshToken: !!enterpriseTokenResponse.refresh_token, companyId: enterpriseTokenResponse.company.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'D' });
      // #endregion
      await switchMode("enterprise");
      console.log("[Profile] Mode changé vers 'enterprise'");
      // #region agent log
      sendIngestEvent({ location: 'profile.tsx:handleSwitchToEnterprise', message: 'Après switchMode', data: { mode: 'enterprise' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'E' });
      // #endregion

      // 6. Invalider le cache de l'intercepteur driver (par sécurité/cohérence)
      // Note: l'intercepteur Enterprise a déjà été invalidé après écriture SecureStore ci-dessus.
      invalidateInterceptorCache();
      console.log("[Profile] Cache intercepteur invalidé");

      // 7. Nettoyer seulement le driver_id dans AsyncStorage
      await asyncStorage.removeDriverId();
      console.log("[Profile] Driver ID nettoyé");

      // 8. Charger la session depuis AsyncStorage et mettre à jour le contexte
      // Au lieu d'appeler refreshEnterprise (qui essaie de rafraîchir le token),
      // nous chargeons directement la session que nous venons de stocker
      await loadEnterpriseSession();
      console.log("[Profile] Session entreprise chargée depuis AsyncStorage");

      // 10. Naviguer vers le dashboard entreprise
      // #region agent log
      sendIngestEvent({ location: 'profile.tsx:handleSwitchToEnterprise', message: 'Avant navigation dashboard', data: { target: '/(enterprise)/dashboard' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'F' });
      // #endregion
      router.replace("/(enterprise)/dashboard" as any);
      Alert.alert(
        "Basculement réussi",
        `Vous êtes maintenant connecté en tant qu'entreprise (${enterpriseTokenResponse.company.name}).`
      );
    } catch (error: any) {
      console.error("Erreur lors du basculement:", error);
      const errorMessage =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de basculer vers le compte entreprise.";
      Alert.alert("Erreur", errorMessage);
    } finally {
      setSwitchingToEnterprise(false);
    }
  }, [switchMode]);

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

        {/* Section Switch de compte (si chauffeur d'urgence) */}
        {driver?.company_id && driver?.driver_type === "EMERGENCY" && (
          <View style={profileStyles.cardContainer}>
            <View style={profileStyles.cardHeader}>
              <Ionicons name="swap-horizontal" size={22} color="#0A7F59" />
              <View style={{ flex: 1, marginLeft: 12 }}>
                <Text style={profileStyles.cardTitle}>Compte entreprise</Text>
                <Text style={[profileStyles.cardTitle, { fontSize: 14, fontWeight: "normal", color: "#5F7369", marginTop: 4 }]}>
                  Vous êtes lié à l'entreprise {driver.company?.name || `#${driver.company_id}`}
                </Text>
              </View>
            </View>

            <TouchableOpacity
              style={[
                profileStyles.saveButton,
                { marginTop: 16, opacity: switchingToEnterprise ? 0.6 : 1 },
              ]}
              onPress={() => setShowSwitchModal(true)}
              disabled={switchingToEnterprise}
            >
              {switchingToEnterprise ? (
                <>
                  <ActivityIndicator size="small" color="#FFFFFF" />
                  <Text style={profileStyles.saveButtonText}>
                    Basculement en cours...
                  </Text>
                </>
              ) : (
                <>
                  <Ionicons name="business-outline" size={20} color="#FFFFFF" />
                  <Text style={profileStyles.saveButtonText}>
                    Basculer vers l'entreprise
                  </Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        )}

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

        {/* P0.1 – Menu debug chauffeur : dernier reason session (X-Session-Diag) */}
        {__DEV__ && lastSessionDiag && (
          <View
            style={{
              marginHorizontal: 16,
              marginTop: 12,
              padding: 10,
              backgroundColor: "#f0f4f0",
              borderRadius: 8,
              borderLeftWidth: 3,
              borderLeftColor: "#0A7F59",
            }}
          >
            <Text
              style={{
                fontSize: 11,
                color: "#5F7369",
                marginBottom: 4,
                fontWeight: "600",
              }}
            >
              Debug – Dernier événement session
            </Text>
            <Text style={{ fontSize: 13, color: "#1a1a1a" }}>
              {lastSessionDiag.event}
            </Text>
            <Text style={{ fontSize: 11, color: "#888", marginTop: 2 }}>
              {new Date(lastSessionDiag.at).toLocaleString()} · envoyé en
              X-Session-Diag
            </Text>
          </View>
        )}

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

      {/* Modal de confirmation switch vers entreprise */}
      <Modal
        visible={showSwitchModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowSwitchModal(false)}
      >
        <View style={profileStyles.modalOverlay}>
          <View style={profileStyles.logoutModalContainer}>
            <View style={[profileStyles.logoutIconContainer, { backgroundColor: "rgba(10,127,89,0.1)" }]}>
              <Ionicons name="business-outline" size={32} color="#0A7F59" />
            </View>
            <Text style={profileStyles.logoutModalTitle}>
              Basculer vers l'entreprise
            </Text>
            <Text style={profileStyles.logoutModalMessage}>
              Vous allez basculer vers votre compte entreprise. Vous pourrez revenir au compte chauffeur depuis les paramètres.
            </Text>
            <View style={profileStyles.logoutModalActions}>
              <TouchableOpacity
                style={profileStyles.logoutCancelButton}
                onPress={() => setShowSwitchModal(false)}
              >
                <Text style={profileStyles.logoutCancelButtonText}>
                  Annuler
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[profileStyles.logoutConfirmButton, { backgroundColor: "#0A7F59" }]}
                onPress={handleSwitchToEnterprise}
              >
                <Text style={profileStyles.logoutConfirmButtonText}>
                  Basculer
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}
