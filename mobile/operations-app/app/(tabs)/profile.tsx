import React, { useState, useEffect, useCallback } from "react";
import {
  View,
  ScrollView,
  Alert,
  TouchableOpacity,
  Image,
  Text,
  Modal,
  Pressable,
  ActivityIndicator,
  Platform,
  StyleSheet,
  Animated,
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
  testPushNotification,
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
import { profileStyles as s } from "@/styles/profileStyles";
import { PushDebugCard } from "@/components/common/PushDebugCard";
import { getLogger } from "@/utils/logger";

const log = getLogger("Profile");

const DefaultDriver = require("../../assets/images/icon.png");

const BRAND = "#00796B";
const TEXT_SEC = "#64748B";

const formatContractType = (ct: string | null | undefined): string => {
  if (!ct) return "Non renseigné";
  const map: Record<string, string> = {
    CDI: "CDI",
    CDD: "CDD",
    TEMP: "Temporaire",
    FREELANCE: "Indépendant",
    STAGE: "Stage",
  };
  return map[ct.toUpperCase()] || ct;
};

const formatDate = (d: string | null | undefined): string => {
  if (!d) return "Non renseigné";
  try {
    return new Date(d).toLocaleDateString("fr-CH", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
  } catch {
    return d;
  }
};

const isExpiringSoon = (d: string | null | undefined): boolean => {
  if (!d) return false;
  try {
    const diff = new Date(d).getTime() - Date.now();
    return diff > 0 && diff < 90 * 24 * 60 * 60 * 1000;
  } catch {
    return false;
  }
};

function InfoRow({
  label,
  value,
  last,
  warn,
}: {
  label: string;
  value: string | null | undefined;
  last?: boolean;
  warn?: boolean;
}) {
  const display = value?.trim() || "Non renseigné";
  const isMuted = !value?.trim();
  return (
    <View style={[s.infoRow, last && s.infoRowLast]}>
      <Text style={s.infoLabel}>{label}</Text>
      {warn && !isMuted ? (
        <View style={[s.infoBadge, s.infoBadgeWarn]}>
          <Text style={[s.infoBadgeText, s.infoBadgeWarnText]}>{display}</Text>
        </View>
      ) : (
        <Text style={[s.infoValue, isMuted && s.infoValueMuted]}>{display}</Text>
      )}
    </View>
  );
}

function AnimatedProgressBar({ duration = 1800 }: { duration?: number }) {
  const anim = React.useRef(new Animated.Value(0)).current;
  React.useEffect(() => {
    Animated.timing(anim, {
      toValue: 1,
      duration,
      useNativeDriver: false,
    }).start();
  }, []);
  const width = anim.interpolate({ inputRange: [0, 1], outputRange: ["0%", "100%"] });
  return (
    <View style={successStyles.progressBar}>
      <Animated.View style={[successStyles.progressFill, { width }]} />
    </View>
  );
}

export default function ProfileScreen() {
  const { driver, refreshProfile, logout, switchMode, loadEnterpriseSession } = useAuth();
  const [form, setForm] = useState({
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
  const [switchSuccessInfo, setSwitchSuccessInfo] = useState<{ visible: boolean; companyName: string }>({ visible: false, companyName: "" });
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
        phone: form.phone,
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
      Alert.alert("Permission requise", "Permission caméra nécessaire pour prendre une photo");
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

  const confirmLogout = async () => {
    setLogoutModalVisible(false);
    try {
      await logout();
    } catch (error) {
      Alert.alert("Erreur", "Impossible de se déconnecter.");
    }
  };

  const handleSwitchToEnterprise = useCallback(async () => {
    setShowSwitchModal(false);
    setSwitchingToEnterprise(true);
    try {
      const enterpriseTokenResponse = await switchToEnterpriseToken();
      log.info("enterprise tokens received", {
        hasToken: !!enterpriseTokenResponse.token,
        hasRefreshToken: !!enterpriseTokenResponse.refresh_token,
        userPublicId: enterpriseTokenResponse.user.public_id,
      });

      await secureStorage.setEnterpriseToken(enterpriseTokenResponse.token);
      if (enterpriseTokenResponse.refresh_token) {
        await secureStorage.setEnterpriseRefreshToken(enterpriseTokenResponse.refresh_token);
      } else {
        await secureStorage.removeEnterpriseRefreshToken();
      }
      invalidateEnterpriseInterceptorCache();
      log.info("enterprise tokens stored, cache invalidated", {});

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
      } catch (sessionError) {
        log.warn("session fetch failed, using switch data", { error: sessionError });
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
        ["enterprise_session_just_created", "true"],
      ]);

      sendIngestEvent({ location: 'profile.tsx:handleSwitchToEnterprise', message: 'Avant switchMode', data: { hasToken: !!enterpriseTokenResponse.token, hasRefreshToken: !!enterpriseTokenResponse.refresh_token, companyId: enterpriseTokenResponse.company.id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'D' });
      await switchMode("enterprise");
      sendIngestEvent({ location: 'profile.tsx:handleSwitchToEnterprise', message: 'Après switchMode', data: { mode: 'enterprise' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'E' });

      invalidateInterceptorCache();
      await asyncStorage.removeDriverId();
      await loadEnterpriseSession();

      sendIngestEvent({ location: 'profile.tsx:handleSwitchToEnterprise', message: 'Avant navigation dashboard', data: { target: '/(enterprise)/dashboard' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'F' });

      setSwitchSuccessInfo({ visible: true, companyName: enterpriseTokenResponse.company.name });
      setTimeout(() => {
        setSwitchSuccessInfo((prev) => ({ ...prev, visible: false }));
        router.replace("/(enterprise)/dashboard" as any);
      }, 1800);
    } catch (error: any) {
      log.error("switch to enterprise failed", { error });
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
      <View style={s.container}>
        <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
          <Loader />
        </View>
      </View>
    );
  }

  return (
    <View style={s.container}>
      <ScrollView showsVerticalScrollIndicator={false} style={s.scrollContainer}>
        {/* ——— Header ——— */}
        <View style={s.headerGradient}>
          <View style={s.headerContent}>
            <View style={s.headerText}>
              <Text style={s.headerTitle}>
                {driver?.first_name} {driver?.last_name}
              </Text>
              <Text style={s.headerSubtitle}>Chauffeur</Text>
            </View>
            <TouchableOpacity
              style={s.headerPhotoContainer}
              onPress={() => setPhotoModalVisible(true)}
            >
              <Image
                source={form.photo ? { uri: form.photo } : DefaultDriver}
                style={s.headerPhoto}
              />
              <View style={s.headerPhotoOverlay}>
                <Ionicons name="camera" size={12} color="#FFFFFF" />
              </View>
            </TouchableOpacity>
          </View>
        </View>

        {/* ——— Informations personnelles ——— */}
        <View style={s.cardContainer}>
          <View style={s.cardHeader}>
            <Ionicons name="person-outline" size={18} color={BRAND} />
            <Text style={s.cardTitle}>Informations personnelles</Text>
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

        {/* ——— Informations professionnelles (lecture seule) ——— */}
        <View style={s.cardContainer}>
          <View style={s.cardHeader}>
            <Ionicons name="briefcase-outline" size={18} color={BRAND} />
            <Text style={s.cardTitle}>Informations professionnelles</Text>
          </View>
          <InfoRow label="Contrat" value={formatContractType(driver.contract_type)} />
          <InfoRow label="Nationalité" value={driver.nationality} />
          <InfoRow label="N° AVS" value={driver.avs_number} />
          <InfoRow
            label="Début d'emploi"
            value={formatDate(driver.employment_start_date)}
          />
          {driver.employment_end_date && (
            <InfoRow label="Fin d'emploi" value={formatDate(driver.employment_end_date)} />
          )}
          {driver.weekly_hours != null && (
            <InfoRow label="Heures / semaine" value={`${driver.weekly_hours}h`} last />
          )}
        </View>

        {/* ——— Permis & Qualifications (lecture seule) ——— */}
        <View style={s.cardContainer}>
          <View style={s.cardHeader}>
            <Ionicons name="shield-checkmark-outline" size={18} color={BRAND} />
            <Text style={s.cardTitle}>Permis & Qualifications</Text>
          </View>
          <View style={s.infoRow}>
            <Text style={s.infoLabel}>Catégories</Text>
            <View style={s.chipRow}>
              {(driver.license_categories ?? []).length > 0
                ? (driver.license_categories as string[]).map((cat: string) => (
                    <View key={cat} style={s.chip}>
                      <Text style={s.chipText}>{cat}</Text>
                    </View>
                  ))
                : <Text style={[s.infoValue, s.infoValueMuted]}>Non renseigné</Text>
              }
            </View>
          </View>
          <InfoRow
            label="Validité permis"
            value={formatDate(driver.license_valid_until)}
            warn={isExpiringSoon(driver.license_valid_until)}
          />
          <InfoRow
            label="Certificat médical"
            value={formatDate(driver.medical_valid_until)}
            warn={isExpiringSoon(driver.medical_valid_until)}
          />
          <View style={[s.infoRow, s.infoRowLast]}>
            <Text style={s.infoLabel}>Formations</Text>
            <View style={s.chipRow}>
              {(driver.trainings ?? []).length > 0
                ? (driver.trainings as string[]).map((t: string, i: number) => (
                    <View key={i} style={s.chip}>
                      <Text style={s.chipText}>{t}</Text>
                    </View>
                  ))
                : <Text style={[s.infoValue, s.infoValueMuted]}>Aucune</Text>
              }
            </View>
          </View>
        </View>

        {/* ——— Contact d'urgence (lecture seule) ——— */}
        {(driver.emergency_contact_name || driver.emergency_contact_phone) && (
          <View style={s.cardContainer}>
            <View style={s.cardHeader}>
              <Ionicons name="medkit-outline" size={18} color={BRAND} />
              <Text style={s.cardTitle}>Contact d'urgence</Text>
            </View>
            <InfoRow label="Nom" value={driver.emergency_contact_name} />
            <InfoRow label="Téléphone" value={driver.emergency_contact_phone} last />
          </View>
        )}

        {/* ——— Mon véhicule (lecture seule) ——— */}
        <View style={s.cardContainer}>
          <View style={s.cardHeader}>
            <Ionicons name="car-outline" size={18} color={BRAND} />
            <Text style={s.cardTitle}>Mon véhicule</Text>
          </View>
          <InfoRow label="Véhicule assigné" value={driver.vehicle_assigned} />
          <InfoRow label="Marque" value={driver.brand} />
          <InfoRow label="Plaque d'immatriculation" value={driver.license_plate} last />
        </View>

        {/* ——— Switch entreprise (si chauffeur d'urgence) ——— */}
        {driver?.company_id && driver?.driver_type === "EMERGENCY" && (
          <View style={s.cardContainer}>
            <View style={s.cardHeader}>
              <Ionicons name="swap-horizontal" size={18} color={BRAND} />
              <View style={{ flex: 1, marginLeft: 10 }}>
                <Text style={s.cardTitle}>Compte entreprise</Text>
                <Text style={s.cardDesc}>
                  Lié à {driver.company?.name || `Entreprise #${driver.company_id}`}
                </Text>
              </View>
            </View>
            <TouchableOpacity
              style={[s.saveButton, { opacity: switchingToEnterprise ? 0.6 : 1 }]}
              onPress={() => setShowSwitchModal(true)}
              disabled={switchingToEnterprise}
            >
              {switchingToEnterprise ? (
                <>
                  <ActivityIndicator size="small" color="#FFFFFF" />
                  <Text style={s.saveButtonText}>Basculement...</Text>
                </>
              ) : (
                <>
                  <Ionicons name="business-outline" size={16} color="#FFFFFF" />
                  <Text style={s.saveButtonText}>Basculer vers l'entreprise</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* ——— Actions ——— */}
        <View style={s.actionsContainer}>
          <TouchableOpacity
            style={s.saveButton}
            onPress={handleSaveProfile}
            disabled={profileLoading}
          >
            <Ionicons name="checkmark-circle-outline" size={16} color="#FFFFFF" />
            <Text style={s.saveButtonText}>
              {profileLoading ? "Enregistrement..." : "Enregistrer"}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={s.logoutButton}
            onPress={() => setLogoutModalVisible(true)}
          >
            <Ionicons name="log-out-outline" size={16} color="#FFFFFF" />
            <Text style={s.logoutButtonText}>Se déconnecter</Text>
          </TouchableOpacity>
        </View>

        {/* Debug section (dev only) */}
        {__DEV__ && lastSessionDiag && (
          <View
            style={{
              marginHorizontal: 16,
              marginTop: 12,
              padding: 10,
              backgroundColor: "rgba(0,121,107,0.04)",
              borderRadius: 10,
              borderLeftWidth: 3,
              borderLeftColor: BRAND,
            }}
          >
            <Text style={{ fontSize: 11, color: TEXT_SEC, marginBottom: 4, fontWeight: "600" }}>
              Debug — Dernier événement session
            </Text>
            <Text style={{ fontSize: 12, color: "#1E293B" }}>{lastSessionDiag.event}</Text>
            <Text style={{ fontSize: 11, color: "#94A3B8", marginTop: 2 }}>
              {new Date(lastSessionDiag.at).toLocaleString()} · X-Session-Diag
            </Text>
          </View>
        )}

        {__DEV__ && <PushDebugCard />}

        {__DEV__ && (
          <TouchableOpacity
            style={{
              backgroundColor: "#2563EB",
              borderRadius: 10,
              paddingVertical: 10,
              paddingHorizontal: 16,
              marginHorizontal: 16,
              marginTop: 12,
              alignItems: "center",
              flexDirection: "row",
              justifyContent: "center",
              gap: 6,
            }}
            onPress={async () => {
              try {
                const res = await testPushNotification();
                if (res.ok) {
                  const details = (res.results ?? [])
                    .map((r) => `${r.platform ?? "?"}: ${r.ok ? "OK" : r.error ?? "Échec"}`)
                    .join("\n");
                  Alert.alert("Test réussi", `Notification envoyée à ${res.tokens_count ?? 0} appareil(s).\n\n${details}`);
                } else {
                  Alert.alert("Échec du test", res.error ?? "La notification n'a pas pu être envoyée.");
                }
              } catch (e: any) {
                Alert.alert("Erreur", e?.message ?? "Erreur réseau");
              }
            }}
          >
            <Ionicons name="notifications-outline" size={14} color="#fff" />
            <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>Tester les notifications</Text>
          </TouchableOpacity>
        )}

        <View style={s.bottomSpacing} />
      </ScrollView>

      {/* ——— Modal photo (bottom sheet) ——— */}
      <Modal
        visible={photoModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setPhotoModalVisible(false)}
      >
        <Pressable style={s.modalOverlay} onPress={() => setPhotoModalVisible(false)}>
          <View
            style={s.modalContainer}
            onStartShouldSetResponder={() => true}
            onTouchEnd={(e) => e.stopPropagation()}
          >
            <View style={s.modalHeader}>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                <View
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 8,
                    backgroundColor: "rgba(0,121,107,0.08)",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Ionicons name="image-outline" size={16} color={BRAND} />
                </View>
                <Text style={s.modalTitle}>Modifier la photo</Text>
              </View>
              <TouchableOpacity
                style={s.modalCloseButton}
                onPress={() => setPhotoModalVisible(false)}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              >
                <Ionicons name="close" size={18} color="#94A3B8" />
              </TouchableOpacity>
            </View>

            <View style={s.modalContent}>
              <TouchableOpacity
                style={s.modalOption}
                onPress={() => handlePhotoSelection("camera")}
              >
                <View style={s.modalOptionIcon}>
                  <Ionicons name="camera-outline" size={20} color={BRAND} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={s.modalOptionText}>Prendre une photo</Text>
                  <Text style={s.modalOptionSubtext}>Utiliser la caméra</Text>
                </View>
                <Ionicons name="chevron-forward" size={16} color="#94A3B8" />
              </TouchableOpacity>

              <TouchableOpacity
                style={s.modalOption}
                onPress={() => handlePhotoSelection("gallery")}
              >
                <View style={s.modalOptionIcon}>
                  <Ionicons name="images-outline" size={20} color={BRAND} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={s.modalOptionText}>Galerie</Text>
                  <Text style={s.modalOptionSubtext}>Choisir une image existante</Text>
                </View>
                <Ionicons name="chevron-forward" size={16} color="#94A3B8" />
              </TouchableOpacity>
            </View>
          </View>
        </Pressable>
      </Modal>

      {/* ——— Modal déconnexion (bottom sheet) ——— */}
      <Modal
        visible={logoutModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setLogoutModalVisible(false)}
      >
        <Pressable style={s.modalOverlay} onPress={() => setLogoutModalVisible(false)}>
          <View
            style={s.logoutModalContainer}
            onStartShouldSetResponder={() => true}
            onTouchEnd={(e) => e.stopPropagation()}
          >
            <View style={s.logoutIconContainer}>
              <Ionicons name="log-out-outline" size={24} color="#dc3545" />
            </View>
            <Text style={s.logoutModalTitle}>Déconnexion</Text>
            <Text style={s.logoutModalMessage}>
              Êtes-vous sûr de vouloir vous déconnecter ? Vous devrez vous reconnecter pour accéder à vos missions.
            </Text>
            <View style={s.logoutModalActions}>
              <TouchableOpacity
                style={s.logoutCancelButton}
                onPress={() => setLogoutModalVisible(false)}
              >
                <Text style={s.logoutCancelButtonText}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity style={s.logoutConfirmButton} onPress={confirmLogout}>
                <Ionicons name="log-out-outline" size={15} color="#fff" />
                <Text style={s.logoutConfirmButtonText}>Se déconnecter</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Pressable>
      </Modal>

      {/* ——— Modal switch entreprise (bottom sheet) ——— */}
      <Modal
        visible={showSwitchModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowSwitchModal(false)}
      >
        <Pressable style={s.modalOverlay} onPress={() => setShowSwitchModal(false)}>
          <View
            style={s.logoutModalContainer}
            onStartShouldSetResponder={() => true}
            onTouchEnd={(e) => e.stopPropagation()}
          >
            <View style={[s.logoutIconContainer, { backgroundColor: "rgba(0,121,107,0.08)" }]}>
              <Ionicons name="business-outline" size={24} color={BRAND} />
            </View>
            <Text style={s.logoutModalTitle}>Basculer vers l'entreprise</Text>
            <Text style={s.logoutModalMessage}>
              Vous allez basculer vers votre compte entreprise. Vous pourrez revenir au compte chauffeur depuis les paramètres.
            </Text>
            <View style={s.logoutModalActions}>
              <TouchableOpacity
                style={s.logoutCancelButton}
                onPress={() => setShowSwitchModal(false)}
              >
                <Text style={s.logoutCancelButtonText}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[s.logoutConfirmButton, { backgroundColor: BRAND }]}
                onPress={handleSwitchToEnterprise}
              >
                <Ionicons name="swap-horizontal-outline" size={15} color="#fff" />
                <Text style={s.logoutConfirmButtonText}>Basculer</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Pressable>
      </Modal>

      {/* ——— Modal succès basculement ——— */}
      <Modal
        visible={switchSuccessInfo.visible}
        transparent
        animationType="fade"
        statusBarTranslucent
      >
        <View style={successStyles.overlay}>
          <View style={successStyles.card}>
            <View style={successStyles.iconWrap}>
              <Ionicons name="checkmark-circle" size={44} color="#0A7F59" />
            </View>
            <Text style={successStyles.title}>Basculement réussi</Text>
            <Text style={successStyles.subtitle}>
              Vous êtes maintenant connecté à{"\n"}
              <Text style={successStyles.companyName}>{switchSuccessInfo.companyName}</Text>
            </Text>
            {switchSuccessInfo.visible && <AnimatedProgressBar duration={1800} />}
          </View>
        </View>
      </Modal>
    </View>
  );
}

const successStyles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(5,22,16,0.6)",
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
  },
  card: {
    width: "100%",
    maxWidth: 320,
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    paddingVertical: 32,
    paddingHorizontal: 24,
    alignItems: "center",
    ...Platform.select({
      ios: {
        shadowColor: "rgba(10,127,89,0.2)",
        shadowOffset: { width: 0, height: 12 },
        shadowOpacity: 1,
        shadowRadius: 28,
      },
      android: { elevation: 12 },
      default: {},
    }),
  },
  iconWrap: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: "rgba(10,127,89,0.08)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 18,
  },
  title: {
    fontSize: 18,
    fontWeight: "700",
    color: "#0F362B",
    letterSpacing: -0.3,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 14,
    color: "#64748B",
    textAlign: "center",
    lineHeight: 20,
    marginTop: 8,
  },
  companyName: {
    fontWeight: "700",
    color: "#0A7F59",
  },
  progressBar: {
    width: "80%",
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(10,127,89,0.1)",
    marginTop: 24,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    borderRadius: 2,
    backgroundColor: "#0A7F59",
  },
});
