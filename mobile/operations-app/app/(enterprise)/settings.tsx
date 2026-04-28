import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { router } from "expo-router";
import { useAuth } from "@/hooks/useAuth";
import { getMyDriverAccount, switchToDriverToken } from "@/services/enterpriseDispatch";
import type { DriverAccountInfo } from "@/types/enterpriseDispatch";
import {
  secureStorage,
  asyncStorage,
  setActiveAuthNamespace,
} from "@/services/storage";
import { invalidateInterceptorCache } from "@/services/api";
import { notifyAuthReady } from "@/services/authSync";
import { sendIngestEvent } from "@/src/config/telemetry";
import { getLogger } from "@/utils/logger";

const log = getLogger("EntSettings");

const BRAND = "#00796B";
const BRAND_DARK = "#00695C";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const BG = "#f4f7fc";
const CARD = "#FFFFFF";
const DANGER = "#dc3545";

const cardShadow = Platform.select({
  ios: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
  },
  android: { elevation: 2 },
  default: {},
});

const btnShadow = Platform.select({
  ios: {
    shadowColor: BRAND,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
  },
  android: { elevation: 3 },
  default: {},
});

const sheetShadow = Platform.select({
  ios: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.1,
    shadowRadius: 16,
  },
  android: { elevation: 12 },
  default: {},
});

export default function EnterpriseSettingsScreen() {
  const { enterpriseSession, logoutEnterprise, switchMode, loadDriverSession } =
    useAuth();

  const [driverAccount, setDriverAccount] = useState<DriverAccountInfo | null>(
    null
  );
  const [checkingDriverAccount, setCheckingDriverAccount] = useState(true);
  const [switchingToDriver, setSwitchingToDriver] = useState(false);
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [showSwitchModal, setShowSwitchModal] = useState(false);

  useEffect(() => {
    const checkDriverAccount = async () => {
      setCheckingDriverAccount(true);
      try {
        const cachedInfo = await asyncStorage.getDriverAccountInfo();
        if (cachedInfo) {
          log.info("driver account loaded from cache", { cachedInfo });
          sendIngestEvent({ location: 'settings.tsx:checkDriverAccount', message: 'Driver account from cache', data: { hasDriverAccount: cachedInfo.has_driver_account, driverType: cachedInfo.driver_type }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });
          setDriverAccount(cachedInfo);
          setCheckingDriverAccount(false);

          try {
            const freshInfo = await getMyDriverAccount();
            log.info("getMyDriverAccount fresh response", { freshInfo });
            sendIngestEvent({ location: 'settings.tsx:checkDriverAccount', message: 'Driver account from API', data: { hasDriverAccount: freshInfo.has_driver_account, driverType: freshInfo.driver_type }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });
            if (JSON.stringify(cachedInfo) !== JSON.stringify(freshInfo)) {
              await asyncStorage.setDriverAccountInfo(freshInfo);
              setDriverAccount(freshInfo);
            }
          } catch (error) {
            log.warn("could not refresh driver account info", { error });
            sendIngestEvent({ location: 'settings.tsx:checkDriverAccount', message: 'Error refreshing driver account', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });
          }
          return;
        }

        log.info("checking driver account");
        const info = await getMyDriverAccount();
        log.info("getMyDriverAccount response", { info });
        sendIngestEvent({ location: 'settings.tsx:checkDriverAccount', message: 'Driver account from API (no cache)', data: { hasDriverAccount: info.has_driver_account, driverType: info.driver_type }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });
        setDriverAccount(info);
        await asyncStorage.setDriverAccountInfo(info);
      } catch (error: any) {
        log.error("driver account check failed", {
          message: error?.message,
          response: error?.response?.data,
          status: error?.response?.status,
        });
        sendIngestEvent({ location: 'settings.tsx:checkDriverAccount', message: 'Error checking driver account', data: { error: String(error), status: error?.response?.status }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });
        setDriverAccount({ has_driver_account: false });
        await asyncStorage.setDriverAccountInfo({ has_driver_account: false });
      } finally {
        setCheckingDriverAccount(false);
      }
    };
    checkDriverAccount();
  }, []);

  const handleSwitchToDriver = useCallback(async () => {
    sendIngestEvent({ location: 'settings.tsx:handleSwitchToDriver', message: 'handleSwitchToDriver entry', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });

    setShowSwitchModal(false);
    setSwitchingToDriver(true);
    try {
      sendIngestEvent({ location: 'settings.tsx:handleSwitchToDriver', message: 'Avant switchToDriverToken', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });

      const driverTokenResponse = await switchToDriverToken();

      sendIngestEvent({ location: 'settings.tsx:handleSwitchToDriver', message: 'Après switchToDriverToken', data: { hasToken: !!driverTokenResponse.token, hasRefreshToken: !!driverTokenResponse.refresh_token }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });

      log.info("driver tokens received", {
        hasToken: !!driverTokenResponse.token,
        hasRefreshToken: !!driverTokenResponse.refresh_token,
      });

      // ✅ Isolation driver/enterprise : nettoyer les tokens entreprise AVANT de stocker les tokens driver
      await logoutEnterprise();
      log.success("enterprise storage cleared before driver tokens");

      await setActiveAuthNamespace({
        role: "driver",
        userId: driverTokenResponse.user.public_id || "unknown",
        tenantId: null,
        sessionId: null,
      });
      await secureStorage.setAccessToken(driverTokenResponse.token);
      if (driverTokenResponse.refresh_token) {
        await secureStorage.setRefreshToken(driverTokenResponse.refresh_token);
      }
      if (driverTokenResponse.user.public_id) {
        await secureStorage.setUserPublicId(driverTokenResponse.user.public_id);
      }
      invalidateInterceptorCache();
      log.success("driver tokens stored and cache invalidated");

      sendIngestEvent({ location: 'settings.tsx:handleSwitchToDriver', message: 'Avant switchMode driver', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });

      await switchMode("driver");

      sendIngestEvent({ location: 'settings.tsx:handleSwitchToDriver', message: 'Après switchMode driver', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });

      log.success("mode switched to driver");

      notifyAuthReady();

      await loadDriverSession();
      log.success("driver session loaded from secure storage");

      await new Promise((resolve) => setTimeout(resolve, 200));

      sendIngestEvent({ location: 'settings.tsx:handleSwitchToDriver', message: 'Avant navigation mission', data: { target: '/(tabs)/mission' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });

      router.replace("/(tabs)/mission" as any);
    } catch (error: any) {
      log.error("switch to driver failed", { error });
      sendIngestEvent({ location: 'settings.tsx:handleSwitchToDriver', message: 'Erreur handleSwitchToDriver', data: { error: String(error), status: error?.response?.status, errorMessage: error?.response?.data?.error }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' });

      const errorMessage =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de basculer vers le compte chauffeur.";
      Alert.alert("Erreur", errorMessage);
    } finally {
      setSwitchingToDriver(false);
    }
  }, [logoutEnterprise, switchMode, loadDriverSession]);

  const companyName = enterpriseSession?.company?.name || "Entreprise";
  const userName = enterpriseSession?.user
    ? `${enterpriseSession.user.first_name || ""} ${enterpriseSession.user.last_name || ""}`.trim()
    : "";
  const userEmail = enterpriseSession?.user?.email || "";

  return (
    <>
    <ScrollView
      style={s.container}
      contentContainerStyle={s.scrollContent}
      showsVerticalScrollIndicator={false}
    >
        {/* ——— Compte entreprise ——— */}
        <View style={s.card}>
          <View style={s.cardHeader}>
            <Ionicons name="business-outline" size={16} color={BRAND} />
            <Text style={s.cardTitle}>Compte entreprise</Text>
          </View>
          <InfoRow label="Entreprise" value={companyName} />
          {userName ? <InfoRow label="Utilisateur" value={userName} /> : null}
          {userEmail ? (
            <InfoRow label="Email" value={userEmail} last />
          ) : (
            <InfoRow label="Entreprise" value={companyName} last />
          )}
        </View>

        {/* ——— Compte chauffeur (loading) ——— */}
        {checkingDriverAccount && (
          <View style={s.card}>
            <View style={s.cardHeader}>
              <ActivityIndicator size="small" color={BRAND} />
              <Text style={[s.cardTitle, { color: TEXT_SEC }]}>
                Vérification du compte chauffeur…
              </Text>
            </View>
          </View>
        )}

        {/* ——— Pas de compte chauffeur ——— */}
        {!checkingDriverAccount &&
          driverAccount &&
          !driverAccount.has_driver_account && (
            <View style={s.card}>
              <View style={s.cardHeader}>
                <Ionicons
                  name="information-circle-outline"
                  size={16}
                  color={TEXT_MUTED}
                />
                <Text style={[s.cardTitle, { color: TEXT_MUTED }]}>
                  Compte chauffeur
                </Text>
              </View>
              <Text style={s.cardMessage}>
                Aucun compte chauffeur associé.{"\n"}Contactez votre
                administrateur pour en créer un.
              </Text>
            </View>
          )}

        {/* ——— Basculer vers chauffeur ——— */}
        {!checkingDriverAccount && driverAccount?.has_driver_account && (
          <View style={s.card}>
            <View style={s.cardHeader}>
              <Ionicons name="swap-horizontal" size={16} color={BRAND} />
              <Text style={s.cardTitle}>Compte chauffeur</Text>
            </View>
            <Text style={s.cardMessage}>
              {driverAccount.driver_type === "EMERGENCY"
                ? "Vous êtes également chauffeur d'urgence."
                : "Vous avez également un compte chauffeur."}
            </Text>
            <TouchableOpacity
              style={[s.primaryBtn, switchingToDriver && s.primaryBtnDisabled]}
              onPress={() => setShowSwitchModal(true)}
              disabled={checkingDriverAccount || switchingToDriver}
              activeOpacity={0.7}
            >
              {switchingToDriver ? (
                <>
                  <ActivityIndicator size="small" color="#fff" />
                  <Text style={s.primaryBtnText}>Basculement…</Text>
                </>
              ) : (
                <>
                  <Ionicons name="car-outline" size={16} color="#fff" />
                  <Text style={s.primaryBtnText}>
                    Basculer vers le compte chauffeur
                  </Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* ——— Déconnexion ——— */}
        <View style={s.card}>
          <View style={s.cardHeader}>
            <Ionicons name="log-out-outline" size={16} color={DANGER} />
            <Text style={[s.cardTitle, { color: DANGER }]}>Déconnexion</Text>
          </View>
          <Text style={s.cardMessage}>
            Se déconnecter de l'espace entreprise.
          </Text>
          <TouchableOpacity
            style={s.dangerBtn}
            onPress={() => setShowLogoutModal(true)}
            activeOpacity={0.7}
          >
            <Ionicons name="log-out-outline" size={15} color="#fff" />
            <Text style={s.dangerBtnText}>Se déconnecter</Text>
          </TouchableOpacity>
        </View>

        <View style={s.bottomSpacing} />
      </ScrollView>

      {/* ——— Modal switch chauffeur (bottom sheet) ——— */}
      {/* Modals rendered outside ScrollView for proper overlay */}
      <Modal
        visible={showSwitchModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowSwitchModal(false)}
      >
        <Pressable
          style={s.modalOverlay}
          onPress={() => setShowSwitchModal(false)}
        >
          <View
            style={s.sheetContainer}
            onStartShouldSetResponder={() => true}
            onTouchEnd={(e) => e.stopPropagation()}
          >
            <View style={s.sheetHandle} />
            <View style={[s.sheetIcon, { backgroundColor: "rgba(0,121,107,0.08)" }]}>
              <Ionicons name="swap-horizontal" size={22} color={BRAND} />
            </View>
            <Text style={s.sheetTitle}>Basculer vers le compte chauffeur</Text>
            <Text style={s.sheetMessage}>
              Vous allez être automatiquement connecté en tant que chauffeur. La
              connexion entreprise sera fermée.
            </Text>
            <View style={s.sheetActions}>
              <TouchableOpacity
                style={s.sheetCancel}
                onPress={() => setShowSwitchModal(false)}
              >
                <Text style={s.sheetCancelText}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[s.sheetConfirm, { backgroundColor: BRAND }]}
                onPress={handleSwitchToDriver}
              >
                <Ionicons
                  name="swap-horizontal-outline"
                  size={15}
                  color="#fff"
                />
                <Text style={s.sheetConfirmText}>Basculer</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Pressable>
      </Modal>

      {/* ——— Modal déconnexion (bottom sheet) ——— */}
      <Modal
        visible={showLogoutModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowLogoutModal(false)}
      >
        <Pressable
          style={s.modalOverlay}
          onPress={() => setShowLogoutModal(false)}
        >
          <View
            style={s.sheetContainer}
            onStartShouldSetResponder={() => true}
            onTouchEnd={(e) => e.stopPropagation()}
          >
            <View style={s.sheetHandle} />
            <View style={[s.sheetIcon, { backgroundColor: "rgba(220,53,69,0.08)" }]}>
              <Ionicons name="log-out-outline" size={22} color={DANGER} />
            </View>
            <Text style={s.sheetTitle}>Déconnexion</Text>
            <Text style={s.sheetMessage}>
              Voulez-vous quitter l'espace entreprise ?
            </Text>
            <View style={s.sheetActions}>
              <TouchableOpacity
                style={s.sheetCancel}
                onPress={() => setShowLogoutModal(false)}
              >
                <Text style={s.sheetCancelText}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[s.sheetConfirm, { backgroundColor: DANGER }]}
                onPress={async () => {
                  setShowLogoutModal(false);
                  await logoutEnterprise();
                  router.replace("/(enterprise-auth)/login" as any);
                }}
              >
                <Ionicons name="log-out-outline" size={15} color="#fff" />
                <Text style={s.sheetConfirmText}>Se déconnecter</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Pressable>
      </Modal>
    </>
  );
}

function InfoRow({
  label,
  value,
  last,
}: {
  label: string;
  value: string;
  last?: boolean;
}) {
  return (
    <View style={[s.infoRow, last && s.infoRowLast]}>
      <Text style={s.infoLabel}>{label}</Text>
      <Text style={s.infoValue} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

const s = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BG,
  },

  // ——— Scroll ———
  scrollContent: {
    padding: 16,
    paddingBottom: Platform.OS === "ios" ? 94 : 84,
    gap: 12,
  },

  // ——— Cards ———
  card: {
    backgroundColor: CARD,
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: BORDER,
    ...cardShadow,
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: TEXT,
    marginLeft: 10,
    flex: 1,
    letterSpacing: -0.1,
  },
  cardMessage: {
    fontSize: 13,
    color: TEXT_SEC,
    lineHeight: 19,
    marginBottom: 4,
  },

  // ——— Info rows ———
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(0,121,107,0.05)",
  },
  infoRowLast: {
    borderBottomWidth: 0,
  },
  infoLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: TEXT_SEC,
    letterSpacing: 0.2,
    textTransform: "uppercase",
    flex: 1,
  },
  infoValue: {
    fontSize: 14,
    fontWeight: "500",
    color: TEXT,
    textAlign: "right",
    flex: 1.5,
  },

  // ——— Buttons ———
  primaryBtn: {
    backgroundColor: BRAND,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    gap: 8,
    marginTop: 8,
    ...btnShadow,
  },
  primaryBtnDisabled: {
    opacity: 0.6,
  },
  primaryBtnText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
  dangerBtn: {
    backgroundColor: DANGER,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    gap: 8,
    marginTop: 8,
    ...Platform.select({
      ios: {
        shadowColor: DANGER,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 6,
      },
      android: { elevation: 3 },
      default: {},
    }),
  },
  dangerBtnText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },

  bottomSpacing: {
    height: 20,
  },

  // ——— Modal overlay ———
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.45)",
    justifyContent: "flex-end",
  },

  // ——— Bottom sheet ———
  sheetContainer: {
    backgroundColor: CARD,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: Platform.OS === "ios" ? 36 : 20,
    alignItems: "center",
    ...sheetShadow,
  },
  sheetHandle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(0,0,0,0.1)",
    marginBottom: 18,
  },
  sheetIcon: {
    width: 52,
    height: 52,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
  },
  sheetTitle: {
    fontSize: 17,
    fontWeight: "600",
    color: TEXT,
    textAlign: "center",
    letterSpacing: -0.2,
  },
  sheetMessage: {
    fontSize: 13,
    color: TEXT_SEC,
    textAlign: "center",
    lineHeight: 19,
    marginTop: 8,
    maxWidth: 300,
  },
  sheetActions: {
    flexDirection: "row",
    gap: 10,
    width: "100%",
    marginTop: 20,
  },
  sheetCancel: {
    flex: 1,
    height: 42,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: BG,
    borderWidth: 1,
    borderColor: "rgba(0,0,0,0.08)",
  },
  sheetCancelText: {
    fontSize: 14,
    fontWeight: "500",
    color: TEXT_SEC,
  },
  sheetConfirm: {
    flex: 1.2,
    height: 42,
    borderRadius: 10,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  sheetConfirmText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#fff",
  },
});
