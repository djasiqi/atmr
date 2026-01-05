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
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";

import { router } from "expo-router";
import { useAuth } from "@/hooks/useAuth";
import {
  getMyDriverAccount,
  DriverAccountInfo,
  switchToDriverToken,
} from "@/services/enterpriseDispatch";
import { secureStorage, asyncStorage } from "@/services/storage";
import { fetchDriverProfile, invalidateInterceptorCache } from "@/services/api";

// ✅ Palette professionnelle cohérente avec le dashboard driver
const palette = {
  background: "#F5F7F6",
  heroGradient: ["#0A7F59", "#0D5F3F"] as [string, string],
  heroBorder: "rgba(15,54,43,0.08)",
  heroText: "#FFFFFF",
  heroMeta: "rgba(255,255,255,0.9)",
  cardBg: "#FFFFFF",
  cardBorder: "rgba(15,54,43,0.08)",
  cardShadow: "rgba(15,54,43,0.08)",
  muted: "#91A59D",
  primary: "#0A7F59",
  primaryText: "#FFFFFF",
  logoutBg: "rgba(239,68,68,0.08)",
  logoutBorder: "rgba(239,68,68,0.2)",
  error: "#EF4444",
  switchBg: "rgba(10,127,89,0.1)",
  switchBorder: "rgba(10,127,89,0.25)",
};

export default function EnterpriseSettingsScreen() {
  const { enterpriseSession, logoutEnterprise, switchMode, loadDriverSession } = useAuth();

  const [driverAccount, setDriverAccount] = useState<DriverAccountInfo | null>(null);
  const [checkingDriverAccount, setCheckingDriverAccount] = useState(true);
  const [switchingToDriver, setSwitchingToDriver] = useState(false);
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [showSwitchModal, setShowSwitchModal] = useState(false);

  // Charger l'info du compte chauffeur (avec cache AsyncStorage)
  useEffect(() => {
    const checkDriverAccount = async () => {
      setCheckingDriverAccount(true);
      try {
        // 1. D'abord, essayer de charger depuis le cache
        const cachedInfo = await asyncStorage.getDriverAccountInfo();
        if (cachedInfo) {
          console.log("[Settings] Info compte chauffeur chargée depuis le cache:", cachedInfo);
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:checkDriverAccount', message: 'Driver account from cache', data: { hasDriverAccount: cachedInfo.has_driver_account, driverType: cachedInfo.driver_type }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
          // #endregion
          setDriverAccount(cachedInfo);
          setCheckingDriverAccount(false);

          // 2. En arrière-plan, vérifier si l'info est toujours à jour
          try {
            const freshInfo = await getMyDriverAccount();
            console.log("[Settings] Réponse getMyDriverAccount (fresh):", freshInfo);
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:checkDriverAccount', message: 'Driver account from API', data: { hasDriverAccount: freshInfo.has_driver_account, driverType: freshInfo.driver_type }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
            // #endregion
            // Mettre à jour le cache et l'état si différent
            if (JSON.stringify(cachedInfo) !== JSON.stringify(freshInfo)) {
              await asyncStorage.setDriverAccountInfo(freshInfo);
              setDriverAccount(freshInfo);
            }
          } catch (error) {
            // Si l'appel échoue, on garde le cache
            console.warn("[Settings] Impossible de rafraîchir l'info du compte chauffeur:", error);
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:checkDriverAccount', message: 'Error refreshing driver account', data: { error: String(error) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
            // #endregion
          }
          return;
        }

        // 3. Si pas de cache, faire l'appel API
        console.log("[Settings] Vérification du compte chauffeur...");
        const info = await getMyDriverAccount();
        console.log("[Settings] Réponse getMyDriverAccount:", info);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:checkDriverAccount', message: 'Driver account from API (no cache)', data: { hasDriverAccount: info.has_driver_account, driverType: info.driver_type }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
        // #endregion
        setDriverAccount(info);
        // Sauvegarder dans le cache
        await asyncStorage.setDriverAccountInfo(info);
      } catch (error: any) {
        console.error("[Settings] Erreur lors de la vérification du compte chauffeur:", error);
        console.error("[Settings] Détails de l'erreur:", {
          message: error?.message,
          response: error?.response?.data,
          status: error?.response?.status,
        });
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:checkDriverAccount', message: 'Error checking driver account', data: { error: String(error), status: error?.response?.status }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
        // #endregion
        setDriverAccount({ has_driver_account: false });
        // Sauvegarder dans le cache même en cas d'erreur
        await asyncStorage.setDriverAccountInfo({ has_driver_account: false });
      } finally {
        setCheckingDriverAccount(false);
      }
    };
    checkDriverAccount();
  }, []);

  // Fonction pour basculer vers le compte chauffeur
  const handleSwitchToDriver = useCallback(async () => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:handleSwitchToDriver', message: 'handleSwitchToDriver entry', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
    // #endregion

    setShowSwitchModal(false);
    setSwitchingToDriver(true);
    try {
      // 1. Obtenir un token driver à partir du token entreprise
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:handleSwitchToDriver', message: 'Avant switchToDriverToken', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
      // #endregion

      const driverTokenResponse = await switchToDriverToken();

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:handleSwitchToDriver', message: 'Après switchToDriverToken', data: { hasToken: !!driverTokenResponse.token, hasRefreshToken: !!driverTokenResponse.refresh_token, userPublicId: driverTokenResponse.user?.public_id }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
      // #endregion

      console.log("[Settings] Tokens driver reçus:", {
        hasToken: !!driverTokenResponse.token,
        hasRefreshToken: !!driverTokenResponse.refresh_token,
        userPublicId: driverTokenResponse.user.public_id,
      });

      // 2. Stocker les tokens driver AVANT de nettoyer l'entreprise
      // Cela garantit que les tokens driver sont sauvegardés avant toute opération de nettoyage
      await secureStorage.setAccessToken(driverTokenResponse.token);
      if (driverTokenResponse.refresh_token) {
        await secureStorage.setRefreshToken(driverTokenResponse.refresh_token);
      }
      if (driverTokenResponse.user.public_id) {
        await secureStorage.setUserPublicId(driverTokenResponse.user.public_id);
      }
      // Invalider le cache de l'intercepteur pour forcer l'utilisation des nouveaux tokens driver
      invalidateInterceptorCache();
      console.log("[Settings] Tokens driver stockés dans SecureStorage et cache intercepteur invalidé");

      // 3. Basculer vers le mode driver AVANT de nettoyer l'entreprise
      // Cela garantit que le contexte auth est mis à jour avec le bon mode
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:handleSwitchToDriver', message: 'Avant switchMode driver', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
      // #endregion

      await switchMode("driver");

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:handleSwitchToDriver', message: 'Après switchMode driver', data: {}, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
      // #endregion

      console.log("[Settings] Mode changé vers 'driver'");

      // 4. Nettoyer l'entreprise (cela ne devrait pas affecter les tokens driver déjà stockés)
      await logoutEnterprise();
      console.log("[Settings] Stockage entreprise nettoyé");

      // 5. Charger la session driver depuis SecureStorage et mettre à jour le contexte
      // Au lieu d'appeler fetchDriverProfile et refreshProfile (qui peuvent échouer),
      // nous chargeons directement la session que nous venons de stocker
      await loadDriverSession();
      console.log("[Settings] Session driver chargée depuis SecureStorage");

      // 6. Attendre un peu pour que le state se mette à jour
      await new Promise(resolve => setTimeout(resolve, 200));

      // 7. Naviguer vers la page driver
      // Le système de navigation dans _layout.tsx gérera automatiquement la redirection
      // mais on force la navigation ici pour être sûr
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:handleSwitchToDriver', message: 'Avant navigation mission', data: { target: '/(tabs)/mission' }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
      // #endregion

      router.replace("/(tabs)/mission" as any);
    } catch (error: any) {
      console.error("Erreur lors du basculement:", error);
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: 'settings.tsx:handleSwitchToDriver', message: 'Erreur handleSwitchToDriver', data: { error: String(error), status: error?.response?.status, errorMessage: error?.response?.data?.error }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'I' }) }).catch(() => { });
      // #endregion

      const errorMessage =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de basculer vers le compte chauffeur.";
      Alert.alert("Erreur", errorMessage);
    } finally {
      setSwitchingToDriver(false);
    }
  }, [logoutEnterprise, switchMode, loadDriverSession]);

  const heroSubtitle = "Gérez votre compte et basculez entre les modes entreprise et chauffeur.";

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <LinearGradient
        colors={palette.heroGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.hero}
      >
        <View style={{ flex: 1 }}>
          <Text style={styles.heroTitle}>Paramètres</Text>
          <Text style={styles.heroSubtitle}>{heroSubtitle}</Text>
        </View>
        <View style={styles.heroBadge}>
          <Ionicons name="settings-outline" size={18} color={palette.primaryText} />
          <Text style={styles.heroBadgeText}>
            {enterpriseSession?.company?.name || "Entreprise"}
          </Text>
        </View>
      </LinearGradient>

      {/* Indicateur de chargement */}
      {checkingDriverAccount && (
        <View style={styles.card}>
          <View style={styles.switchAccountHeader}>
            <ActivityIndicator size="small" color={palette.primary} />
            <Text style={[styles.sectionDescription, { marginLeft: 12 }]}>
              Vérification du compte chauffeur...
            </Text>
          </View>
        </View>
      )}

      {/* Message de débogage si pas de compte chauffeur */}
      {!checkingDriverAccount && driverAccount && !driverAccount.has_driver_account && (
        <View style={styles.card}>
          <View style={styles.switchAccountHeader}>
            <Ionicons name="information-circle-outline" size={24} color={palette.muted} />
            <Text style={[styles.sectionDescription, { marginLeft: 12, flex: 1 }]}>
              Aucun compte chauffeur associé à votre compte entreprise.
            </Text>
          </View>
        </View>
      )}

      {/* Section Switch de compte */}
      {!checkingDriverAccount && driverAccount?.has_driver_account && (
        <View style={styles.card}>
          <View style={styles.switchAccountHeader}>
            <Ionicons name="swap-horizontal" size={24} color={palette.primary} />
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.sectionTitle}>Compte chauffeur</Text>
              <Text style={styles.sectionDescription}>
                {driverAccount.driver_type === "EMERGENCY"
                  ? "Vous êtes également chauffeur d'urgence"
                  : "Vous avez également un compte chauffeur"}
              </Text>
            </View>
          </View>

          <TouchableOpacity
            style={[styles.switchButton, switchingToDriver && styles.switchButtonDisabled]}
            onPress={() => setShowSwitchModal(true)}
            disabled={checkingDriverAccount || switchingToDriver}
          >
            {switchingToDriver ? (
              <>
                <ActivityIndicator size="small" color={palette.primaryText} />
                <Text style={styles.switchButtonText}>Basculement en cours...</Text>
              </>
            ) : (
              <>
                <Ionicons name="car-outline" size={20} color={palette.primaryText} />
                <Text style={styles.switchButtonText}>
                  Basculer vers le compte chauffeur
                </Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      )}

      {/* Bouton de déconnexion */}
      <TouchableOpacity
        style={styles.logoutButton}
        onPress={() => setShowLogoutModal(true)}
      >
        <Text style={styles.logoutButtonText}>Se déconnecter</Text>
      </TouchableOpacity>

      {/* Modal de confirmation de switch */}
      <Modal
        visible={showSwitchModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowSwitchModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <View style={styles.modalIconContainer}>
              <Ionicons
                name="swap-horizontal"
                size={32}
                color={palette.primary}
              />
            </View>
            <Text style={styles.modalTitle}>Basculer vers le compte chauffeur</Text>
            <Text style={styles.modalMessage}>
              Vous allez être automatiquement connecté en tant que chauffeur. La connexion au compte entreprise sera fermée.
            </Text>
            <View style={styles.modalActions}>
              <Pressable
                style={styles.modalCancel}
                onPress={() => setShowSwitchModal(false)}
              >
                <Text style={styles.modalCancelText}>Annuler</Text>
              </Pressable>
              <Pressable
                style={styles.modalConfirm}
                onPress={handleSwitchToDriver}
              >
                <Text style={styles.modalConfirmText}>Continuer</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      {/* Modal de confirmation de déconnexion */}
      <Modal
        visible={showLogoutModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowLogoutModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <View style={styles.modalIconContainer}>
              <Ionicons
                name="log-out-outline"
                size={32}
                color={palette.error}
              />
            </View>
            <Text style={styles.modalTitle}>Déconnexion</Text>
            <Text style={styles.modalMessage}>
              Voulez-vous quitter l'espace entreprise ?
            </Text>
            <View style={styles.modalActions}>
              <Pressable
                style={styles.modalCancel}
                onPress={() => setShowLogoutModal(false)}
              >
                <Text style={styles.modalCancelText}>Annuler</Text>
              </Pressable>
              <Pressable
                style={styles.modalConfirm}
                onPress={async () => {
                  setShowLogoutModal(false);
                  await logoutEnterprise();
                  await switchMode("driver");
                }}
              >
                <Text style={styles.modalConfirmText}>Se déconnecter</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: palette.background,
  },
  content: {
    padding: 20,
    paddingBottom: Platform.OS === "ios" ? 94 : 84,
    gap: 22,
  },
  hero: {
    borderRadius: 24,
    padding: 24,
    flexDirection: "row",
    alignItems: "center",
    gap: 18,
    borderWidth: 1,
    borderColor: palette.heroBorder,
    shadowColor: "rgba(10,127,89,0.15)",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 1,
    shadowRadius: 24,
    elevation: 8,
  },
  heroTitle: {
    color: palette.heroText,
    fontSize: 26,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  heroSubtitle: {
    color: palette.heroMeta,
    fontSize: 14,
    marginTop: 6,
  },
  heroBadge: {
    backgroundColor: palette.primary,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: palette.cardBorder,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  heroBadgeText: {
    color: palette.primaryText,
    fontWeight: "700",
    fontSize: 13,
  },
  card: {
    backgroundColor: palette.cardBg,
    borderRadius: 20,
    padding: 22,
    borderWidth: 1,
    borderColor: palette.cardBorder,
    shadowColor: palette.cardShadow,
    shadowOpacity: 1,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 12,
    elevation: 2,
    gap: 16,
  },
  switchAccountHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  sectionTitle: {
    color: "#15362B",
    fontSize: 18,
    fontWeight: "700",
    letterSpacing: -0.2,
  },
  sectionDescription: {
    color: palette.muted,
    fontSize: 14,
    lineHeight: 20,
    marginTop: 4,
  },
  switchButton: {
    backgroundColor: palette.primary,
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 20,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    shadowColor: palette.primary,
    shadowOpacity: 0.35,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 12,
    elevation: 6,
  },
  switchButtonDisabled: {
    opacity: 0.6,
  },
  switchButtonText: {
    color: palette.primaryText,
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.4,
  },
  logoutButton: {
    backgroundColor: palette.logoutBg,
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: palette.logoutBg,
    shadowOpacity: 0.35,
    shadowOffset: { width: 0, height: 10 },
    shadowRadius: 18,
    elevation: 6,
  },
  logoutButtonText: {
    color: palette.heroText,
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.4,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(5,22,16,0.82)",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  modalCard: {
    width: "100%",
    maxWidth: 400,
    backgroundColor: palette.cardBg,
    borderRadius: 24,
    padding: 24,
    borderWidth: 1,
    borderColor: palette.cardBorder,
    gap: 20,
    alignItems: "center",
  },
  modalIconContainer: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "rgba(10,127,89,0.15)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 4,
  },
  modalTitle: {
    color: "#15362B",
    fontSize: 22,
    fontWeight: "700",
    textAlign: "center",
  },
  modalMessage: {
    color: palette.muted,
    fontSize: 15,
    lineHeight: 22,
    textAlign: "center",
  },
  modalActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 12,
    width: "100%",
    marginTop: 8,
  },
  modalCancel: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: palette.cardBorder,
  },
  modalCancelText: {
    color: palette.muted,
    fontWeight: "600",
    fontSize: 15,
  },
  modalConfirm: {
    backgroundColor: palette.primary,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 14,
    shadowColor: palette.primary,
    shadowOpacity: 0.35,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 12,
    elevation: 6,
  },
  modalConfirmText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 15,
  },
});
