import { useState } from "react";
import { Pressable, View } from "react-native";
import * as ExpoLinking from "expo-linking";
import { PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import { deleteClientAccount } from "../../../src/features/client/api";
import { useClientProfileQuery } from "../../../src/features/client/hooks";
import { useClientBottomContentPadding } from "../../../src/features/client/navigation/ClientFloatingAppBar";
import {
  AppButton,
  AppCard,
  AppText,
  Modal,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../src/design/responsive";

const TERMS_URL = "https://www.lirie.ch/conditions";
const PRIVACY_URL = "https://www.lirie.ch/privacy";

export default function ClientAccountScreen() {
  const { logout } = useSession();
  const bottomPad = useClientBottomContentPadding();
  const profileQuery = useClientProfileQuery();
  const { horizontalPadding } = useAppViewport();
  const t = useResponsiveTokens();
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [deletePending, setDeletePending] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleDeleteAccount = async () => {
    setDeletePending(true);
    setDeleteError(null);
    try {
      await deleteClientAccount();
      setDeleteModalVisible(false);
      await logout();
    } catch (error) {
      setDeleteError(
        error instanceof Error ? error.message : "Impossible de supprimer le compte."
      );
    } finally {
      setDeletePending(false);
    }
  };

  return (
    <PermissionGuard permission="profile:read:self">
      <Screen
        scroll
        backgroundColor="#f8fafc"
        withHorizontalPadding={false}
        includeSafeAreaInScrollBottomPadding={false}
        extraScrollBottomPadding={bottomPad}
        contentContainerStyle={{
          paddingHorizontal: horizontalPadding,
          paddingTop: t.spacingSm,
          gap: t.pageGap,
        }}
      >
        <AppText variant="screenTitle">Mon compte</AppText>
        <AppText variant="bodyMuted">
          Édition limitée aux informations supportées par le backend actuel.
        </AppText>
        {profileQuery.isLoading ? <AppText variant="bodyMuted">Chargement du profil…</AppText> : null}
        {profileQuery.isError ? (
          <AppText variant="error">
            Impossible de charger le profil : {(profileQuery.error as Error)?.message ?? "Erreur"}
          </AppText>
        ) : null}
        {profileQuery.data ? (
          <AppCard variant="surface">
            <View style={{ gap: t.fieldGap }}>
              <AppText variant="body">
                {(
                  profileQuery.data.full_name ??
                  `${profileQuery.data.first_name ?? ""} ${profileQuery.data.last_name ?? ""}`.trim()
                ) || "Nom non renseigné"}
              </AppText>
              <AppText variant="body">
                {profileQuery.data.contact_email ?? profileQuery.data.user?.email ?? "Email non renseigné"}
              </AppText>
              <AppText variant="body">{profileQuery.data.phone ?? "Téléphone non renseigné"}</AppText>
              <AppText variant="body">{profileQuery.data.domicile?.address ?? "Adresse non renseignée"}</AppText>
            </View>
          </AppCard>
        ) : null}

        <AppCard variant="surface">
          <View style={{ gap: t.fieldGap }}>
            <AppText variant="sectionTitle">Informations légales</AppText>
            <Pressable onPress={() => void ExpoLinking.openURL(PRIVACY_URL)}>
              <AppText variant="body" style={{ color: "#0A7F59", textDecorationLine: "underline" }}>
                Politique de confidentialité
              </AppText>
            </Pressable>
            <Pressable onPress={() => void ExpoLinking.openURL(TERMS_URL)}>
              <AppText variant="body" style={{ color: "#0A7F59", textDecorationLine: "underline" }}>
                Conditions d&apos;utilisation
              </AppText>
            </Pressable>
          </View>
        </AppCard>

        <AppButton title="Déconnexion" variant="primary" onPress={() => logout()} />
        <AppButton
          title="Supprimer mon compte"
          variant="secondary"
          onPress={() => {
            setDeleteError(null);
            setDeleteModalVisible(true);
          }}
        />
        <AppText variant="caption">
          Besoin d&apos;aide ? Contactez le support depuis l&apos;écran public d&apos;aide.
        </AppText>
      </Screen>

      <Modal
        visible={deleteModalVisible}
        title="Supprimer le compte"
        subtitle="Action irréversible"
        onClose={() => {
          if (!deletePending) setDeleteModalVisible(false);
        }}
        footer={
          <View style={{ gap: 10, width: "100%" }}>
            {deleteError ? <AppText variant="error">{deleteError}</AppText> : null}
            <AppButton
              title={deletePending ? "Suppression…" : "Confirmer la suppression"}
              variant="primary"
              loading={deletePending}
              disabled={deletePending}
              onPress={() => void handleDeleteAccount()}
            />
            <AppButton
              title="Annuler"
              variant="secondary"
              disabled={deletePending}
              onPress={() => setDeleteModalVisible(false)}
            />
          </View>
        }
      >
        <AppText variant="body">
          Votre compte client sera désactivé et vous serez déconnecté. Cette action est conforme
          aux exigences Google Play pour la suppression de compte in-app.
        </AppText>
      </Modal>
    </PermissionGuard>
  );
}
