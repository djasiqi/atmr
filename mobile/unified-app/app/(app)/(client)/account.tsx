import { View } from "react-native";
import { PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import { useClientProfileQuery } from "../../../src/features/client/hooks";
import { useClientBottomContentPadding } from "../../../src/features/client/navigation/ClientFloatingAppBar";
import {
  AppButton,
  AppCard,
  AppText,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../src/design/responsive";

export default function ClientAccountScreen() {
  const { logout } = useSession();
  const bottomPad = useClientBottomContentPadding();
  const profileQuery = useClientProfileQuery();
  const { horizontalPadding } = useAppViewport();
  const t = useResponsiveTokens();

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

        <AppButton title="Déconnexion" variant="primary" onPress={() => logout()} />
        <AppText variant="caption">
          Besoin d&apos;aide ? Contactez le support depuis l&apos;écran public d&apos;aide.
        </AppText>
      </Screen>
    </PermissionGuard>
  );
}
