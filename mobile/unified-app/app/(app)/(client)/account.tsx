import { Pressable, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import { useClientProfileQuery } from "../../../src/features/client/hooks";
import { useClientBottomContentPadding } from "../../../src/features/client/navigation/ClientFloatingAppBar";

export default function ClientAccountScreen() {
  const { logout } = useSession();
  const insets = useSafeAreaInsets();
  const bottomPad = useClientBottomContentPadding();
  const profileQuery = useClientProfileQuery();

  return (
    <PermissionGuard permission="profile:read:self">
      <View
        style={{
          flex: 1,
          padding: 24,
          paddingTop: Math.max(24, insets.top + 8),
          paddingBottom: bottomPad,
          gap: 12,
        }}
      >
        <Text style={{ fontSize: 22, fontWeight: "700" }}>Mon compte</Text>
        <Text style={{ color: "#475569" }}>
          Edition limitee aux informations supportees par le backend actuel.
        </Text>
        {profileQuery.isLoading ? <Text>Chargement du profil...</Text> : null}
        {profileQuery.isError ? (
          <Text>
            Impossible de charger le profil: {(profileQuery.error as Error)?.message ?? "Erreur"}
          </Text>
        ) : null}
        {profileQuery.data ? (
          <>
            <Text>
              {(
                profileQuery.data.full_name ??
                `${profileQuery.data.first_name ?? ""} ${profileQuery.data.last_name ?? ""}`.trim()
              ) || "Nom non renseigné"}
            </Text>
            <Text>{profileQuery.data.contact_email ?? profileQuery.data.user?.email ?? "Email non renseigné"}</Text>
            <Text>{profileQuery.data.phone ?? "Téléphone non renseigné"}</Text>
            <Text>{profileQuery.data.domicile?.address ?? "Adresse non renseignée"}</Text>
          </>
        ) : null}

        <Pressable
          onPress={() => logout()}
          style={{
            marginTop: 16,
            borderRadius: 8,
            backgroundColor: "#0a7ea4",
            paddingVertical: 10,
            paddingHorizontal: 14,
            alignSelf: "flex-start",
          }}
        >
          <Text style={{ color: "#fff", fontWeight: "600" }}>Déconnexion</Text>
        </Pressable>
        <Text style={{ color: "#64748b" }}>
          Besoin d&apos;aide ? Contactez le support depuis l&apos;ecran public d&apos;aide.
        </Text>
      </View>
    </PermissionGuard>
  );
}
