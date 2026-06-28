import { Pressable, StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import { Ionicons } from "@expo/vector-icons";
import { AppEmptyState } from "../../../../design/ui/AppEmptyState";
import type { InboxTab } from "../inboxDisplay";

type Props = {
  tab: InboxTab;
  hasSearch: boolean;
  urgentFilter: boolean;
  onOpenTeam?: () => void;
  onOpenDispatch?: () => void;
  onOpenColleagues?: () => void;
  onOpenSupport?: () => void;
};

export function MessagesInboxEmpty({
  tab,
  hasSearch,
  urgentFilter,
  onOpenTeam,
  onOpenDispatch,
  onOpenColleagues,
  onOpenSupport,
}: Props) {
  if (hasSearch) {
    return (
      <View style={styles.wrap}>
        <AppEmptyState
          icon={<Ionicons name="search-outline" size={40} color="#94A3B8" />}
          title="Aucun résultat"
          description="Essayez un autre mot-clé ou le numéro de mission."
        />
      </View>
    );
  }

  if (urgentFilter) {
    return (
      <View style={styles.wrap}>
        <AppEmptyState
          icon={<Ionicons name="notifications-off-outline" size={40} color="#94A3B8" />}
          title="Rien d'urgent"
          description="Toutes vos conversations sont à jour."
        />
      </View>
    );
  }

  if (tab === "missions") {
    return (
      <View style={styles.wrap}>
        <AppEmptyState
          icon={<Ionicons name="car-outline" size={40} color={M.BRAND} />}
          title="Pas de mission en cours"
          description="Le chat par mission est limité sur mobile. Utilisez le canal équipe ou le dispatch pour vos échanges."
        />
        <View style={styles.actions}>
          {onOpenTeam ? (
            <QuickAction icon="people-outline" label="Canal équipe" onPress={onOpenTeam} />
          ) : null}
          {onOpenDispatch ? (
            <QuickAction icon="business-outline" label="Dispatch" onPress={onOpenDispatch} />
          ) : null}
        </View>
      </View>
    );
  }

  if (tab === "contacts") {
    return (
      <View style={styles.wrap}>
        <AppEmptyState
          icon={<Ionicons name="business-outline" size={40} color={M.BRAND} />}
          title="Contacts"
          description="Dispatch de votre entreprise, canal équipe, collègues et assistance LIRIE."
        />
        <View style={styles.actions}>
          {onOpenDispatch ? (
            <QuickAction icon="business-outline" label="Dispatch" onPress={onOpenDispatch} />
          ) : null}
          {onOpenColleagues ? (
            <QuickAction icon="person-add-outline" label="Nouveau contact" onPress={onOpenColleagues} />
          ) : null}
          {onOpenSupport ? (
            <QuickAction icon="headset-outline" label="Support LIRIE" onPress={onOpenSupport} />
          ) : null}
        </View>
      </View>
    );
  }

  return (
    <View style={styles.wrap}>
      <AppEmptyState
        icon={<Ionicons name="chatbubbles-outline" size={40} color={M.BRAND} />}
        title="Pas encore de messages"
        description="Canal équipe, dispatch, collègues et assistance LIRIE."
        actionLabel={onOpenTeam ? "Ouvrir le canal équipe" : undefined}
        onAction={onOpenTeam ?? onOpenDispatch}
      />
    </View>
  );
}

function QuickAction({
  icon,
  label,
  onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable style={quickStyles.btn} onPress={onPress} accessibilityRole="button">
      <Ionicons name={icon} size={20} color={M.BRAND} />
      <AppText variant="body" style={quickStyles.label}>
        {label}
      </AppText>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: { paddingTop: 48, paddingHorizontal: 24, gap: 16 },
  actions: { gap: 10, width: "100%" },
});

const quickStyles = StyleSheet.create({
  btn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: "#F0FDF9",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#99F6E4",
  },
  label: { color: M.BRAND, fontWeight: "600" },
});
