import { useEffect, useState, type ReactNode } from "react";
import { Keyboard, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { brandPrimary, brandText, useAccessibilityScale, useResponsiveTokens } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { E } from "../../theme/enterpriseOpsTheme";
import { type RideClientOption, useCompanyClientSearch } from "../../useRideForms";
import { CreateRideKeyboardSheet } from "./CreateRideKeyboardSheet";

const ROW_RADIUS = 12;
const UI_SEPARATOR = "rgba(15, 23, 42, 0.08)";

type CreateClientTriggerProps = {
  selectedId: number | null;
  selectedLabel: string;
  selectedSubtitle?: string;
  onPress: () => void;
  onClear: () => void;
  leftSlot?: ReactNode;
};

/** Carte client fermée, ou champ « rechercher » — jamais de dropdown inline. */
export function CreateClientTrigger({
  selectedId,
  selectedLabel,
  selectedSubtitle,
  onPress,
  onClear,
  leftSlot,
}: CreateClientTriggerProps) {
  const t = useResponsiveTokens();
  if (selectedId != null && selectedLabel.trim().length > 0) {
    return (
      <View style={[s.card, { minHeight: Math.max(t.fieldShellMinHeight, 48) }]}>
        <Pressable
          onPress={onPress}
          style={s.cardMain}
          accessibilityRole="button"
          accessibilityLabel={`Client ${selectedLabel}. Toucher pour changer`}
        >
          {leftSlot}
          <View style={s.cardText}>
            <AppText variant="body" numberOfLines={1} style={s.cardTitle}>
              {selectedLabel}
            </AppText>
            {selectedSubtitle ? (
              <AppText variant="caption" numberOfLines={1} style={s.cardSubtitle}>
                {selectedSubtitle}
              </AppText>
            ) : null}
          </View>
        </Pressable>
        <Pressable
          onPress={onClear}
          accessibilityRole="button"
          accessibilityLabel="Effacer le client"
          hitSlop={8}
          style={s.clearHit}
        >
          <Ionicons name="close" size={18} color={E.TEXT_SEC} />
        </Pressable>
      </View>
    );
  }

  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel="Rechercher un client"
      style={[s.emptyField, { minHeight: Math.max(t.fieldShellMinHeight, 48) }]}
    >
      {leftSlot}
      <AppText variant="body" style={s.emptyPlaceholder} numberOfLines={1}>
        Rechercher un client…
      </AppText>
      <AppText
        variant="label"
        accessibilityLabel="Champ obligatoire"
        style={s.requiredMark}
      >
        *
      </AppText>
    </Pressable>
  );
}

type ClientPickerSheetProps = {
  visible: boolean;
  selectedId: number | null;
  onClose: () => void;
  onSelect: (client: RideClientOption) => void;
  onCreateClient?: () => void;
};

export function ClientPickerSheet({
  visible,
  selectedId,
  onClose,
  onSelect,
  onCreateClient,
}: ClientPickerSheetProps) {
  const t = useResponsiveTokens();
  const { isVeryLargeText } = useAccessibilityScale();
  const suggestionLines = isVeryLargeText ? undefined : 1;
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");

  useEffect(() => {
    if (!visible) {
      setQuery("");
      setDebouncedQuery("");
    }
  }, [visible]);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query.trim()), 250);
    return () => clearTimeout(timer);
  }, [query]);

  const clientsQuery = useCompanyClientSearch(visible ? debouncedQuery : "");
  const trimmedQuery = query.trim();
  const canSearch = trimmedQuery.length > 1;
  const results = canSearch ? clientsQuery.data ?? [] : [];

  return (
    <CreateRideKeyboardSheet
      visible={visible}
      title="Sélectionner un client"
      subtitle="Recherchez puis choisissez un client"
      onClose={onClose}
      search={
        <View style={s.searchOffset}>
          <AppInput
            value={query}
            onChangeText={setQuery}
            placeholder="Rechercher un client…"
            autoFocus={visible}
            leftSlot={<Ionicons name="search-outline" size={18} color={E.TEXT_SEC} />}
            shellStyle={{ borderRadius: ROW_RADIUS, minHeight: 48, paddingHorizontal: 10 }}
          />
        </View>
      }
      footer={
        onCreateClient ? (
          <Pressable
            onPress={() => {
              Keyboard.dismiss();
              onCreateClient();
            }}
            accessibilityRole="button"
            accessibilityLabel="Créer un nouveau client"
            style={[s.createRow, { minHeight: Math.max(t.minTouchHeight, 48) }]}
          >
            <AppText variant="body" style={s.createLabel}>
              + Nouveau client
            </AppText>
          </Pressable>
        ) : null
      }
    >
      {!canSearch ? (
        <AppText variant="caption" style={s.hint}>
          Saisissez au moins 2 caractères.
        </AppText>
      ) : null}
      {canSearch && (clientsQuery.isLoading || clientsQuery.isFetching) ? (
        <AppText variant="caption" style={s.hint}>
          Recherche…
        </AppText>
      ) : null}
      {results.map((client, index) => (
        <Pressable
          key={client.id}
          accessibilityRole="button"
          accessibilityLabel={`Sélectionner le client ${client.label}`}
          accessibilityState={{ selected: selectedId === client.id }}
          onPress={() => {
            Keyboard.dismiss();
            onSelect(client);
          }}
          style={({ pressed }) => [
            s.resultRow,
            { minHeight: Math.max(t.minTouchHeight, 52) },
            index < results.length - 1 ? s.resultRowBorder : null,
            selectedId === client.id ? s.resultRowSelected : null,
            pressed ? s.resultRowPressed : null,
          ]}
        >
          <AppText
            variant="body"
            numberOfLines={suggestionLines}
            style={{
              color: selectedId === client.id ? brandPrimary : brandText,
              fontWeight: selectedId === client.id ? "600" : "500",
            }}
          >
            {client.label}
          </AppText>
          {client.pickupAddressCandidate?.label &&
          client.pickupAddressCandidate.label.trim() &&
          client.pickupAddressCandidate.label !== client.label ? (
            <AppText variant="caption" numberOfLines={suggestionLines} style={s.resultSub}>
              {client.pickupAddressCandidate.label}
            </AppText>
          ) : null}
        </Pressable>
      ))}
    </CreateRideKeyboardSheet>
  );
}

const s = StyleSheet.create({
  searchOffset: { marginTop: 10 },
  createRow: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(15, 23, 42, 0.08)",
    paddingVertical: 10,
    justifyContent: "center",
  },
  createLabel: { color: brandPrimary, fontWeight: "600" },
  card: {
    flexDirection: "row",
    alignItems: "center",
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.38)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 10,
    paddingVertical: 8,
    gap: 8,
  },
  cardMain: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    minWidth: 0,
  },
  cardText: { flex: 1, minWidth: 0, gap: 2 },
  cardTitle: { color: E.TEXT, fontWeight: "600" },
  cardSubtitle: { color: "rgba(71, 85, 105, 0.85)" },
  clearHit: {
    width: 36,
    height: 36,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyField: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.38)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 10,
  },
  emptyPlaceholder: {
    flex: 1,
    color: "rgba(100, 116, 139, 0.92)",
  },
  requiredMark: {
    color: "#DC2626",
    fontWeight: "700",
    fontSize: FONT_SIZE.px16,
    lineHeight: 18,
    width: 16,
    textAlign: "center",
  },
  hint: {
    paddingVertical: 8,
    color: "rgba(71, 85, 105, 0.92)",
  },
  resultRow: {
    paddingVertical: 10,
    paddingHorizontal: 4,
    justifyContent: "center",
    gap: 2,
  },
  resultRowBorder: {
    borderBottomWidth: 1,
    borderBottomColor: UI_SEPARATOR,
  },
  resultRowSelected: {
    backgroundColor: "rgba(0, 121, 107, 0.09)",
    borderRadius: 10,
  },
  resultRowPressed: {
    backgroundColor: "rgba(15, 23, 42, 0.06)",
  },
  resultSub: { color: "rgba(71, 85, 105, 0.85)" },
});
