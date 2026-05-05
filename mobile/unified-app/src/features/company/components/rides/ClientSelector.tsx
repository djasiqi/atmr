import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { Pressable, ScrollView, View } from "react-native";
import { brandPrimary, brandText, useResponsiveTokens } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { type RideClientOption, useCompanyClientSearch } from "../../useRideForms";

const UI_BORDER_SOFT = "rgba(15, 23, 42, 0.12)";
const UI_SEPARATOR = "rgba(15, 23, 42, 0.08)";
const UI_BG_LIST = "rgba(248, 250, 252, 0.95)";
const UI_BG_HOVER = "rgba(15, 23, 42, 0.03)";
const UI_BG_PRESSED = "rgba(15, 23, 42, 0.06)";
const UI_BG_SELECTED = "rgba(0, 121, 107, 0.09)";
const UI_BG_SELECTED_PRESSED = "rgba(0, 121, 107, 0.14)";
const ROW_RADIUS = 12;
const RESULTS_ROW_HEIGHT = 58;
const RESULTS_VISIBLE_ROWS = 4;
const RESULTS_MAX_HEIGHT = RESULTS_ROW_HEIGHT * RESULTS_VISIBLE_ROWS;

type ClientSelectorProps = {
  value: number | null;
  onChange: (clientId: number | null) => void;
  onSelectClient?: (client: RideClientOption) => void;
  onCreateClient?: () => void;
  /** Faux si le titre est déjà affiché au-dessus (ex. section modale). */
  showFieldLabel?: boolean;
  fieldLabel?: string;
  placeholder?: string;
  helperText?: string;
  leftSlot?: ReactNode;
};

export function ClientSelector({
  value,
  onChange,
  onSelectClient,
  onCreateClient,
  showFieldLabel = true,
  fieldLabel = "Client *",
  placeholder = "Rechercher un client…",
  helperText,
  leftSlot,
}: ClientSelectorProps) {
  const t = useResponsiveTokens();
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [isListOpen, setIsListOpen] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query.trim()), 250);
    return () => clearTimeout(timer);
  }, [query]);
  const clientsQuery = useCompanyClientSearch(debouncedQuery);
  const trimmedQuery = query.trim();
  const canSearchClients = trimmedQuery.length > 1;
  const resultsCount = clientsQuery.data?.length ?? 0;
  const shouldShowResults = canSearchClients && isListOpen && resultsCount > 0;
  const shouldShowCreateClient =
    Boolean(onCreateClient) &&
    canSearchClients &&
    isListOpen &&
    !clientsQuery.isLoading &&
    !clientsQuery.isFetching &&
    resultsCount === 0;
  return (
    <View style={{ gap: t.fieldGap }}>
      {showFieldLabel ? <AppText variant="label">{fieldLabel}</AppText> : null}
      <AppInput
        value={query}
        onChangeText={(nextValue) => {
          setQuery(nextValue);
          setIsListOpen(true);
        }}
        placeholder={placeholder}
        leftSlot={leftSlot}
        shellStyle={{ borderRadius: ROW_RADIUS, minHeight: Math.max(t.fieldShellMinHeight, 48) }}
        helperText={helperText}
      />
      {shouldShowResults ? (
        <ScrollView
          style={{
            borderWidth: 1,
            borderColor: UI_BORDER_SOFT,
            borderRadius: ROW_RADIUS,
            backgroundColor: UI_BG_LIST,
            overflow: "hidden",
            maxHeight: RESULTS_MAX_HEIGHT,
          }}
          nestedScrollEnabled
          showsVerticalScrollIndicator
          keyboardShouldPersistTaps="handled"
        >
          {clientsQuery.data?.map((client, index) => (
            <Pressable
              key={client.id}
              onPress={() => {
                onChange(client.id);
                onSelectClient?.(client);
                setQuery(client.label);
                setDebouncedQuery(client.label);
                setIsListOpen(false);
              }}
              accessibilityRole="button"
              accessibilityLabel={`Sélectionner le client ${client.label}`}
              accessibilityState={{ selected: value === client.id }}
              style={({ hovered, pressed }) => {
                const isSelected = value === client.id;
                let backgroundColor = isSelected ? UI_BG_SELECTED : hovered ? UI_BG_HOVER : "transparent";
                if (pressed) backgroundColor = isSelected ? UI_BG_SELECTED_PRESSED : UI_BG_PRESSED;
                return {
                  minHeight: Math.max(t.minTouchHeight, 48),
                  paddingVertical: 10,
                  paddingHorizontal: 14,
                  justifyContent: "center",
                  gap: 2,
                  backgroundColor,
                  borderBottomWidth: index < resultsCount - 1 ? 1 : 0,
                  borderBottomColor: index < resultsCount - 1 ? UI_SEPARATOR : "transparent",
                  cursor: "pointer",
                };
              }}
            >
              <AppText
                variant="body"
                numberOfLines={1}
                style={{
                  color: value === client.id ? brandPrimary : brandText,
                  fontWeight: value === client.id ? "600" : "500",
                }}
              >
                {client.label}
              </AppText>
              {client.pickupAddressCandidate?.label &&
              client.pickupAddressCandidate.label.trim() &&
              client.pickupAddressCandidate.label !== client.label ? (
                <AppText
                  variant="caption"
                  numberOfLines={1}
                  style={{ color: "rgba(71, 85, 105, 0.85)" }}
                >
                  {client.pickupAddressCandidate.label}
                </AppText>
              ) : null}
            </Pressable>
          ))}
        </ScrollView>
      ) : null}
      {clientsQuery.isLoading ? <AppText variant="caption">Recherche…</AppText> : null}
      {shouldShowCreateClient ? (
        <Pressable
          onPress={() => {
            setIsListOpen(false);
            onCreateClient?.();
          }}
          accessibilityRole="button"
          accessibilityLabel="Créer un nouveau client"
        >
          <AppText variant="body" style={{ color: brandPrimary, fontWeight: "600" }}>
            + Nouveau client
          </AppText>
        </Pressable>
      ) : null}
    </View>
  );
}
