import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Keyboard, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { brandText, useAccessibilityScale, useResponsiveTokens } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { E } from "../../theme/enterpriseOpsTheme";
import { type RideAddressOption, useCompanyAddressSearch } from "../../useRideForms";
import {
  ADDRESS_IGNORE_QUERIES,
  isAliasSuggestion,
  isGoogleLikeSuggestion,
  looksLikePoi,
  sortAddressSuggestions,
  splitAddressLabel,
} from "./addressSuggestionRank";
import { CreateRideKeyboardSheet } from "./CreateRideKeyboardSheet";

const ROW_RADIUS = 12;
const UI_SEPARATOR = "rgba(15, 23, 42, 0.08)";
const SUGGESTIONS_MAX_VISIBLE = 12;

type AddressFieldTriggerProps = {
  value: string;
  placeholder: string;
  required?: boolean;
  onPress: () => void;
  onClear: () => void;
  leftSlot?: ReactNode;
};

/** Carte adresse fermée — la saisie se fait dans le sheet clavier. */
export function AddressFieldTrigger({
  value,
  placeholder,
  required = false,
  onPress,
  onClear,
  leftSlot,
}: AddressFieldTriggerProps) {
  const t = useResponsiveTokens();
  const trimmed = value.trim();
  if (trimmed.length > 0) {
    return (
      <View style={[s.card, { minHeight: Math.max(t.fieldShellMinHeight, 48) }]}>
        <Pressable
          onPress={onPress}
          style={s.cardMain}
          accessibilityRole="button"
          accessibilityLabel={`${placeholder} ${trimmed}. Toucher pour changer`}
        >
          {leftSlot}
          <AppText variant="body" numberOfLines={2} style={s.cardTitle}>
            {trimmed}
          </AppText>
        </Pressable>
        <Pressable
          onPress={onClear}
          accessibilityRole="button"
          accessibilityLabel="Effacer l’adresse"
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
      accessibilityLabel={placeholder}
      style={[s.emptyField, { minHeight: Math.max(t.fieldShellMinHeight, 48) }]}
    >
      {leftSlot}
      <AppText variant="body" style={s.emptyPlaceholder} numberOfLines={1}>
        {placeholder}
      </AppText>
      {required ? (
        <AppText variant="label" accessibilityLabel="Champ obligatoire" style={s.requiredMark}>
          *
        </AppText>
      ) : null}
    </Pressable>
  );
}

type AddressPickerSheetProps = {
  visible: boolean;
  title: string;
  value: string;
  onClose: () => void;
  onChange: (value: string) => void;
  onSelect: (address: RideAddressOption) => void;
};

export function AddressPickerSheet({
  visible,
  title,
  value,
  onClose,
  onChange,
  onSelect,
}: AddressPickerSheetProps) {
  const t = useResponsiveTokens();
  const { isVeryLargeText } = useAccessibilityScale();
  const suggestionLines = isVeryLargeText ? undefined : 1;
  const [query, setQuery] = useState(value);

  useEffect(() => {
    if (visible) {
      setQuery(value);
    }
  }, [value, visible]);

  const normalizedQuery = query.trim().toLowerCase();
  const canQuery = query.trim().length > 2;
  const shouldIgnoreQuery =
    normalizedQuery.length === 0 || ADDRESS_IGNORE_QUERIES.has(normalizedQuery);
  const addressesQuery = useCompanyAddressSearch(visible && canQuery && !shouldIgnoreQuery ? query : "");
  const sortedSuggestions = useMemo(() => {
    const rows = Array.isArray(addressesQuery.data) ? addressesQuery.data : [];
    return sortAddressSuggestions(rows, normalizedQuery).slice(0, SUGGESTIONS_MAX_VISIBLE);
  }, [addressesQuery.data, normalizedQuery]);
  const groupedSuggestions = useMemo(() => {
    const primary = sortedSuggestions.filter(
      (item) => isAliasSuggestion(item) || looksLikePoi(item) || isGoogleLikeSuggestion(item)
    );
    const secondary = sortedSuggestions.filter((item) => !primary.includes(item));
    return { primary, secondary };
  }, [sortedSuggestions]);

  return (
    <CreateRideKeyboardSheet
      visible={visible}
      title={title}
      subtitle="Recherchez puis choisissez une adresse"
      onClose={() => {
        const next = query.trim();
        if (next !== value.trim()) {
          onChange(next);
        }
        onClose();
      }}
      search={
        <AppInput
          value={query}
          onChangeText={(next) => {
            setQuery(next);
            onChange(next);
          }}
          placeholder="Rechercher une adresse…"
          autoFocus={visible}
          leftSlot={<Ionicons name="search-outline" size={18} color={E.TEXT_SEC} />}
          shellStyle={{ borderRadius: ROW_RADIUS, minHeight: 48, paddingHorizontal: 10 }}
        />
      }
    >
      {!canQuery ? (
        <AppText variant="caption" style={s.hint}>
          Saisissez au moins 3 caractères.
        </AppText>
      ) : null}
      {canQuery && !shouldIgnoreQuery && addressesQuery.isLoading ? (
        <AppText variant="caption" style={s.hint}>
          Recherche…
        </AppText>
      ) : null}
      {canQuery && !shouldIgnoreQuery && !addressesQuery.isLoading && sortedSuggestions.length === 0 ? (
        <AppText variant="caption" style={s.hint}>
          Aucun résultat
        </AppText>
      ) : null}
      {[
        { title: "", rows: groupedSuggestions.primary },
        { title: "Autres résultats", rows: groupedSuggestions.secondary },
      ]
        .filter(() => !addressesQuery.isLoading && sortedSuggestions.length > 0)
        .map((section) => (
          <View key={section.title || "primary"}>
            {section.rows.length > 0 && section.title ? (
              <AppText variant="caption" style={s.sectionTitle}>
                {section.title}
              </AppText>
            ) : null}
            {section.rows.map((address, index) => {
              const { primary, secondary } = splitAddressLabel(address.label);
              const displayPrimary = address.mainText?.trim() || primary;
              const displaySecondary = address.secondaryText?.trim() || secondary;
              const isLast = index === section.rows.length - 1;
              return (
                <Pressable
                  key={`${section.title}-${address.id}-${index}`}
                  accessibilityRole="button"
                  accessibilityLabel={`Suggestion adresse ${address.label}`}
                  onPress={() => {
                    Keyboard.dismiss();
                    onSelect(address);
                  }}
                  style={({ pressed }) => [
                    s.resultRow,
                    { minHeight: Math.max(t.minTouchHeight, 52) },
                    !isLast ? s.resultRowBorder : null,
                    pressed ? s.resultRowPressed : null,
                  ]}
                >
                  <AppText variant="body" numberOfLines={suggestionLines} style={s.resultTitle}>
                    {displayPrimary}
                  </AppText>
                  {displaySecondary ? (
                    <AppText variant="caption" numberOfLines={suggestionLines} style={s.resultSub}>
                      {displaySecondary}
                    </AppText>
                  ) : null}
                </Pressable>
              );
            })}
          </View>
        ))}
    </CreateRideKeyboardSheet>
  );
}

const s = StyleSheet.create({
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
  cardTitle: { flex: 1, color: E.TEXT, fontWeight: "600" },
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
  sectionTitle: {
    color: "rgba(30, 41, 59, 0.72)",
    fontWeight: "700",
    paddingTop: 6,
    paddingBottom: 4,
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
  resultRowPressed: {
    backgroundColor: "rgba(15, 23, 42, 0.06)",
  },
  resultTitle: { color: brandText, fontWeight: "600" },
  resultSub: { color: "rgba(71, 85, 105, 0.85)" },
});
