import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { Pressable, ScrollView, View, type TextStyle, type ViewStyle } from "react-native";
import { useAccessibilityScale, useResponsiveTokens } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { RideAddressOption, useCompanyAddressSearch } from "../../useRideForms";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import {
  suggestionDropdownAnchorStyle,
  suggestionDropdownHitAreaStyle,
  suggestionDropdownPanelStyle,
  suggestionFieldOpenStyle,
} from "./suggestionOverlayStyles";

const UI_BORDER_SOFT = "rgba(0, 121, 107, 0.12)";
const ROW_RADIUS = 12;
const SUGGESTIONS_MAX_VISIBLE = 5;
const SELECTION_REOPEN_GUARD_MS = 450;
const IGNORE_QUERIES = new Set(["non spécifié", "non specifie", "n/a", "na"]);

function splitAddressLabel(label: string) {
  const parts = label
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length <= 1) {
    return { primary: label.trim(), secondary: "" };
  }
  const [primary, ...rest] = parts;
  return { primary, secondary: rest.join(", ") };
}

function isAliasSuggestion(item: RideAddressOption): boolean {
  return item.source === "alias";
}

function isGoogleLikeSuggestion(item: RideAddressOption): boolean {
  return item.source === "google_places" || item.source === "google";
}

function looksLikePoi(item: RideAddressOption): boolean {
  return (
    isGoogleLikeSuggestion(item) &&
    Array.isArray(item.types) &&
    item.types.some((t) => t !== "geocode" && t !== "route" && t !== "street_address")
  );
}

type AddressSelectorProps = {
  label: string;
  value: string;
  onChange: (value: string) => void;
  onSelectAddress?: (address: RideAddressOption) => void;
  placeholder?: string;
  leftSlot?: ReactNode;
  containerStyle?: ViewStyle;
  shellStyle?: ViewStyle;
  inputStyle?: TextStyle;
  /** Affiche un astérisque rouge tant que le champ est vide. */
  required?: boolean;
  /** Notifie le parent lorsque la liste de suggestions s’ouvre ou se ferme (empilement z-index). */
  onSuggestionsVisibilityChange?: (visible: boolean) => void;
};

export function AddressSelector({
  label,
  value,
  onChange,
  onSelectAddress,
  placeholder = "Adresse",
  leftSlot,
  containerStyle,
  shellStyle,
  inputStyle,
  required = false,
  onSuggestionsVisibilityChange,
}: AddressSelectorProps) {
  const t = useResponsiveTokens();
  const { isVeryLargeText } = useAccessibilityScale();
  const suggestionLines = isVeryLargeText ? undefined : 1;
  const [query, setQuery] = useState(value);
  const [isSuggestionsOpen, setIsSuggestionsOpen] = useState(false);
  const [lastSelectedNormalized, setLastSelectedNormalized] = useState("");
  const [justSelected, setJustSelected] = useState(false);
  const addressesQuery = useCompanyAddressSearch(query);
  const normalizedQuery = query.trim().toLowerCase();
  const canQuery = query.trim().length > 2;
  const shouldIgnoreQuery = useMemo(
    () => normalizedQuery.length === 0 || IGNORE_QUERIES.has(normalizedQuery),
    [normalizedQuery]
  );
  const shouldShowDropdown = useMemo(
    () =>
      isSuggestionsOpen &&
      canQuery &&
      !shouldIgnoreQuery &&
      normalizedQuery !== lastSelectedNormalized,
    [canQuery, isSuggestionsOpen, lastSelectedNormalized, normalizedQuery, shouldIgnoreQuery]
  );
  const sortedSuggestions = useMemo(() => {
    const rows = Array.isArray(addressesQuery.data) ? [...addressesQuery.data] : [];
    const score = (item: RideAddressOption): number => {
      if (isAliasSuggestion(item)) return 500;
      const label = (item.label || "").toLowerCase();
      const mainText = (item.mainText || "").toLowerCase();
      const startsWithQuery = label.startsWith(normalizedQuery) || mainText.startsWith(normalizedQuery);
      if (startsWithQuery && looksLikePoi(item)) return 400;
      if (startsWithQuery && isGoogleLikeSuggestion(item)) return 350;
      if (looksLikePoi(item)) return 300;
      if (isGoogleLikeSuggestion(item)) return 250;
      if (startsWithQuery) return 200;
      return 100;
    };
    rows.sort((a, b) => score(b) - score(a));
    return rows;
  }, [addressesQuery.data, normalizedQuery]);
  const groupedSuggestions = useMemo(() => {
    const limited = sortedSuggestions.slice(0, SUGGESTIONS_MAX_VISIBLE);
    const primary = limited.filter(
      (item) => isAliasSuggestion(item) || looksLikePoi(item) || isGoogleLikeSuggestion(item)
    );
    const secondary = limited.filter((item) => !primary.includes(item));
    return { primary, secondary };
  }, [sortedSuggestions]);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  useEffect(() => {
    if (!justSelected) return undefined;
    const timer = setTimeout(() => setJustSelected(false), SELECTION_REOPEN_GUARD_MS);
    return () => clearTimeout(timer);
  }, [justSelected]);

  useEffect(() => {
    onSuggestionsVisibilityChange?.(shouldShowDropdown);
    return () => {
      if (shouldShowDropdown) {
        onSuggestionsVisibilityChange?.(false);
      }
    };
  }, [onSuggestionsVisibilityChange, shouldShowDropdown]);

  const openStackStyle = shouldShowDropdown ? suggestionFieldOpenStyle : null;

  return (
    <View style={[{ gap: t.fieldGap }, openStackStyle, containerStyle]}>
      {label ? <AppText variant="label">{label}</AppText> : null}
      <View style={[{ position: "relative" }, openStackStyle]}>
        <AppInput
          value={query}
          onChangeText={(next) => {
            setQuery(next);
            onChange(next);
            if (justSelected) {
              setJustSelected(false);
            }
            const normalizedNext = next.trim().toLowerCase();
            if (normalizedNext.length <= 2) {
              setIsSuggestionsOpen(false);
              return;
            }
            if (normalizedNext !== lastSelectedNormalized) {
              setLastSelectedNormalized("");
            }
            setIsSuggestionsOpen(true);
          }}
          onFocus={() => {
            if (justSelected) return;
            if (canQuery && normalizedQuery !== lastSelectedNormalized) {
              setIsSuggestionsOpen(true);
            }
          }}
          placeholder={placeholder}
          leftSlot={leftSlot}
          rightSlot={
            required ? (
              <View
                style={{ width: 16, alignItems: "center", justifyContent: "center" }}
                accessibilityElementsHidden={query.trim().length !== 0}
              >
                {query.trim().length === 0 ? (
                  <AppText
                    variant="label"
                    accessibilityLabel="Champ obligatoire"
                    style={{ color: "#DC2626", fontWeight: "700", fontSize: 16, lineHeight: 18 }}
                  >
                    *
                  </AppText>
                ) : null}
              </View>
            ) : null
          }
          shellStyle={{
            borderRadius: ROW_RADIUS,
            minHeight: Math.max(t.fieldShellMinHeight, 34),
            ...shellStyle,
          }}
          style={inputStyle}
        />
        {shouldShowDropdown ? (
          <View style={suggestionDropdownAnchorStyle}>
          <View
            style={{
              borderWidth: 1,
              borderColor: UI_BORDER_SOFT,
              borderRadius: ROW_RADIUS,
              ...suggestionDropdownPanelStyle,
              ...suggestionDropdownHitAreaStyle,
            }}
          >
            <ScrollView
              nestedScrollEnabled
              keyboardShouldPersistTaps="always"
              style={{ maxHeight: SUGGESTIONS_MAX_VISIBLE * 46 }}
            >
              {addressesQuery.isLoading ? (
                <AppText
                  variant="caption"
                  style={{ color: "rgba(71, 85, 105, 0.92)", paddingHorizontal: 10, paddingVertical: 8 }}
                >
                  Recherche…
                </AppText>
              ) : null}
              {!addressesQuery.isLoading && sortedSuggestions.length === 0 ? (
                <AppText
                  variant="caption"
                  style={{ color: "rgba(71, 85, 105, 0.92)", paddingHorizontal: 10, paddingVertical: 8 }}
                >
                  Aucun résultat
                </AppText>
              ) : null}
              {[
                { title: "", rows: groupedSuggestions.primary },
                { title: "Autres résultats", rows: groupedSuggestions.secondary },
              ]
                .filter(() => !addressesQuery.isLoading && sortedSuggestions.length > 0)
                .map((section) => (
                <View key={section.title}>
                  {section.rows.length > 0 && section.title ? (
                    <AppText
                      variant="caption"
                      scaleRole="chrome"
                      style={{
                        color: "rgba(30, 41, 59, 0.72)",
                        fontWeight: "700",
                        paddingHorizontal: 9,
                        paddingTop: 5,
                        paddingBottom: 3,
                        backgroundColor: "rgba(248, 250, 252, 0.65)",
                      }}
                    >
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
                        // Sélection dès pressIn : évite la course blur/focus web qui annule onPress.
                        onPressIn={(event) => {
                          event.preventDefault?.();
                          setQuery(address.label);
                          onChange(address.label);
                          onSelectAddress?.(address);
                          setLastSelectedNormalized(address.label.trim().toLowerCase());
                          setIsSuggestionsOpen(false);
                          setJustSelected(true);
                        }}
                        style={({ hovered, pressed }) => ({
                          minHeight: 36,
                          justifyContent: "center",
                          paddingVertical: 5,
                          paddingHorizontal: 9,
                          backgroundColor: pressed
                            ? "rgba(15, 23, 42, 0.06)"
                            : hovered
                              ? "rgba(15, 23, 42, 0.03)"
                              : "#FFFFFF",
                          borderBottomWidth: isLast ? 0 : 1,
                          borderBottomColor: "rgba(148, 163, 184, 0.22)",
                          cursor: "pointer",
                          minWidth: 0,
                        })}
                      >
                        <AppText
                          variant="body"
                          numberOfLines={suggestionLines}
                          style={{
                            color: "rgba(15, 23, 42, 0.9)",
                            fontSize: FONT_SIZE.px14,
                            lineHeight: 17,
                            fontWeight: "600",
                            flexShrink: 1,
                            minWidth: 0,
                          }}
                        >
                          {displayPrimary}
                        </AppText>
                        {displaySecondary ? (
                          <AppText
                            variant="caption"
                            numberOfLines={suggestionLines}
                            style={{
                              color: "rgba(71, 85, 105, 0.82)",
                              marginTop: 1,
                              fontSize: FONT_SIZE.px12,
                              lineHeight: 14,
                              flexShrink: 1,
                              minWidth: 0,
                            }}
                          >
                            {displaySecondary}
                          </AppText>
                        ) : null}
                      </Pressable>
                    );
                  })}
                </View>
              ))}
            </ScrollView>
          </View>
          </View>
        ) : null}
      </View>
    </View>
  );
}
