import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { Pressable, ScrollView, View, type TextStyle, type ViewStyle } from "react-native";
import { useResponsiveTokens } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { RideAddressOption, useCompanyAddressSearch } from "../../useRideForms";

const UI_BORDER_SOFT = "rgba(0, 121, 107, 0.12)";
const ROW_RADIUS = 12;
const SUGGESTIONS_MAX_VISIBLE = 5;

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
}: AddressSelectorProps) {
  const t = useResponsiveTokens();
  const [query, setQuery] = useState(value);
  const [isSuggestionsOpen, setIsSuggestionsOpen] = useState(false);
  const [lastSelectedNormalized, setLastSelectedNormalized] = useState("");
  const addressesQuery = useCompanyAddressSearch(query);
  const normalizedQuery = query.trim().toLowerCase();
  const canQuery = query.trim().length > 2;
  const shouldShowSuggestions = useMemo(
    () =>
      isSuggestionsOpen &&
      canQuery &&
      normalizedQuery.length > 0 &&
      normalizedQuery !== lastSelectedNormalized &&
      (addressesQuery.data?.length ?? 0) > 0,
    [addressesQuery.data?.length, canQuery, isSuggestionsOpen, lastSelectedNormalized, normalizedQuery]
  );

  useEffect(() => {
    setQuery(value);
  }, [value]);

  return (
    <View style={[{ gap: t.fieldGap }, containerStyle]}>
      {label ? <AppText variant="label">{label}</AppText> : null}
      <AppInput
        value={query}
        onChangeText={(next) => {
          setQuery(next);
          onChange(next);
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
          if (canQuery && normalizedQuery !== lastSelectedNormalized) {
            setIsSuggestionsOpen(true);
          }
        }}
        placeholder={placeholder}
        leftSlot={leftSlot}
        shellStyle={[
          { borderRadius: ROW_RADIUS, minHeight: Math.max(t.fieldShellMinHeight, 48) },
          shellStyle,
        ]}
        style={inputStyle}
      />
      {shouldShowSuggestions ? (
        <View style={{ gap: 8 }}>
          <AppText variant="caption" style={{ color: "rgba(71, 85, 105, 0.9)" }}>
            Suggestions d’adresse
          </AppText>
          <View
            style={{
              borderWidth: 1,
              borderColor: UI_BORDER_SOFT,
              borderRadius: ROW_RADIUS,
              backgroundColor: "#FFFFFF",
              shadowColor: "rgba(15, 23, 42, 0.15)",
              shadowOpacity: 0.1,
              shadowRadius: 10,
              shadowOffset: { width: 0, height: 4 },
              elevation: 2,
              overflow: "hidden",
            }}
          >
            <ScrollView
              nestedScrollEnabled
              keyboardShouldPersistTaps="handled"
              style={{ maxHeight: SUGGESTIONS_MAX_VISIBLE * 56 }}
            >
              {addressesQuery.data?.slice(0, SUGGESTIONS_MAX_VISIBLE).map((address, index, list) => {
                const { primary, secondary } = splitAddressLabel(address.label);
                const isLast = index === list.length - 1;
                return (
                  <Pressable
                    key={address.id}
                    accessibilityRole="button"
                    accessibilityLabel={`Suggestion adresse ${address.label}`}
                    onPress={() => {
                      setQuery(address.label);
                      onChange(address.label);
                      onSelectAddress?.(address);
                      setLastSelectedNormalized(address.label.trim().toLowerCase());
                      setIsSuggestionsOpen(false);
                    }}
                    style={({ hovered, focused, pressed }) => ({
                      minHeight: t.minTouchHeight,
                      justifyContent: "center",
                      paddingVertical: 8,
                      paddingHorizontal: 12,
                      backgroundColor: pressed
                        ? "rgba(15, 23, 42, 0.06)"
                        : hovered || focused
                          ? "rgba(15, 23, 42, 0.03)"
                          : "#FFFFFF",
                      borderBottomWidth: isLast ? 0 : 1,
                      borderBottomColor: "rgba(148, 163, 184, 0.28)",
                    })}
                  >
                    <AppText
                      variant="body"
                      numberOfLines={1}
                      style={{ color: "rgba(15, 23, 42, 0.98)", fontWeight: "600" }}
                    >
                      {primary}
                    </AppText>
                    {secondary ? (
                      <AppText
                        variant="caption"
                        numberOfLines={1}
                        style={{ color: "rgba(71, 85, 105, 0.92)", marginTop: 2 }}
                      >
                        {secondary}
                      </AppText>
                    ) : null}
                  </Pressable>
                );
              })}
            </ScrollView>
          </View>
        </View>
      ) : null}
      {addressesQuery.isLoading ? <AppText variant="caption">Suggestions…</AppText> : null}
    </View>
  );
}
