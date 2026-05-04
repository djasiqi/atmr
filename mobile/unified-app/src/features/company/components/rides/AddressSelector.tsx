import { useEffect, useState } from "react";
import { Pressable, TextInput, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { RideAddressOption, useCompanyAddressSearch } from "../../useRideForms";

type AddressSelectorProps = {
  label: string;
  value: string;
  onChange: (value: string) => void;
  onSelectAddress?: (address: RideAddressOption) => void;
  placeholder?: string;
};

export function AddressSelector({
  label,
  value,
  onChange,
  onSelectAddress,
  placeholder = "Adresse",
}: AddressSelectorProps) {
  const [query, setQuery] = useState(value);
  const addressesQuery = useCompanyAddressSearch(query);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  return (
    <View style={{ gap: 6 }}>
      <AppText variant="label">{label}</AppText>
      <TextInput
        value={query}
        onChangeText={(next) => {
          setQuery(next);
          onChange(next);
        }}
        placeholder={placeholder}
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10 }}
      />
      {addressesQuery.data?.slice(0, 5).map((address) => (
        <Pressable
          key={address.id}
          onPress={() => {
            setQuery(address.label);
            onChange(address.label);
            onSelectAddress?.(address);
          }}
          style={{ borderWidth: 1, borderColor: "#eee", borderRadius: 8, padding: 8 }}
        >
          <AppText variant="body">{address.label}</AppText>
        </Pressable>
      ))}
      {addressesQuery.isLoading ? <AppText variant="bodyMuted">Suggestions adresses...</AppText> : null}
    </View>
  );
}
