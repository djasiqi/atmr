import { useState } from "react";
import { Pressable, TextInput, View } from "react-native";
import { AppButton, Modal, brandPrimary, brandText } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { useClientCreate, useCompanyBillingPartiesQuery } from "../../useRideForms";
import { useSession } from "../../../../core/sessionProvider";

type ClientCreateModalProps = {
  visible: boolean;
  onClose: () => void;
  onCreated?: () => void;
};

export function ClientCreateModal({ visible, onClose, onCreated }: ClientCreateModalProps) {
  const { activeContext } = useSession();
  const createClient = useClientCreate();
  const billingParties = useCompanyBillingPartiesQuery();
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [gender, setGender] = useState<"male" | "female">("female");
  const [phone, setPhone] = useState("");
  const [stayStartDate, setStayStartDate] = useState("");
  const [selectedBillingPartyId, setSelectedBillingPartyId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!firstName.trim() || !lastName.trim()) {
      setError("Prenom et nom requis.");
      return;
    }
    try {
      await createClient.mutateAsync({
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        gender,
        phone: phone.trim() || null,
        stay_start_date: stayStartDate.trim() || null,
        stay_company_id:
          activeContext?.context_type === "company"
            ? Number(activeContext.context_id.split(":")[1] ?? NaN)
            : null,
        billing_party_id: selectedBillingPartyId,
      });
      setFirstName("");
      setLastName("");
      setGender("female");
      setPhone("");
      setStayStartDate("");
      setSelectedBillingPartyId(null);
      setError(null);
      onCreated?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Creation client impossible.");
    }
  };

  return (
    <Modal visible={visible} title="Creer un client" onClose={onClose}>
      <View style={{ flexDirection: "row", gap: 8 }}>
        <TextInput
          value={firstName}
          onChangeText={setFirstName}
          placeholder="Prenom"
          style={{ flex: 1, borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10 }}
        />
        <TextInput
          value={lastName}
          onChangeText={setLastName}
          placeholder="Nom"
          style={{ flex: 1, borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10 }}
        />
      </View>
      <View style={{ flexDirection: "row", gap: 8, marginTop: 8 }}>
        <AppButton
          title={gender === "female" ? "Civilite: Femme" : "Femme"}
          onPress={() => setGender("female")}
          variant={gender === "female" ? "primary" : "secondary"}
        />
        <AppButton
          title={gender === "male" ? "Civilite: Homme" : "Homme"}
          onPress={() => setGender("male")}
          variant={gender === "male" ? "primary" : "secondary"}
        />
      </View>
      <TextInput
        value={stayStartDate}
        onChangeText={setStayStartDate}
        placeholder="Date debut stay (YYYY-MM-DD) optionnel"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10, marginTop: 8 }}
      />
      <TextInput
        value={phone}
        onChangeText={setPhone}
        placeholder="Telephone"
        keyboardType="phone-pad"
        style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10, marginTop: 8 }}
      />
      <View style={{ marginTop: 8, gap: 6 }}>
        <AppText variant="label">Billing party par defaut (optionnel)</AppText>
        {billingParties.data?.map((party) => (
          <Pressable
            key={party.id}
            onPress={() => setSelectedBillingPartyId(party.id)}
            style={{
              borderWidth: 1,
              borderColor: selectedBillingPartyId === party.id ? "#0A8F7A" : "#ddd",
              borderRadius: 8,
              padding: 8,
            }}
          >
            <AppText
              variant="body"
              style={{ color: selectedBillingPartyId === party.id ? brandPrimary : brandText }}
            >
              {party.display_name} ({party.type})
            </AppText>
          </Pressable>
        ))}
      </View>
      <AppButton
        title={createClient.isPending ? "Creation..." : "Creer"}
        variant="primary"
        onPress={() => void submit()}
      />
      {error ? <AppText variant="error">{error}</AppText> : null}
    </Modal>
  );
}
