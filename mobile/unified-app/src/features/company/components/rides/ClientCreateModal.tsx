import { useState } from "react";
import { Pressable, View } from "react-native";
import { AppButton, Modal, brandPrimary, brandText, useResponsiveTokens } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { useClientCreate, useCompanyBillingPartiesQuery } from "../../useRideForms";
import { useSession } from "../../../../core/sessionProvider";

type ClientCreateModalProps = {
  visible: boolean;
  onClose: () => void;
  onCreated?: () => void;
};

const CTA_STYLE = { minHeight: 54, borderRadius: 14 } as const;
const SECONDARY_CTA_BORDER = { borderColor: "rgba(10, 143, 122, 0.45)" } as const;
const ROW_RADIUS = 14;
const UI_BORDER_SOFT = "rgba(145, 165, 157, 0.38)";

export function ClientCreateModal({ visible, onClose, onCreated }: ClientCreateModalProps) {
  const t = useResponsiveTokens();
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
      setError(e instanceof Error ? e.message : "Création du client impossible.");
    }
  };

  return (
    <Modal visible={visible} title="Créer un client" onClose={onClose}>
      <View style={{ flexDirection: "row", gap: t.spacingSm }}>
        <AppInput
          value={firstName}
          onChangeText={setFirstName}
          placeholder="Prénom"
          containerStyle={{ flex: 1, minWidth: 0 }}
          shellStyle={{ borderRadius: ROW_RADIUS }}
        />
        <AppInput
          value={lastName}
          onChangeText={setLastName}
          placeholder="Nom"
          containerStyle={{ flex: 1, minWidth: 0 }}
          shellStyle={{ borderRadius: ROW_RADIUS }}
        />
      </View>
      <View style={{ flexDirection: "row", gap: t.spacingSm, marginTop: t.spacingSm }}>
        <AppButton
          title={gender === "female" ? "Civilité : Femme" : "Femme"}
          onPress={() => setGender("female")}
          variant={gender === "female" ? "primary" : "secondary"}
          style={
            gender === "female"
              ? CTA_STYLE
              : { ...CTA_STYLE, ...SECONDARY_CTA_BORDER }
          }
        />
        <AppButton
          title={gender === "male" ? "Civilité : Homme" : "Homme"}
          onPress={() => setGender("male")}
          variant={gender === "male" ? "primary" : "secondary"}
          style={
            gender === "male"
              ? CTA_STYLE
              : { ...CTA_STYLE, ...SECONDARY_CTA_BORDER }
          }
        />
      </View>
      <AppInput
        value={stayStartDate}
        onChangeText={setStayStartDate}
        placeholder="Date début séjour (YYYY-MM-DD) optionnel"
        containerStyle={{ marginTop: t.spacingSm }}
        shellStyle={{ borderRadius: ROW_RADIUS }}
      />
      <AppInput
        value={phone}
        onChangeText={setPhone}
        placeholder="Téléphone"
        keyboardType="phone-pad"
        containerStyle={{ marginTop: t.spacingSm }}
        shellStyle={{ borderRadius: ROW_RADIUS }}
      />
      <View style={{ marginTop: t.spacingSm, gap: t.fieldGap }}>
        <AppText variant="label">Billing party par défaut (optionnel)</AppText>
        {billingParties.data?.map((party) => (
          <Pressable
            key={party.id}
            onPress={() => setSelectedBillingPartyId(party.id)}
            style={{
              borderWidth: 1,
              borderColor: selectedBillingPartyId === party.id ? brandPrimary : UI_BORDER_SOFT,
              borderRadius: ROW_RADIUS,
              paddingVertical: t.spacingSm,
              paddingHorizontal: t.spacingSm + 2,
              minHeight: t.minTouchHeight,
              justifyContent: "center",
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
        title={createClient.isPending ? "Création…" : "Créer"}
        variant="primary"
        onPress={() => void submit()}
        style={CTA_STYLE}
      />
      {error ? <AppText variant="error">{error}</AppText> : null}
    </Modal>
  );
}
