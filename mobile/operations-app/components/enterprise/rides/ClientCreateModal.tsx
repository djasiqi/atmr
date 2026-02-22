import React, { useState, useEffect, useCallback } from "react";
import {
  Modal,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import { createClient, CreateClientPayload } from "@/services/enterpriseDispatch";
import { AddressSelector } from "./AddressSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { AddressSuggestion } from "@/types/enterpriseDispatch";
import { createShadow } from "@/styles/shadowStyles";
import { getLogger } from "@/utils/logger";

const log = getLogger("ClientCreate");

const BRAND = "#00796B";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "#E2E8F0";
const BG = "#F8FAFC";
const CARD = "#FFFFFF";
const DANGER = "#EF4444";

type AccordionKey = "identity" | "contact" | "residence" | "billing" | "curator" | null;

interface ClientCreateModalProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: (client: { id: string; name: string }) => void;
}

export const ClientCreateModal: React.FC<ClientCreateModalProps> = ({
  visible,
  onClose,
  onSuccess,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openAccordion, setOpenAccordion] = useState<AccordionKey>("identity");

  // --- Identité ---
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [gender, setGender] = useState<"male" | "female" | "">("");
  const [avsNumber, setAvsNumber] = useState("");
  const [birthDate, setBirthDate] = useState<Date | null>(null);
  const [isInstitution, setIsInstitution] = useState(false);
  const [institutionName, setInstitutionName] = useState("");

  // --- Contact & domicile ---
  const [phone, setPhone] = useState("");
  const [domicileAddress, setDomicileAddress] = useState("");
  const [domicileSuggestion, setDomicileSuggestion] = useState<AddressSuggestion | undefined>();
  const [doorCode, setDoorCode] = useState("");
  const [floor, setFloor] = useState("");
  const [accessNotes, setAccessNotes] = useState("");
  const [gpName, setGpName] = useState("");
  const [gpPhone, setGpPhone] = useState("");

  // --- Établissement de résidence ---
  const [residenceFacility, setResidenceFacility] = useState("");

  // --- Facturation ---
  const [contactEmail, setContactEmail] = useState("");
  const [contactPhone, setContactPhone] = useState("");
  const [preferentialRate, setPreferentialRate] = useState("");
  const [showBillingAddress, setShowBillingAddress] = useState(false);
  const [billingAddress, setBillingAddress] = useState("");
  const [billingSuggestion, setBillingSuggestion] = useState<AddressSuggestion | undefined>();
  const [defaultBilledToType, setDefaultBilledToType] = useState("");
  const [defaultBilledToContact, setDefaultBilledToContact] = useState("");

  // --- Curateur ---
  const [curatorName, setCuratorName] = useState("");
  const [curatorEmail, setCuratorEmail] = useState("");
  const [curatorPhone, setCuratorPhone] = useState("");

  useEffect(() => {
    if (!visible) {
      setFirstName(""); setLastName(""); setGender(""); setAvsNumber("");
      setBirthDate(null); setIsInstitution(false); setInstitutionName("");
      setPhone(""); setDomicileAddress(""); setDomicileSuggestion(undefined);
      setDoorCode(""); setFloor(""); setAccessNotes("");
      setGpName(""); setGpPhone(""); setResidenceFacility("");
      setContactEmail(""); setContactPhone(""); setPreferentialRate("");
      setShowBillingAddress(false); setBillingAddress(""); setBillingSuggestion(undefined);
      setDefaultBilledToType(""); setDefaultBilledToContact("");
      setCuratorName(""); setCuratorEmail(""); setCuratorPhone("");
      setError(null); setOpenAccordion("identity");
    }
  }, [visible]);

  const toggleAccordion = useCallback((key: AccordionKey) => {
    setOpenAccordion((prev) => (prev === key ? null : key));
  }, []);

  const summaryName = isInstitution
    ? institutionName || "Client non renseigné"
    : `${firstName} ${lastName}`.trim() || "Client non renseigné";

  const handleSubmit = async () => {
    if (!firstName.trim() || !lastName.trim()) {
      setError("Le prénom et le nom sont requis"); return;
    }
    if (!gender) {
      setError("La civilité (Madame/Monsieur) est obligatoire"); return;
    }
    if (isInstitution && !institutionName.trim()) {
      setError("Le nom de l'institution est requis"); return;
    }
    if (!domicileAddress.trim()) {
      setError("L'adresse de domicile est requise"); return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload: CreateClientPayload = {
        client_type: "PRIVATE",
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        gender: gender as "male" | "female",
        phone: phone.trim() || undefined,
        avs_number: avsNumber.trim() || undefined,
        birth_date: birthDate ? dayjs(birthDate).format("YYYY-MM-DD") : undefined,
        is_institution: isInstitution,
        institution_name: isInstitution ? institutionName.trim() : undefined,
        address: domicileAddress,
        domicile_lat: domicileSuggestion?.lat || null,
        domicile_lon: domicileSuggestion?.lon || null,
        residence_facility: residenceFacility.trim() || undefined,
        contact_email: contactEmail.trim() || undefined,
        contact_phone: contactPhone.trim() || undefined,
        preferential_rate: preferentialRate ? parseFloat(preferentialRate) : undefined,
        billing_address: showBillingAddress && billingAddress.trim() ? billingAddress.trim() : undefined,
        billing_lat: showBillingAddress ? (billingSuggestion?.lat || null) : null,
        billing_lon: showBillingAddress ? (billingSuggestion?.lon || null) : null,
        door_code: doorCode.trim() || undefined,
        floor: floor.trim() || undefined,
        access_notes: accessNotes.trim() || undefined,
        gp_name: gpName.trim() || undefined,
        gp_phone: gpPhone.trim() || undefined,
        default_billed_to_type: defaultBilledToType || undefined,
        default_billed_to_contact: defaultBilledToContact.trim() || undefined,
      };

      log.info("create client payload", { payload, gender });
      const newClient = await createClient(payload);
      onSuccess({ id: newClient.id, name: newClient.name });
      Alert.alert("Client créé", `Le client ${newClient.name} a été créé avec succès.`, [
        { text: "OK", onPress: onClose },
      ]);
    } catch (err: any) {
      log.error("create client failed", { error: err, responseData: err?.response?.data });
      let msg = "Erreur lors de la création du client";
      if (err?.response?.data?.errors) {
        msg = Object.entries(err.response.data.errors)
          .map(([f, m]: [string, any]) => `${f}: ${Array.isArray(m) ? m.join(", ") : m}`)
          .join("\n");
      } else if (err?.response?.data?.error) {
        msg = err.response.data.error;
      } else if (err?.message) {
        msg = err.message;
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const renderAccordion = (
    key: AccordionKey,
    icon: keyof typeof Ionicons.glyphMap,
    title: string,
    children: React.ReactNode,
  ) => {
    const isOpen = openAccordion === key;
    return (
      <View style={s.accordion}>
        <TouchableOpacity
          style={s.accordionHeader}
          onPress={() => toggleAccordion(key)}
          activeOpacity={0.7}
        >
          <View style={s.accordionIconWrap}>
            <Ionicons name={icon} size={16} color={BRAND} />
          </View>
          <Text style={s.accordionTitle}>{title}</Text>
          <Ionicons
            name={isOpen ? "chevron-up" : "chevron-down"}
            size={16}
            color={TEXT_SEC}
          />
        </TouchableOpacity>
        {isOpen && <View style={s.accordionBody}>{children}</View>}
      </View>
    );
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={s.root}
      >
        <Pressable style={s.overlay} onPress={onClose} />
        <View style={s.sheet}>
          <View style={s.handle} />

          {/* Header */}
          <View style={s.header}>
            <View style={s.headerIconWrap}>
              <Ionicons name="person-add-outline" size={20} color={BRAND} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={s.headerTitle}>
                {isInstitution ? "Nouvelle institution" : "Nouveau client"}
              </Text>
              <Text style={s.headerSub}>
                Renseignez l'identité, l'adresse et la facturation.
              </Text>
            </View>
            <TouchableOpacity onPress={onClose} style={s.closeBtn}>
              <Ionicons name="close" size={22} color={TEXT_SEC} />
            </TouchableOpacity>
          </View>

          <ScrollView
            style={s.scroll}
            contentContainerStyle={s.scrollContent}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
            nestedScrollEnabled
          >
            {error && (
              <View style={s.errorBanner}>
                <Ionicons name="alert-circle" size={16} color={DANGER} />
                <Text style={s.errorText}>{error}</Text>
              </View>
            )}

            {/* ====== 1. Identité ====== */}
            {renderAccordion("identity", "person-outline", "Identité du client", (
              <>
                {/* Institution toggle */}
                <TouchableOpacity
                  style={s.checkboxRow}
                  onPress={() => setIsInstitution(!isInstitution)}
                >
                  <View style={[s.checkbox, isInstitution && s.checkboxActive]}>
                    {isInstitution && <Ionicons name="checkmark" size={14} color="#FFF" />}
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={s.checkboxLabel}>Est une institution</Text>
                    <Text style={s.checkboxHint}>Clinique, hôpital, centre médical, etc.</Text>
                  </View>
                </TouchableOpacity>

                {isInstitution && (
                  <View style={s.field}>
                    <Text style={s.label}>Nom de l'institution <Text style={s.req}>*</Text></Text>
                    <View style={s.inputRow}>
                      <Ionicons name="business-outline" size={16} color={TEXT_MUTED} />
                      <TextInput
                        style={s.input}
                        value={institutionName}
                        onChangeText={setInstitutionName}
                        placeholder="Ex: Clinique du Léman"
                        placeholderTextColor={TEXT_MUTED}
                      />
                    </View>
                  </View>
                )}

                {/* Civilité */}
                <View style={s.field}>
                  <Text style={s.label}>Civilité <Text style={s.req}>*</Text></Text>
                  <View style={s.genderRow}>
                    <TouchableOpacity
                      style={[s.genderBtn, gender === "male" && s.genderBtnActive]}
                      onPress={() => setGender("male")}
                    >
                      <Ionicons name="male" size={16} color={gender === "male" ? "#FFF" : BRAND} />
                      <Text style={[s.genderText, gender === "male" && s.genderTextActive]}>Monsieur</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[s.genderBtn, gender === "female" && s.genderBtnActive]}
                      onPress={() => setGender("female")}
                    >
                      <Ionicons name="female" size={16} color={gender === "female" ? "#FFF" : BRAND} />
                      <Text style={[s.genderText, gender === "female" && s.genderTextActive]}>Madame</Text>
                    </TouchableOpacity>
                  </View>
                </View>

                {/* Prénom / Nom */}
                <View style={s.rowTwo}>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Prénom <Text style={s.req}>*</Text></Text>
                    <View style={s.inputRow}>
                      <TextInput style={s.input} value={firstName} onChangeText={setFirstName} placeholder="Prénom" placeholderTextColor={TEXT_MUTED} />
                    </View>
                  </View>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Nom <Text style={s.req}>*</Text></Text>
                    <View style={s.inputRow}>
                      <TextInput style={s.input} value={lastName} onChangeText={setLastName} placeholder="Nom" placeholderTextColor={TEXT_MUTED} />
                    </View>
                  </View>
                </View>

                {/* Date de naissance */}
                {!isInstitution && (
                  <View style={s.field}>
                    <Text style={s.label}>Date de naissance</Text>
                    <TimeDatePicker
                      label="Date de naissance"
                      value={birthDate}
                      onChange={setBirthDate}
                      mode="date"
                      maximumDate={new Date()}
                      minimumDate={new Date(1900, 0, 1)}
                    />
                  </View>
                )}

                {/* Numéro AVS */}
                <View style={s.field}>
                  <Text style={s.label}>Numéro AVS</Text>
                  <View style={s.inputRow}>
                    <Ionicons name="card-outline" size={16} color={TEXT_MUTED} />
                    <TextInput
                      style={s.input}
                      value={avsNumber}
                      onChangeText={setAvsNumber}
                      placeholder="756.XXXX.XXXX.XX"
                      placeholderTextColor={TEXT_MUTED}
                      keyboardType="numbers-and-punctuation"
                    />
                  </View>
                </View>
              </>
            ))}

            {/* ====== 2. Contact et domicile ====== */}
            {renderAccordion("contact", "home-outline", "Contact et domicile", (
              <>
                <View style={s.field}>
                  <Text style={s.label}>Téléphone</Text>
                  <View style={s.inputRow}>
                    <Ionicons name="call-outline" size={16} color={TEXT_MUTED} />
                    <TextInput
                      style={s.input}
                      value={phone}
                      onChangeText={setPhone}
                      placeholder="+41 22 123 45 67"
                      placeholderTextColor={TEXT_MUTED}
                      keyboardType="phone-pad"
                    />
                  </View>
                </View>

                <View style={s.field}>
                  <Text style={s.label}>Adresse de domicile <Text style={s.req}>*</Text></Text>
                  <AddressSelector
                    label=""
                    value={domicileAddress}
                    onChange={(address, suggestion) => {
                      setDomicileAddress(address);
                      setDomicileSuggestion(suggestion);
                    }}
                    icon="location-outline"
                  />
                </View>

                <View style={s.rowTwo}>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Code porte</Text>
                    <View style={s.inputRow}>
                      <TextInput style={s.input} value={doorCode} onChangeText={setDoorCode} placeholder="Ex: 4521" placeholderTextColor={TEXT_MUTED} />
                    </View>
                  </View>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Étage</Text>
                    <View style={s.inputRow}>
                      <TextInput style={s.input} value={floor} onChangeText={setFloor} placeholder="Ex: 2e" placeholderTextColor={TEXT_MUTED} />
                    </View>
                  </View>
                </View>

                <View style={s.field}>
                  <Text style={s.label}>Notes d'accès</Text>
                  <View style={[s.inputRow, { alignItems: "flex-start", minHeight: 60 }]}>
                    <TextInput
                      style={[s.input, { textAlignVertical: "top" }]}
                      value={accessNotes}
                      onChangeText={setAccessNotes}
                      placeholder="Ex: appeler avant, sonnette à gauche..."
                      placeholderTextColor={TEXT_MUTED}
                      multiline
                      numberOfLines={2}
                    />
                  </View>
                </View>

                <View style={s.rowTwo}>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Médecin traitant</Text>
                    <View style={s.inputRow}>
                      <TextInput style={s.input} value={gpName} onChangeText={setGpName} placeholder="Nom du médecin" placeholderTextColor={TEXT_MUTED} />
                    </View>
                  </View>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Tél. médecin</Text>
                    <View style={s.inputRow}>
                      <TextInput style={s.input} value={gpPhone} onChangeText={setGpPhone} placeholder="+41..." placeholderTextColor={TEXT_MUTED} keyboardType="phone-pad" />
                    </View>
                  </View>
                </View>
              </>
            ))}

            {/* ====== 3. Établissement de résidence ====== */}
            {renderAccordion("residence", "business-outline", "Établissement de résidence", (
              <View style={s.field}>
                <Text style={s.label}>Établissement (EMS, foyer, etc.)</Text>
                <View style={s.inputRow}>
                  <Ionicons name="business-outline" size={16} color={TEXT_MUTED} />
                  <TextInput
                    style={s.input}
                    value={residenceFacility}
                    onChangeText={setResidenceFacility}
                    placeholder="Ex: EMS Maison de Vessy..."
                    placeholderTextColor={TEXT_MUTED}
                  />
                </View>
                <Text style={s.hint}>Indiquer si le client réside en EMS, foyer ou autre.</Text>
              </View>
            ))}

            {/* ====== 4. Facturation ====== */}
            {renderAccordion("billing", "receipt-outline", "Facturation", (
              <>
                <View style={s.rowTwo}>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Email facturation</Text>
                    <View style={s.inputRow}>
                      <Ionicons name="mail-outline" size={16} color={TEXT_MUTED} />
                      <TextInput
                        style={s.input}
                        value={contactEmail}
                        onChangeText={setContactEmail}
                        placeholder="facturation@..."
                        placeholderTextColor={TEXT_MUTED}
                        keyboardType="email-address"
                        autoCapitalize="none"
                      />
                    </View>
                  </View>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Tél. contact</Text>
                    <View style={s.inputRow}>
                      <Ionicons name="call-outline" size={16} color={TEXT_MUTED} />
                      <TextInput
                        style={s.input}
                        value={contactPhone}
                        onChangeText={setContactPhone}
                        placeholder="+41..."
                        placeholderTextColor={TEXT_MUTED}
                        keyboardType="phone-pad"
                      />
                    </View>
                  </View>
                </View>

                {/* Adresse de facturation différente */}
                <TouchableOpacity
                  style={s.checkboxRow}
                  onPress={() => setShowBillingAddress(!showBillingAddress)}
                >
                  <View style={[s.checkbox, showBillingAddress && s.checkboxActive]}>
                    {showBillingAddress && <Ionicons name="checkmark" size={14} color="#FFF" />}
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={s.checkboxLabel}>Adresse de facturation différente</Text>
                    <Text style={s.checkboxHint}>Par défaut, la facturation utilise le domicile</Text>
                  </View>
                </TouchableOpacity>

                {showBillingAddress && (
                  <View style={s.field}>
                    <Text style={s.label}>Adresse de facturation</Text>
                    <AddressSelector
                      label=""
                      value={billingAddress}
                      onChange={(address, suggestion) => {
                        setBillingAddress(address);
                        setBillingSuggestion(suggestion);
                      }}
                      icon="receipt-outline"
                    />
                  </View>
                )}

                {!isInstitution && (
                  <View style={s.field}>
                    <Text style={s.label}>Tarif préférentiel (CHF)</Text>
                    <View style={s.inputRow}>
                      <Ionicons name="cash-outline" size={16} color={TEXT_MUTED} />
                      <TextInput
                        style={s.input}
                        value={preferentialRate}
                        onChangeText={setPreferentialRate}
                        placeholder="Ex: 45.00"
                        placeholderTextColor={TEXT_MUTED}
                        keyboardType="decimal-pad"
                      />
                    </View>
                    <Text style={s.hint}>Prix d'un trajet simple. Vide = tarif standard.</Text>
                  </View>
                )}
              </>
            ))}

            {/* ====== 5. Curateur / Tiers payeur ====== */}
            {!isInstitution && renderAccordion("curator", "briefcase-outline", "Curateur / tiers payeur", (
              <>
                <View style={s.field}>
                  <Text style={s.label}>Nom du curateur</Text>
                  <View style={s.inputRow}>
                    <Ionicons name="person-outline" size={16} color={TEXT_MUTED} />
                    <TextInput
                      style={s.input}
                      value={curatorName}
                      onChangeText={setCuratorName}
                      placeholder="Ex: Curateur A"
                      placeholderTextColor={TEXT_MUTED}
                    />
                  </View>
                </View>
                <View style={s.rowTwo}>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Email curateur</Text>
                    <View style={s.inputRow}>
                      <TextInput
                        style={s.input}
                        value={curatorEmail}
                        onChangeText={setCuratorEmail}
                        placeholder="curateur@..."
                        placeholderTextColor={TEXT_MUTED}
                        keyboardType="email-address"
                        autoCapitalize="none"
                      />
                    </View>
                  </View>
                  <View style={[s.field, { flex: 1 }]}>
                    <Text style={s.label}>Tél. curateur</Text>
                    <View style={s.inputRow}>
                      <TextInput
                        style={s.input}
                        value={curatorPhone}
                        onChangeText={setCuratorPhone}
                        placeholder="+41..."
                        placeholderTextColor={TEXT_MUTED}
                        keyboardType="phone-pad"
                      />
                    </View>
                  </View>
                </View>
              </>
            ))}

            {/* ====== Résumé ====== */}
            <View style={s.summaryCard}>
              <Text style={s.summaryTitle}>Résumé</Text>
              <View style={s.summaryRow}>
                <Text style={s.summaryLabel}>Client</Text>
                <Text style={[s.summaryValue, summaryName === "Client non renseigné" && s.summaryEmpty]}>
                  {summaryName}
                </Text>
              </View>
              <View style={s.summaryRow}>
                <Text style={s.summaryLabel}>Adresse</Text>
                <Text style={[s.summaryValue, !domicileAddress && s.summaryEmpty]} numberOfLines={1}>
                  {domicileAddress || "Adresse non renseignée"}
                </Text>
              </View>
              <View style={s.summaryRow}>
                <Text style={s.summaryLabel}>Facturation</Text>
                <Text style={[s.summaryValue, !(showBillingAddress ? billingAddress : domicileAddress) && s.summaryEmpty]} numberOfLines={1}>
                  {(showBillingAddress && billingAddress ? billingAddress : domicileAddress) || "Adresse non renseignée"}
                </Text>
              </View>
              {curatorName.trim() !== "" && (
                <View style={s.summaryRow}>
                  <Text style={s.summaryLabel}>Curateur</Text>
                  <Text style={s.summaryValue}>{curatorName}</Text>
                </View>
              )}
            </View>

            <View style={{ height: 20 }} />
          </ScrollView>

          {/* Footer */}
          <View style={s.footer}>
            <Text style={s.footerSummary} numberOfLines={1}>{summaryName}</Text>
            <View style={s.footerActions}>
              <TouchableOpacity style={s.footerCancel} onPress={onClose} disabled={loading}>
                <Text style={s.footerCancelText}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[s.footerSubmit, loading && s.footerSubmitDisabled]}
                onPress={handleSubmit}
                disabled={loading}
                activeOpacity={0.85}
              >
                {loading ? (
                  <ActivityIndicator color="#FFF" size="small" />
                ) : (
                  <>
                    <Ionicons name="checkmark" size={16} color="#FFF" />
                    <Text style={s.footerSubmitText}>Créer le client</Text>
                  </>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
};

const sheetShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: -4 },
  shadowOpacity: 0.12,
  shadowRadius: 24,
  elevation: 12,
});

const s = StyleSheet.create({
  root: {
    flex: 1,
    justifyContent: "flex-end" as const,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.35)",
  },
  sheet: {
    backgroundColor: CARD,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    height: "92%",
    overflow: "hidden",
    ...sheetShadow,
  },
  handle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#D1D5DB",
    alignSelf: "center",
    marginTop: 10,
    marginBottom: 6,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 20,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  headerIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: TEXT,
  },
  headerSub: {
    fontSize: 12,
    color: TEXT_SEC,
    marginTop: 2,
  },
  closeBtn: {
    padding: 4,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 20,
  },
  errorBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "rgba(239,68,68,0.08)",
    padding: 12,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.15)",
  },
  errorText: {
    flex: 1,
    color: DANGER,
    fontSize: 13,
  },

  // --- Accordion ---
  accordion: {
    backgroundColor: CARD,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: BORDER,
    marginBottom: 10,
    overflow: "hidden",
  },
  accordionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 14,
    paddingVertical: 13,
  },
  accordionIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  accordionTitle: {
    flex: 1,
    fontSize: 14,
    fontWeight: "600",
    color: TEXT,
  },
  accordionBody: {
    paddingHorizontal: 14,
    paddingBottom: 14,
    borderTopWidth: 1,
    borderTopColor: BORDER,
    paddingTop: 12,
  },

  // --- Fields ---
  field: {
    marginBottom: 12,
  },
  label: {
    fontSize: 13,
    fontWeight: "600",
    color: TEXT,
    marginBottom: 6,
  },
  req: {
    color: DANGER,
    fontWeight: "700",
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: BG,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: BORDER,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 8,
  },
  input: {
    flex: 1,
    color: TEXT,
    fontSize: 14,
    padding: 0,
  },
  hint: {
    fontSize: 11,
    color: TEXT_MUTED,
    marginTop: 4,
    fontStyle: "italic",
  },
  rowTwo: {
    flexDirection: "row",
    gap: 10,
  },

  // --- Gender buttons ---
  genderRow: {
    flexDirection: "row",
    gap: 10,
  },
  genderBtn: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    paddingVertical: 11,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: BORDER,
    backgroundColor: BG,
  },
  genderBtnActive: {
    backgroundColor: BRAND,
    borderColor: BRAND,
  },
  genderText: {
    fontSize: 14,
    fontWeight: "600",
    color: TEXT,
  },
  genderTextActive: {
    color: "#FFFFFF",
  },

  // --- Checkbox ---
  checkboxRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 8,
    marginBottom: 8,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: BORDER,
    backgroundColor: BG,
    alignItems: "center",
    justifyContent: "center",
  },
  checkboxActive: {
    backgroundColor: BRAND,
    borderColor: BRAND,
  },
  checkboxLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: TEXT,
  },
  checkboxHint: {
    fontSize: 11,
    color: TEXT_MUTED,
    marginTop: 1,
  },

  // --- Summary card ---
  summaryCard: {
    backgroundColor: BG,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: BORDER,
    padding: 14,
    marginTop: 4,
  },
  summaryTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: TEXT,
    marginBottom: 10,
  },
  summaryRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 5,
  },
  summaryLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: TEXT_SEC,
  },
  summaryValue: {
    fontSize: 12,
    color: TEXT,
    maxWidth: "60%",
    textAlign: "right",
  },
  summaryEmpty: {
    color: TEXT_MUTED,
    fontStyle: "italic",
  },

  // --- Footer ---
  footer: {
    borderTopWidth: 1,
    borderTopColor: BORDER,
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: Platform.OS === "ios" ? 30 : 16,
  },
  footerSummary: {
    fontSize: 12,
    color: TEXT_MUTED,
    marginBottom: 8,
  },
  footerActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  footerCancel: {
    paddingHorizontal: 16,
    paddingVertical: 11,
  },
  footerCancelText: {
    fontSize: 14,
    fontWeight: "600",
    color: TEXT_SEC,
  },
  footerSubmit: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: BRAND,
    paddingVertical: 13,
    borderRadius: 12,
  },
  footerSubmitDisabled: {
    opacity: 0.5,
  },
  footerSubmitText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#FFFFFF",
  },
});
