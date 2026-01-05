import React, { useState, useEffect } from "react";
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
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { createClient, CreateClientPayload } from "@/services/enterpriseDispatch";
import { AddressSelector } from "./AddressSelector";
import { AddressSuggestion } from "@/types/enterpriseDispatch";

const palette = {
  modalOverlay: "rgba(21,54,43,0.75)",
  modalBackground: "#FFFFFF",
  modalBorder: "rgba(15,54,43,0.12)",
  modalTitle: "#15362B",
  modalText: "#5F7369",
  modalButton: "#0A7F59",
  modalButtonText: "#FFFFFF",
  modalCancelText: "#5F7369",
  error: "#EF4444",
};

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

  // Informations personnelles
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [phone, setPhone] = useState("");
  const [isInstitution, setIsInstitution] = useState(false);
  const [institutionName, setInstitutionName] = useState("");

  // Adresse de domicile
  const [domicileAddress, setDomicileAddress] = useState("");
  const [domicileSuggestion, setDomicileSuggestion] = useState<AddressSuggestion | undefined>();

  useEffect(() => {
    if (!visible) {
      // Reset form when modal closes
      setFirstName("");
      setLastName("");
      setPhone("");
      setIsInstitution(false);
      setInstitutionName("");
      setDomicileAddress("");
      setDomicileSuggestion(undefined);
      setError(null);
    }
  }, [visible]);

  const handleSubmit = async () => {
    // Validation
    if (!firstName.trim() || !lastName.trim()) {
      setError("Le prénom et le nom sont requis");
      return;
    }

    if (isInstitution && !institutionName.trim()) {
      setError("Le nom de l'institution est requis");
      return;
    }

    if (!domicileAddress.trim()) {
      setError("L'adresse de domicile est requise");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Construire l'adresse complète
      let fullAddress = domicileAddress;
      if (domicileSuggestion) {
        // Utiliser l'adresse complète de la suggestion
        fullAddress = domicileSuggestion.address || domicileAddress;
      }

      // Parser l'adresse pour extraire code postal et ville
      let domicileZip = "";
      let domicileCity = "";
      const addressMatch = fullAddress.match(/,\s*(\d{4})\s+([^,]+?)(?:\s*,\s*Suisse)?$/);
      if (addressMatch) {
        domicileZip = addressMatch[1];
        domicileCity = addressMatch[2].trim();
      }

      const payload: CreateClientPayload = {
        client_type: "PRIVATE",
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        phone: phone.trim() || undefined,
        is_institution: isInstitution,
        institution_name: isInstitution ? institutionName.trim() : undefined,
        domicile_address: fullAddress,
        domicile_zip: domicileZip || undefined,
        domicile_city: domicileCity || undefined,
        domicile_lat: domicileSuggestion?.lat || null,
        domicile_lon: domicileSuggestion?.lon || null,
        // Pour l'instant, utiliser la même adresse pour la facturation
        billing_address: fullAddress,
        billing_lat: domicileSuggestion?.lat || null,
        billing_lon: domicileSuggestion?.lon || null,
      };

      const newClient = await createClient(payload);
      
      // Appeler onSuccess immédiatement avec toutes les données du client
      onSuccess({
        id: newClient.id,
        name: newClient.name,
      });
      
      Alert.alert(
        "Client créé",
        `Le client ${newClient.name} a été créé avec succès.`,
        [
          {
            text: "OK",
            onPress: () => {
              onClose();
            },
          },
        ]
      );
    } catch (err: any) {
      const errorMessage =
        err?.response?.data?.error ||
        err?.response?.data?.message ||
        err?.message ||
        "Erreur lors de la création du client";
      setError(errorMessage);
      console.error("[ClientCreateModal] Erreur:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalCard}>
          <View style={styles.modalHeader}>
            <View>
              <Text style={styles.modalTitle}>
                {isInstitution ? "Nouvelle institution" : "Nouveau client"}
              </Text>
              <Text style={styles.modalSubtitle}>
                Remplissez les informations requises
              </Text>
            </View>
            <TouchableOpacity onPress={onClose} style={styles.closeButton}>
              <Ionicons name="close" size={24} color={palette.modalText} />
            </TouchableOpacity>
          </View>

          <ScrollView
            style={styles.modalScroll}
            contentContainerStyle={styles.modalContent}
            showsVerticalScrollIndicator={false}
          >
            {error && (
              <View style={styles.errorContainer}>
                <Ionicons name="alert-circle" size={18} color={palette.error} />
                <Text style={styles.errorText}>{error}</Text>
              </View>
            )}

            {/* Type de client */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Type de client</Text>
              <TouchableOpacity
                style={styles.checkboxRow}
                onPress={() => setIsInstitution(!isInstitution)}
              >
                <View style={styles.checkbox}>
                  {isInstitution && (
                    <Ionicons name="checkmark" size={16} color={palette.modalButton} />
                  )}
                </View>
                <Text style={styles.checkboxLabel}>
                  Est une institution (clinique, hôpital, etc.)
                </Text>
              </TouchableOpacity>

              {isInstitution && (
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>Nom de l'institution *</Text>
                  <View style={styles.textInputContainer}>
                    <Ionicons name="business-outline" size={18} color={palette.modalButton} />
                    <TextInput
                      style={styles.textInput}
                      value={institutionName}
                      onChangeText={setInstitutionName}
                      placeholder="Ex: Clinique du Léman"
                      placeholderTextColor={palette.modalText}
                    />
                  </View>
                </View>
              )}
            </View>

            {/* Informations personnelles */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>
                {isInstitution
                  ? "Contact principal"
                  : "Informations personnelles"}
              </Text>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Prénom *</Text>
                <View style={styles.textInputContainer}>
                  <Ionicons name="person-outline" size={18} color={palette.modalButton} />
                  <TextInput
                    style={styles.textInput}
                    value={firstName}
                    onChangeText={setFirstName}
                    placeholder="Prénom"
                    placeholderTextColor={palette.modalText}
                  />
                </View>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Nom *</Text>
                <View style={styles.textInputContainer}>
                  <Ionicons name="person-outline" size={18} color={palette.modalButton} />
                  <TextInput
                    style={styles.textInput}
                    value={lastName}
                    onChangeText={setLastName}
                    placeholder="Nom"
                    placeholderTextColor={palette.modalText}
                  />
                </View>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Téléphone</Text>
                <View style={styles.textInputContainer}>
                  <Ionicons name="call-outline" size={18} color={palette.modalButton} />
                  <TextInput
                    style={styles.textInput}
                    value={phone}
                    onChangeText={setPhone}
                    placeholder="+41 22 123 45 67"
                    placeholderTextColor={palette.modalText}
                    keyboardType="phone-pad"
                  />
                </View>
              </View>
            </View>

            {/* Adresse de domicile */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>
                {isInstitution
                  ? "📍 Adresse de l'institution"
                  : "🏠 Adresse de domicile"}
              </Text>
              <AddressSelector
                label="Adresse complète *"
                value={domicileAddress}
                onChange={(address, suggestion) => {
                  setDomicileAddress(address);
                  setDomicileSuggestion(suggestion);
                }}
                icon="location-outline"
              />
            </View>
          </ScrollView>

          <View style={styles.modalActions}>
            <TouchableOpacity
              style={styles.modalCancel}
              onPress={onClose}
              disabled={loading}
            >
              <Text style={styles.modalCancelText}>Annuler</Text>
            </TouchableOpacity>
            <View style={{ flex: 1 }} />
            <TouchableOpacity
              style={[
                styles.modalSave,
                loading && styles.modalSaveDisabled,
              ]}
              onPress={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#FFFFFF" size="small" />
              ) : (
                <>
                  <Text style={styles.modalSaveText}>Créer</Text>
                  <Ionicons name="checkmark" size={16} color="#FFFFFF" />
                </>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    backgroundColor: palette.modalOverlay,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  modalCard: {
    width: "100%",
    maxWidth: 500,
    maxHeight: "90%",
    backgroundColor: palette.modalBackground,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: palette.modalBorder,
    shadowColor: "rgba(15,54,43,0.15)",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 1,
    shadowRadius: 24,
    elevation: 8,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    padding: 24,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(15,54,43,0.08)",
  },
  modalTitle: {
    color: palette.modalTitle,
    fontSize: 20,
    fontWeight: "700",
  },
  modalSubtitle: {
    color: palette.modalText,
    fontSize: 13,
    marginTop: 4,
  },
  closeButton: {
    padding: 4,
  },
  modalScroll: {
    flex: 1,
  },
  modalContent: {
    padding: 24,
  },
  errorContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "rgba(239,68,68,0.1)",
    padding: 12,
    borderRadius: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.2)",
  },
  errorText: {
    flex: 1,
    color: palette.error,
    fontSize: 14,
  },
  section: {
    marginBottom: 24,
    gap: 16,
  },
  sectionTitle: {
    color: palette.modalTitle,
    fontSize: 16,
    fontWeight: "700",
  },
  inputGroup: {
    gap: 8,
  },
  inputLabel: {
    color: palette.modalTitle,
    fontSize: 14,
    fontWeight: "600",
  },
  textInputContainer: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: palette.modalBackground,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: palette.modalBorder,
    paddingHorizontal: 14,
    paddingVertical: 12,
    gap: 10,
  },
  textInput: {
    flex: 1,
    color: palette.modalTitle,
    fontSize: 15,
    padding: 0,
  },
  checkboxRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 8,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: palette.modalBorder,
    backgroundColor: palette.modalBackground,
    alignItems: "center",
    justifyContent: "center",
  },
  checkboxLabel: {
    color: palette.modalTitle,
    fontSize: 15,
    flex: 1,
  },
  modalActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    padding: 24,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: "rgba(15,54,43,0.08)",
  },
  modalCancel: {
    paddingHorizontal: 18,
    paddingVertical: 12,
  },
  modalCancelText: {
    color: palette.modalCancelText,
    fontSize: 15,
    fontWeight: "600",
  },
  modalSave: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: palette.modalButton,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 14,
  },
  modalSaveDisabled: {
    backgroundColor: "rgba(10,127,89,0.4)",
  },
  modalSaveText: {
    color: palette.modalButtonText,
    fontSize: 15,
    fontWeight: "700",
  },
});

