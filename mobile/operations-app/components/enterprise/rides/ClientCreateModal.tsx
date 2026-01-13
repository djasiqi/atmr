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
import { shadowPresets } from "@/styles/shadowStyles";

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
  const [gender, setGender] = useState<'male' | 'female' | ''>(''); // ✅ Civilité obligatoire
  const [avsNumber, setAvsNumber] = useState(''); // ✅ Numéro AVS optionnel
  const [phone, setPhone] = useState("");
  const [isInstitution, setIsInstitution] = useState(false);
  const [institutionName, setInstitutionName] = useState("");

  // ✅ Priority 2: Nouveaux champs de contact et tarif
  const [contactEmail, setContactEmail] = useState("");
  const [contactPhone, setContactPhone] = useState("");
  const [preferentialRate, setPreferentialRate] = useState("");

  // Adresse de domicile
  const [domicileAddress, setDomicileAddress] = useState("");
  const [domicileSuggestion, setDomicileSuggestion] = useState<AddressSuggestion | undefined>();
  const [residenceFacility, setResidenceFacility] = useState(""); // Établissement de résidence

  // Fonction utilitaire pour nettoyer les adresses avec doublons
  const cleanAddressString = (address: string): string => {
    if (!address) return address;

    // Séparer par virgules
    const parts = address.split(",").map((p) => p.trim()).filter((p) => p.length > 0);

    if (parts.length < 3) {
      return address; // Pas assez de parties pour nettoyer
    }

    // Normaliser les abréviations de rue pour comparaison (retirer le préfixe)
    const normalizeStreet = (street: string): string => {
      return street
        .toLowerCase()
        .replace(/^(av\.|av|ave|avenue)\s+/i, "")
        .replace(/^(rue|r\.)\s+/i, "")
        .replace(/^(chemin|ch\.)\s+/i, "")
        .replace(/^(boulevard|bd|bvd|bd\.)\s+/i, "")
        .replace(/^(place|pl\.)\s+/i, "")
        .trim();
    };

    // Extraire le numéro d'une partie
    const extractNumber = (part: string): string | null => {
      const match = part.match(/(\d+[a-z]?)$/i);
      return match ? match[1].toLowerCase() : null;
    };

    // Extraire la rue d'une partie (sans le numéro)
    const extractStreet = (part: string): string => {
      return part.replace(/\s+\d+[a-z]?$/i, "").trim();
    };

    // Pattern de duplication: "Rue abrégée + numéro, Rue complète, Numéro, ..."
    // Exemple: "Av. Ernest-Pictet 9, Avenue Ernest-Pictet, 9, 1203, Genève"
    const firstPart = parts[0]; // "Av. Ernest-Pictet 9"
    const secondPart = parts[1]; // "Avenue Ernest-Pictet"
    const thirdPart = parts.length > 2 ? parts[2] : null; // "9"

    const firstNum = extractNumber(firstPart); // "9"
    const thirdNum = thirdPart ? extractNumber(thirdPart) : null; // "9"

    // Si la première partie a un numéro et la troisième partie est juste un numéro identique
    if (firstNum && thirdNum && firstNum === thirdNum) {
      const firstStreetRaw = extractStreet(firstPart); // "Av. Ernest-Pictet"
      const firstStreet = normalizeStreet(firstStreetRaw); // "ernest-pictet"
      const secondStreet = normalizeStreet(secondPart); // "ernest-pictet"

      // Vérifier si c'est la même rue (normalisée, sans le préfixe)
      if (firstStreet === secondStreet && firstStreet.length > 0) {
        // Construire l'adresse nettoyée: "Rue complète + numéro, code postal, ville"
        const fullStreet = secondPart; // Garder la version complète "Avenue Ernest-Pictet"
        const cleanedParts = [`${fullStreet} ${firstNum}`]; // "Avenue Ernest-Pictet 9"

        // Ajouter le reste (code postal, ville, etc.) en sautant le numéro dupliqué (index 2)
        for (let i = 3; i < parts.length; i++) {
          cleanedParts.push(parts[i]);
        }

        return cleanedParts.join(", ");
      }
    }

    // Si pas de pattern de duplication détecté, retourner l'adresse originale
    return address;
  };

  useEffect(() => {
    if (!visible) {
      // Reset form when modal closes
      setFirstName("");
      setLastName("");
      setGender(''); // ✅ Reset civilité
      setAvsNumber(''); // ✅ Reset numéro AVS
      setPhone("");
      setIsInstitution(false);
      setInstitutionName("");
      // ✅ Priority 2: Reset nouveaux champs
      setContactEmail("");
      setContactPhone("");
      setPreferentialRate("");
      setDomicileAddress("");
      setDomicileSuggestion(undefined);
      setResidenceFacility("");
      setError(null);
    }
  }, [visible]);

  const handleSubmit = async () => {
    // Validation
    if (!firstName.trim() || !lastName.trim()) {
      setError("Le prénom et le nom sont requis");
      return;
    }

    // ✅ Validation civilité obligatoire
    if (!gender) {
      setError("La civilité (Madame/Monsieur) est obligatoire");
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
      // Construire l'adresse complète (déjà nettoyée lors de la sélection)
      let fullAddress = domicileAddress;
      if (domicileSuggestion && !domicileAddress) {
        // Si l'adresse n'est pas encore définie, utiliser la suggestion et la nettoyer
        fullAddress = cleanAddressString(domicileSuggestion.label || domicileSuggestion.address || "");
      }

      // ✅ Parser l'adresse avec détection d'établissement
      // Mots-clés pour détecter les établissements
      const establishmentKeywords = [
        'clinique', 'hôpital', 'hopital', 'hospital', 'ems', 'foyer', 'centre',
        'maison', 'résidence', 'residence', 'institut', 'institution',
        'établissement', 'etablissement', 'cabinet', 'dispensaire',
        'polyclinique', 'sanatorium', 'maison de santé', 'maison de retraite',
      ];

      const isEstablishment = (text: string): boolean => {
        if (!text) return false;
        const lowerText = text.toLowerCase();
        return establishmentKeywords.some((keyword) => lowerText.includes(keyword));
      };

      // Parser l'adresse pour extraire établissement, rue, code postal et ville
      // Format attendu: "Établissement, Rue Numéro, Code postal, Ville" ou "Rue Numéro, Code postal, Ville"
      let establishment = "";
      let domicileStreet = fullAddress;
      let domicileZip = "";
      let domicileCity = "";

      // Séparer par virgules
      const parts = fullAddress.split(',').map((p) => p.trim()).filter((p) => p.length > 0);

      if (parts.length >= 3) {
        // Vérifier si la première partie est un établissement
        if (isEstablishment(parts[0])) {
          establishment = parts[0];
          // La deuxième partie est la rue (avec ou sans numéro)
          domicileStreet = parts[1];
          // La troisième partie peut être le code postal ou la ville
          if (/^\d{4}$/.test(parts[2])) {
            domicileZip = parts[2];
            // La quatrième partie est la ville
            if (parts.length >= 4) {
              domicileCity = parts[3].replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
            }
          } else {
            // La troisième partie est la ville (code postal manquant ou dans la deuxième partie)
            domicileCity = parts[2].replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
            // Essayer d'extraire le code postal de la deuxième partie si possible
            const zipMatch = parts[1].match(/\b(\d{4})\b/);
            if (zipMatch) {
              domicileZip = zipMatch[1];
              // Retirer le code postal de la rue
              domicileStreet = parts[1].replace(/\b\d{4}\b/, '').trim();
            }
          }
        } else {
          // Pas d'établissement : format classique "Rue Numéro, CP, Ville"
          domicileStreet = parts[0];
          if (/^\d{4}$/.test(parts[1])) {
            domicileZip = parts[1];
            if (parts.length >= 3) {
              domicileCity = parts[2].replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
            }
          } else {
            // Format "Rue, CP Ville"
            const zipCityMatch = parts[1].match(/^(\d{4})\s+(.+?)(?:\s*,\s*(?:Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia))?$/i);
            if (zipCityMatch) {
              domicileZip = zipCityMatch[1];
              domicileCity = zipCityMatch[2].trim();
            } else {
              domicileCity = parts[1].replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
            }
          }
        }
      } else if (parts.length === 2) {
        // Format "Rue Numéro, CP Ville" ou "Établissement, Rue"
        if (isEstablishment(parts[0])) {
          establishment = parts[0];
          domicileStreet = parts[1];
          // Code postal et ville manquants dans ce format
        } else {
          // Format classique "Rue Numéro, CP Ville"
          domicileStreet = parts[0];
          const zipCityMatch = parts[1].match(/^(\d{4})\s+(.+?)(?:\s*,\s*(?:Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia))?$/i);
          if (zipCityMatch) {
            domicileZip = zipCityMatch[1];
            domicileCity = zipCityMatch[2].trim();
          } else {
            domicileCity = parts[1].replace(/\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$/i, '').trim();
          }
        }
      } else {
        // Format simple : essayer de trouver le code postal
        const zipMatch = fullAddress.match(/(\d{4})/);
        if (zipMatch) {
          const zipIndex = fullAddress.indexOf(zipMatch[1]);
          domicileStreet = fullAddress.substring(0, zipIndex).replace(/,\s*$/, "").trim();
          domicileZip = zipMatch[1];
          const afterZip = fullAddress.substring(zipIndex + 4).trim();
          const cityMatch = afterZip.match(/^,\s*([^,]+?)(?:\s*,\s*(?:Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia))?$/i);
          if (cityMatch) {
            domicileCity = cityMatch[1].trim();
          }
        }
      }

      // Construire l'adresse complète au format frontend: "Rue, Code postal, Ville"
      const addressComplete = domicileZip && domicileCity
        ? `${domicileStreet}, ${domicileZip}, ${domicileCity}`.trim()
        : fullAddress;

      // Construire l'adresse de facturation (même que domicile pour l'instant)
      const billingAddressComplete = addressComplete;

      const payload: CreateClientPayload = {
        client_type: "PRIVATE",
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        // ✅ Civilité obligatoire
        gender: gender as 'male' | 'female',
        phone: phone.trim() || undefined,
        // ✅ Numéro AVS optionnel
        avs_number: avsNumber.trim() || undefined,
        is_institution: isInstitution,
        institution_name: isInstitution ? institutionName.trim() : undefined,
        // ✅ Établissement de résidence si détecté
        residence_facility: establishment || undefined,
        // ✅ Priority 2: Nouveaux champs de contact et tarif
        contact_email: contactEmail.trim() || undefined,
        contact_phone: contactPhone.trim() || undefined,
        preferential_rate: preferentialRate ? parseFloat(preferentialRate) : undefined,
        // Adresse complète (comme dans le frontend)
        address: addressComplete,
        // Adresse de domicile structurée
        domicile_address: domicileStreet || undefined,
        domicile_zip: domicileZip || undefined,
        domicile_city: domicileCity || undefined,
        // Coordonnées GPS du domicile
        domicile_lat: domicileSuggestion?.lat || null,
        domicile_lon: domicileSuggestion?.lon || null,
        // Adresse de facturation (même que domicile pour l'instant)
        billing_address: billingAddressComplete,
        billing_lat: domicileSuggestion?.lat || null,
        billing_lon: domicileSuggestion?.lon || null,
      };

      console.log("[ClientCreateModal] Payload avant envoi:", JSON.stringify(payload, null, 2));
      console.log("[ClientCreateModal] Adresse parsée:", {
        fullAddress,
        domicileStreet,
        domicileZip,
        domicileCity,
        addressComplete,
      });

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
      console.error("[ClientCreateModal] Erreur complète:", err);
      console.error("[ClientCreateModal] Response data:", err?.response?.data);
      console.error("[ClientCreateModal] Response status:", err?.response?.status);

      // Essayer d'extraire les messages d'erreur de validation
      let errorMessage = "Erreur lors de la création du client";

      if (err?.response?.data) {
        const data = err.response.data;

        // Si c'est une erreur de validation Marshmallow
        if (data.errors) {
          const validationErrors = Object.entries(data.errors)
            .map(([field, messages]: [string, any]) => {
              const msg = Array.isArray(messages) ? messages.join(", ") : String(messages);
              return `${field}: ${msg}`;
            })
            .join("\n");
          errorMessage = `Erreurs de validation:\n${validationErrors}`;
        } else if (data.error) {
          errorMessage = data.error;
        } else if (data.message) {
          errorMessage = data.message;
        } else if (typeof data === "string") {
          errorMessage = data;
        }
      } else if (err?.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
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

          <View style={styles.scrollContainer}>
            <ScrollView
              style={styles.modalScroll}
              contentContainerStyle={styles.modalContent}
              showsVerticalScrollIndicator={false}
              keyboardShouldPersistTaps="handled"
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

              {/* ✅ Civilité obligatoire */}
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Civilité *</Text>
                <View style={styles.genderButtonGroup}>
                  <TouchableOpacity
                    style={[
                      styles.genderButton,
                      gender === 'male' && styles.genderButtonActive
                    ]}
                    onPress={() => setGender('male')}
                  >
                    <Ionicons
                      name="male"
                      size={18}
                      color={gender === 'male' ? '#FFFFFF' : palette.modalButton}
                    />
                    <Text style={[
                      styles.genderButtonText,
                      gender === 'male' && styles.genderButtonTextActive
                    ]}>
                      Monsieur
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[
                      styles.genderButton,
                      gender === 'female' && styles.genderButtonActive
                    ]}
                    onPress={() => setGender('female')}
                  >
                    <Ionicons
                      name="female"
                      size={18}
                      color={gender === 'female' ? '#FFFFFF' : palette.modalButton}
                    />
                    <Text style={[
                      styles.genderButtonText,
                      gender === 'female' && styles.genderButtonTextActive
                    ]}>
                      Madame
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>

              {/* ✅ Numéro AVS optionnel */}
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Numéro AVS</Text>
                <View style={styles.textInputContainer}>
                  <Ionicons name="card-outline" size={18} color={palette.modalButton} />
                  <TextInput
                    style={styles.textInput}
                    value={avsNumber}
                    onChangeText={setAvsNumber}
                    placeholder="756.XXXX.XXXX.XX"
                    placeholderTextColor={palette.modalText}
                    keyboardType="numbers-and-punctuation"
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

            {/* ✅ Priority 2: Coordonnées de facturation (optionnelles) */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>📋 Coordonnées de facturation</Text>
              <Text style={styles.sectionDescription}>
                Informations optionnelles pour la facturation
              </Text>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Email de contact</Text>
                <View style={styles.textInputContainer}>
                  <Ionicons name="mail-outline" size={18} color={palette.modalButton} />
                  <TextInput
                    style={styles.textInput}
                    value={contactEmail}
                    onChangeText={setContactEmail}
                    placeholder="facturation@example.com"
                    placeholderTextColor={palette.modalText}
                    keyboardType="email-address"
                    autoCapitalize="none"
                  />
                </View>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Téléphone de contact</Text>
                <View style={styles.textInputContainer}>
                  <Ionicons name="call-outline" size={18} color={palette.modalButton} />
                  <TextInput
                    style={styles.textInput}
                    value={contactPhone}
                    onChangeText={setContactPhone}
                    placeholder="+41 22 123 45 67"
                    placeholderTextColor={palette.modalText}
                    keyboardType="phone-pad"
                  />
                </View>
              </View>

              {!isInstitution && (
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>💰 Tarif préférentiel (CHF)</Text>
                  <View style={styles.textInputContainer}>
                    <Ionicons name="cash-outline" size={18} color={palette.modalButton} />
                    <TextInput
                      style={styles.textInput}
                      value={preferentialRate}
                      onChangeText={setPreferentialRate}
                      placeholder="Ex: 45.00"
                      placeholderTextColor={palette.modalText}
                      keyboardType="decimal-pad"
                    />
                  </View>
                  <Text style={styles.inputHint}>
                    Prix d'un trajet simple. Laisser vide pour le tarif standard.
                  </Text>
                </View>
              )}
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
                  // Nettoyer l'adresse dès la sélection pour éviter les doublons
                  let cleanedAddress = address;
                  if (suggestion?.label) {
                    cleanedAddress = cleanAddressString(suggestion.label);
                  } else if (address) {
                    cleanedAddress = cleanAddressString(address);
                  }
                  setDomicileAddress(cleanedAddress);
                  setDomicileSuggestion(suggestion);
                }}
                icon="location-outline"
              />
            </View>
            </ScrollView>
          </View>

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
    height: "90%",
    backgroundColor: palette.modalBackground,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: palette.modalBorder,
    ...shadowPresets.large, // ✅ Compatible web/native
    flexDirection: "column",
    overflow: "hidden",
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
  scrollContainer: {
    flex: 1,
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
  sectionDescription: {
    color: palette.modalText,
    fontSize: 13,
    marginTop: -8,
    marginBottom: 8,
  },
  inputGroup: {
    gap: 8,
  },
  inputLabel: {
    color: palette.modalTitle,
    fontSize: 14,
    fontWeight: "600",
  },
  inputHint: {
    color: palette.modalText,
    fontSize: 12,
    marginTop: -4,
    fontStyle: "italic",
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
  // ✅ Styles pour les boutons de civilité
  genderButtonGroup: {
    flexDirection: 'row',
    gap: 12,
  },
  genderButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: palette.modalBorder,
    backgroundColor: palette.modalBackground,
  },
  genderButtonActive: {
    backgroundColor: palette.modalButton,
    borderColor: palette.modalButton,
  },
  genderButtonText: {
    color: palette.modalTitle,
    fontSize: 15,
    fontWeight: '600',
  },
  genderButtonTextActive: {
    color: '#FFFFFF',
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

