import React, { useState } from "react";
import {
  View,
  Text,
  Modal,
  TouchableOpacity,
  Pressable,
  StyleSheet,
  ScrollView,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

type CancelOption = {
  id: string;
  label: string;
  description: string;
  isClientFault: boolean;
};

const CANCEL_OPTIONS: CancelOption[] = [
  {
    id: "LAST_MINUTE",
    label: "Annulation dernière minute",
    description: "Le client ou l'organisation a annulé tardivement",
    isClientFault: true,
  },
  {
    id: "CLIENT_NO_SHOW",
    label: "Client absent",
    description: "Le client n'était pas au lieu de rendez-vous",
    isClientFault: true,
  },
  {
    id: "CLIENT_REQUEST",
    label: "Demande du client",
    description: "Le client a demandé d'annuler la course",
    isClientFault: true,
  },
  {
    id: "COMPANY_ISSUE",
    label: "Problème entreprise",
    description: "Problème technique ou organisationnel interne",
    isClientFault: false,
  },
  {
    id: "DELAY",
    label: "Retard important",
    description: "Retard trop important pour honorer la course",
    isClientFault: false,
  },
  {
    id: "VEHICLE_ISSUE",
    label: "Problème véhicule",
    description: "Panne ou problème mécanique",
    isClientFault: false,
  },
  {
    id: "OTHER",
    label: "Autre raison",
    description: "Autre raison nécessitant une justification",
    isClientFault: false,
  },
];

type Props = {
  visible: boolean;
  onClose: () => void;
  onConfirm: (reason: string, isClientFault: boolean) => void;
};

export default function CancelJustificationModal({
  visible,
  onClose,
  onConfirm,
}: Props) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const handleConfirm = () => {
    if (!selectedOption) return;
    const option = CANCEL_OPTIONS.find((opt) => opt.id === selectedOption);
    if (option) {
      onConfirm(option.id, option.isClientFault);
      setSelectedOption(null);
    }
  };

  const handleClose = () => {
    setSelectedOption(null);
    onClose();
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={handleClose}
    >
      <Pressable style={s.backdrop} onPress={handleClose}>
        <View
          style={s.sheet}
          onStartShouldSetResponder={() => true}
          onTouchEnd={(e) => e.stopPropagation()}
        >
          {/* Header — icône danger + titre + fermer */}
          <View style={s.header}>
            <View style={s.headerLeft}>
              <View style={s.headerIcon}>
                <Ionicons name="alert-circle" size={16} color="#dc3545" />
              </View>
              <Text style={s.title}>Justifier l'annulation</Text>
            </View>
            <TouchableOpacity
              onPress={handleClose}
              style={s.closeBtn}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Ionicons name="close" size={18} color="#94A3B8" />
            </TouchableOpacity>
          </View>

          {/* Sous-titre */}
          <Text style={s.subtitle}>
            Sélectionnez la raison de l'annulation de cette course.
          </Text>

          {/* Options */}
          <ScrollView
            style={s.optionsList}
            showsVerticalScrollIndicator={false}
            bounces={false}
          >
            {CANCEL_OPTIONS.map((option) => {
              const isSelected = selectedOption === option.id;
              return (
                <TouchableOpacity
                  key={option.id}
                  activeOpacity={0.7}
                  style={[
                    s.optionCard,
                    isSelected && s.optionCardSelected,
                  ]}
                  onPress={() => setSelectedOption(option.id)}
                >
                  {/* Radio + label */}
                  <View style={s.optionRow}>
                    <View style={[s.radio, isSelected && s.radioSelected]}>
                      {isSelected && <View style={s.radioDot} />}
                    </View>
                    <View style={s.optionTextWrap}>
                      <Text
                        style={[s.optionLabel, isSelected && s.optionLabelSelected]}
                        numberOfLines={1}
                      >
                        {option.label}
                      </Text>
                      <Text style={s.optionDesc} numberOfLines={2}>
                        {option.description}
                      </Text>
                    </View>
                  </View>

                  {/* Badge facturation */}
                  {option.isClientFault ? (
                    <View style={s.badgeBillable}>
                      <Ionicons name="card-outline" size={11} color="#92400e" />
                      <Text style={s.badgeBillableText}>Facturable</Text>
                    </View>
                  ) : (
                    <View style={s.badgeNoBill}>
                      <Ionicons name="checkmark-circle-outline" size={11} color="#00796B" />
                      <Text style={s.badgeNoBillText}>Non facturable</Text>
                    </View>
                  )}
                </TouchableOpacity>
              );
            })}
            <View style={{ height: 8 }} />
          </ScrollView>

          {/* Footer */}
          <View style={s.footer}>
            <TouchableOpacity style={s.btnCancel} onPress={handleClose}>
              <Text style={s.btnCancelText}>Retour</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[s.btnConfirm, !selectedOption && s.btnConfirmDisabled]}
              onPress={handleConfirm}
              disabled={!selectedOption}
            >
              <Ionicons
                name="close-circle-outline"
                size={15}
                color={selectedOption ? "#fff" : "rgba(255,255,255,0.5)"}
              />
              <Text
                style={[
                  s.btnConfirmText,
                  !selectedOption && s.btnConfirmTextDisabled,
                ]}
              >
                Confirmer
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </Pressable>
    </Modal>
  );
}

const BRAND = "#00796B";
const BRAND_DARK = "#00695C";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const DANGER = "#dc3545";
const BG = "#f8fafc";

const shadow = Platform.OS === "web"
  ? { boxShadow: "0 -4px 24px rgba(0,0,0,0.12)" }
  : {
      shadowColor: "#000",
      shadowOffset: { width: 0, height: -4 },
      shadowOpacity: 0.1,
      shadowRadius: 16,
      elevation: 12,
    };

const s = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.45)",
    justifyContent: "flex-end",
  },
  sheet: {
    backgroundColor: "#fff",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: "85%",
    ...shadow,
  },

  // Header
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  headerLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    flex: 1,
  },
  headerIcon: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: "rgba(220,53,69,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 16,
    fontWeight: "600",
    color: TEXT,
    letterSpacing: -0.2,
  },
  closeBtn: {
    width: 28,
    height: 28,
    borderRadius: 6,
    alignItems: "center",
    justifyContent: "center",
  },

  // Subtitle
  subtitle: {
    fontSize: 13,
    color: TEXT_SEC,
    lineHeight: 19,
    paddingHorizontal: 20,
    paddingTop: 14,
    paddingBottom: 12,
  },

  // Options list
  optionsList: {
    paddingHorizontal: 16,
    maxHeight: 380,
  },
  optionCard: {
    backgroundColor: BG,
    borderRadius: 12,
    padding: 14,
    marginBottom: 8,
    borderWidth: 1.5,
    borderColor: "transparent",
  },
  optionCardSelected: {
    backgroundColor: "rgba(0,121,107,0.04)",
    borderColor: BRAND,
  },
  optionRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
  },
  radio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: TEXT_MUTED,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 1,
  },
  radioSelected: {
    borderColor: BRAND,
  },
  radioDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: BRAND,
  },
  optionTextWrap: {
    flex: 1,
  },
  optionLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: TEXT,
    marginBottom: 2,
  },
  optionLabelSelected: {
    color: BRAND_DARK,
  },
  optionDesc: {
    fontSize: 12,
    color: TEXT_SEC,
    lineHeight: 17,
  },

  // Badges
  badgeBillable: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    alignSelf: "flex-start",
    marginTop: 8,
    marginLeft: 32,
    backgroundColor: "rgba(245,158,11,0.1)",
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  badgeBillableText: {
    fontSize: 11,
    fontWeight: "600",
    color: "#92400e",
  },
  badgeNoBill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    alignSelf: "flex-start",
    marginTop: 8,
    marginLeft: 32,
    backgroundColor: "rgba(0,121,107,0.06)",
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  badgeNoBillText: {
    fontSize: 11,
    fontWeight: "600",
    color: BRAND,
  },

  // Footer
  footer: {
    flexDirection: "row",
    gap: 10,
    paddingHorizontal: 20,
    paddingVertical: 16,
    paddingBottom: Platform.OS === "ios" ? 32 : 16,
    borderTopWidth: 1,
    borderTopColor: BORDER,
  },
  btnCancel: {
    flex: 1,
    height: 40,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: BG,
    borderWidth: 1,
    borderColor: "rgba(0,0,0,0.08)",
  },
  btnCancelText: {
    fontSize: 14,
    fontWeight: "500",
    color: TEXT_SEC,
  },
  btnConfirm: {
    flex: 1.5,
    height: 40,
    borderRadius: 10,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: DANGER,
    gap: 6,
  },
  btnConfirmDisabled: {
    opacity: 0.4,
  },
  btnConfirmText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#fff",
  },
  btnConfirmTextDisabled: {
    color: "rgba(255,255,255,0.7)",
  },
});
