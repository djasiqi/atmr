// components/dashboard/ConfirmCompletionModal.tsx

import React from 'react';
import { Modal, View, Text, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { modalStyles } from '@/styles/ConfirmCompletionModalStyles';

// Props pour le modal de confirmation de fin de mission
type Props = {
  visible: boolean; 
  onClose: () => void;
  onConfirm: () => void;
  isLoading?: boolean; // état de chargement pour bloquer le bouton et afficher un spinner
};

const ConfirmCompletionModal: React.FC<Props> = ({
  visible,
  onClose,
  onConfirm,
  isLoading = false, // valeur par défaut
}) => {
  return (
    <Modal
      transparent
      animationType="fade"
      visible={visible}
      onRequestClose={onClose}
    >
      <View style={modalStyles.overlay}>
        <View style={modalStyles.modalContainer}>
          <View style={modalStyles.iconWrapper}>
            <Ionicons name="checkmark-circle-outline" size={48} color="green" />
          </View>
          <Text style={modalStyles.title}>Confirmer la fin de mission ?</Text>
          <Text style={modalStyles.subtitle}>
            Une fois la mission marquée comme terminée, elle sera archivée.
          </Text>
          <View style={modalStyles.buttonRow}>
            <TouchableOpacity
              onPress={onClose}
              style={[modalStyles.cancelButton, isLoading && { opacity: 0.6 }]}
              disabled={isLoading}
            >
              <Text style={modalStyles.cancelText}>Annuler</Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => {
                console.log("Bouton Confirmer pressé !");
                onConfirm();
              }}
              style={[modalStyles.confirmButton, isLoading && { opacity: 0.6 }]}
              disabled={isLoading}
            >
              {isLoading ? (
                <ActivityIndicator size="small" />
              ) : (
                <Text style={modalStyles.confirmText}>Confirmer</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

export default ConfirmCompletionModal;
