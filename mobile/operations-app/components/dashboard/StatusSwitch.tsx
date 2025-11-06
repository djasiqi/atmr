import React, { useState } from 'react';
import { View, Text, Switch, ActivityIndicator, Alert } from 'react-native';
import { updateDriverAvailability } from '@/services/api';

type Props = {
  isAvailable: boolean;
  onStatusChange?: (newStatus: boolean) => void;
};

const StatusSwitch: React.FC<Props> = ({ isAvailable, onStatusChange }) => {
  const [status, setStatus] = useState(isAvailable);
  const [loading, setLoading] = useState(false);

  const toggleStatus = async () => {
    const newStatus = !status;
    setLoading(true);
    try {
      await updateDriverAvailability(newStatus);
      setStatus(newStatus);
      onStatusChange?.(newStatus);
    } catch (error) {
      Alert.alert("Erreur", "Impossible de mettre à jour votre statut. Réessayez plus tard.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View className="flex-row items-center justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-200">
      <View>
        <Text className="text-base font-semibold">Statut</Text>
        <Text className="text-sm text-gray-500">
          {status ? '✅ Disponible pour les courses' : '⏸️ En pause'}
        </Text>
      </View>
      {loading ? (
        <ActivityIndicator size="small" color="#666" />
      ) : (
        <Switch value={status} onValueChange={toggleStatus} />
      )}
    </View>
  );
};

export default StatusSwitch;


