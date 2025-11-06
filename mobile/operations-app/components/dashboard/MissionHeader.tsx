import React from 'react';
import { View, Text } from 'react-native';
import { styles } from '@/styles/missionHeaderStyles';

type Props = {
  driverName: string;
  date?: string;
};

const MissionHeader: React.FC<Props> = ({ driverName, date }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>
        Bonjour, {driverName ? driverName : 'chauffeur'} <Text>👋</Text>
      </Text>
      {date ? (
        <Text style={styles.subtitle}>Nous sommes le {date}</Text>
      ) : null}
      <Text style={styles.status}>Statut : Disponible pour les courses</Text>
    </View>
  );
};

export default MissionHeader;
