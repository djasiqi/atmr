// components/dashboard/MissionMap.web.tsx
// Version web : carte indisponible — placeholder discret pour ne pas voler l’attention
import React from 'react';
import { View, Text } from 'react-native';
import { styles } from '@/styles/missionMapStyles';

type Props = {
  location: { coords: { latitude: number; longitude: number } };
  destination: string;
  /** Largeur responsive (alignée avec la card) */
  contentWidth?: number;
  /** Hauteur du bloc carte (responsive) */
  mapHeight?: number;
};

const MissionMap: React.FC<Props> = ({ contentWidth, mapHeight }) => {
  const dynamicStyle =
    contentWidth != null || mapHeight != null
      ? {
        ...(contentWidth != null && { width: contentWidth, alignSelf: 'center' as const, marginHorizontal: 0 }),
        ...(mapHeight != null && { height: mapHeight }),
      }
      : undefined;

  return (
    <View style={[styles.container, styles.webPlaceholder, dynamicStyle]}>
      <View style={styles.webPlaceholderInner}>
        <Text style={styles.webPlaceholderText}>
          Carte indisponible sur le web
        </Text>
      </View>
    </View>
  );
};

export default MissionMap;

