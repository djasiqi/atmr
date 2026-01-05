// components/dashboard/MissionMap.web.tsx
// Version web du composant MissionMap (sans react-native-maps)
import React from 'react';
import { View, Text } from 'react-native';
import { styles } from '@/styles/missionMapStyles';

type Props = {
    location: { coords: { latitude: number; longitude: number } };
    destination: string;
};

const MissionMap: React.FC<Props> = ({ location, destination }) => {
    return (
        <View style={styles.container}>
            <View style={[styles.map, { justifyContent: 'center', alignItems: 'center', backgroundColor: '#e0e0e0' }]}>
                <Text style={{ color: '#666', textAlign: 'center', padding: 20 }}>
                    🗺️ Carte non disponible sur le web{'\n'}
                    Utilisez l'application mobile pour voir la carte
                </Text>
                {destination && (
                    <Text style={{ color: '#999', fontSize: 12, marginTop: 10, textAlign: 'center', paddingHorizontal: 20 }}>
                        Destination: {destination}
                    </Text>
                )}
                {location && (
                    <Text style={{ color: '#999', fontSize: 12, marginTop: 5, textAlign: 'center', paddingHorizontal: 20 }}>
                        Position: {location.coords.latitude.toFixed(4)}, {location.coords.longitude.toFixed(4)}
                    </Text>
                )}
            </View>
        </View>
    );
};

export default MissionMap;

