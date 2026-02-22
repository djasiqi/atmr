import { StyleSheet, Platform } from 'react-native';

export const MAP_BRAND = {
  primary: '#00796B',
  primaryDark: '#00695C',
  primaryLight: '#26a69a',
  dropoff: '#1E293B',
  dropoffDark: '#0f172a',
  success: '#22c55e',
  warning: '#f59e0b',
  danger: '#ef4444',
  muted: '#91A3A0',
  text: '#1E293B',
  bg: '#ffffff',
} as const;

/** Style carte Lirie — épuré, professionnel, calme (adapté de la plateforme web) */
export const LIRIE_MAP_STYLE = [
  { featureType: 'poi', stylers: [{ visibility: 'off' }] },
  { featureType: 'poi.medical', stylers: [{ visibility: 'on' }] },
  { featureType: 'poi.medical', elementType: 'labels.icon', stylers: [{ saturation: -60 }] },
  { featureType: 'transit', stylers: [{ visibility: 'simplified' }] },
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#c8dce8' }] },
  { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#94A3B8' }] },
  { featureType: 'landscape.man_made', elementType: 'geometry', stylers: [{ color: '#f0f2f4' }] },
  { featureType: 'landscape.natural', elementType: 'geometry', stylers: [{ color: '#e4ebe7' }] },
  { featureType: 'landscape.natural.terrain', elementType: 'geometry', stylers: [{ color: '#dde5e0' }] },
  { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#d5dbe0' }] },
  { featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{ color: '#c3cad0' }] },
  { featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{ color: '#64748B' }] },
  { featureType: 'road.arterial', elementType: 'geometry', stylers: [{ color: '#e0e5e9' }] },
  { featureType: 'road.local', elementType: 'geometry', stylers: [{ color: '#ebeef1' }] },
  { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#94A3B8' }] },
  { featureType: 'administrative', elementType: 'labels.text.fill', stylers: [{ color: '#64748B' }] },
  { featureType: 'administrative.locality', elementType: 'labels.text.fill', stylers: [{ color: '#1E293B' }] },
  { featureType: 'administrative.locality', elementType: 'labels.text.stroke', stylers: [{ color: '#ffffff' }, { weight: 3 }] },
  { featureType: 'administrative.neighborhood', elementType: 'labels.text.fill', stylers: [{ color: '#94A3B8' }] },
];

const CONTENT_WIDTH = 380;

const containerShadow = Platform.OS === 'web'
  ? { boxShadow: '0 4px 16px rgba(0,0,0,0.08)' }
  : {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.08,
      shadowRadius: 16,
      elevation: 4,
    };

export const styles = StyleSheet.create({
  container: {
    height: 200,
    borderRadius: 16,
    overflow: 'hidden',
    marginHorizontal: 20,
    marginTop: 8,
    backgroundColor: '#f0f2f4',
    borderWidth: 1,
    borderColor: 'rgba(15,54,43,0.06)',
    ...containerShadow,
    ...(Platform.OS === 'web' ? { width: CONTENT_WIDTH, alignSelf: 'center' as const, marginHorizontal: 0 } : {}),
  },
  map: {
    flex: 1,
  },

  // Marker position chauffeur — cercle brand
  markerPickup: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: MAP_BRAND.primary,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: '#fff',
    ...Platform.select({
      ios: { shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 3 },
      android: { elevation: 5 },
    }),
  },

  // Marker destination — cercle sombre
  markerDropoff: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: MAP_BRAND.dropoff,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: '#fff',
    ...Platform.select({
      ios: { shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 3 },
      android: { elevation: 5 },
    }),
  },

  // Web placeholder
  webPlaceholder: Platform.OS === 'web' ? { width: CONTENT_WIDTH, height: 170 } : { height: 72 },
  webPlaceholderInner: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f2f4',
    borderRadius: 16,
  },
  webPlaceholderText: {
    fontSize: 13,
    color: '#94A3B8',
  },
});
