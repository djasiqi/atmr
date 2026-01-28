import { StyleSheet, Platform } from 'react-native';

// Styles shadow conditionnels pour éviter l'avertissement de dépréciation
const containerShadow = Platform.OS === 'web'
  ? { boxShadow: '0 1px 2px rgba(0,0,0,0.1)' }
  : {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.1,
      shadowRadius: 2,
      elevation: 2,
    };

const CONTENT_WIDTH = 380;
const MAP_HEIGHT_WEB = 170;

export const styles = StyleSheet.create({
  container: {
    height: 220,
    borderRadius: 12,
    overflow: 'hidden',
    marginHorizontal: 16,
    marginTop: 12,
    ...containerShadow,
    ...(Platform.OS === 'web' ? { width: CONTENT_WIDTH, alignSelf: 'center' as const, marginHorizontal: 0 } : {}),
  },
  map: {
    flex: 1,
  },
  // Web : placeholder 380×493 pour aligner avec la card
  webPlaceholder: Platform.OS === 'web' ? { width: CONTENT_WIDTH, height: MAP_HEIGHT_WEB } : { height: 72 },
  webPlaceholderInner: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F0F2F1',
  },
  webPlaceholderText: {
    fontSize: 13,
    color: '#91A59D',
  },
});
