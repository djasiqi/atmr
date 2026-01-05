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

export const styles = StyleSheet.create({
  container: {
    height: 220,
    borderRadius: 12,
    overflow: 'hidden',
    marginHorizontal: 16,
    marginTop: 12,
    ...containerShadow,
  },
  map: {
    flex: 1,
  },
});
