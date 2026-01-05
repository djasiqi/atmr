// src/styles/tripDetailsStyles.ts
import { StyleSheet, Platform } from 'react-native';

// Styles shadow conditionnels pour éviter l'avertissement de dépréciation
const containerShadow = Platform.OS === 'web'
  ? { boxShadow: '0 2px 6px rgba(0,0,0,0.10)' }
  : {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.10,
      shadowRadius: 6,
      elevation: 4,
    };

const actionButtonShadow = Platform.OS === 'web'
  ? { boxShadow: '0 2px 4px rgba(0,121,107,0.12)' }
  : {
      shadowColor: '#00796B',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.12,
      shadowRadius: 4,
      elevation: 2,
    };

export const styles = StyleSheet.create({
  container: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 18,
    marginHorizontal: 12,
    marginVertical: 12,
    ...containerShadow,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },

  title: {
    fontWeight: '700',
    fontSize: 18,
    color: '#104F55',
    marginBottom: 12,
  },

  rowBetween: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 6,
  },

  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#00796B',
  },

  value: {
    flex: 1,           // occupe tout l'espace restant
    fontSize: 14,
    color: '#232323',
    textAlign: 'right' // aligne le texte à droite
  },

  section: {
    marginBottom: 8,
  },

  sectionHeader: {
    fontSize: 15,
    fontWeight: '700',
    color: '#333',
    marginBottom: 4,
  },

  metaText: {
    fontSize: 13,
    color: '#616161',
    marginTop: 2,
  },

  badge: {
    backgroundColor: '#B2DFDB',
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 12,
  },

  badgeText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#00796B',
  },

  actionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    marginTop: 16,
    gap: 8,
  },

  actionButton: {
    flexGrow: 1,
    backgroundColor: '#00796B',
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
    alignItems: 'center',
    ...actionButtonShadow,
    marginVertical: 4,
  },

  actionButtonText: {
    fontSize: 13,
    color: '#FFFFFF',
    fontWeight: '600',
  },
});
