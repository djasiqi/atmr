import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  
  //bloc principal de l’en-tête
  container: { 
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 16,
    backgroundColor: '#F7F9FB',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },

  // ligne de bienvenue (ex: Bonjour, Drin 👋)
  title: { 
    fontSize: 22,
    fontWeight: '700',
    color: '#004D40',
    marginBottom: 5,
  },

  // texte de date ou info complémentaire
  subtitle: { 
    fontSize: 14,
    color: '#555',
    marginBottom: 5,
  },

  // statut du chauffeur (disponible, indisponible…)
  status: { 
    fontSize: 14,
    color: '#00796B',
    fontWeight: '500',
  },
});
