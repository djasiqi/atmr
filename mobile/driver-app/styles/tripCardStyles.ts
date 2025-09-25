// styles/tripCardStyles.ts

import { StyleSheet } from 'react-native';

export const tripCardStyles = StyleSheet.create({
    //Style global de la carte de course
  cardContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 16,
    marginVertical: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },

  // Titre de section (ex: "🕒 Courses assignées", "Matin", "Soirée")
  sectionHeader: {
    fontSize: 18,
    fontWeight: '700',
    paddingTop: 10,
    marginHorizontal: 16,
    paddingBottom: 0,
    color: '#004D40',
  },

  //Texte d’horaire (heures simplifiées)
  timeText: {
    fontSize: 13,
    fontWeight: '600',
    marginHorizontal: 16,
    color: '#00695C',
  },

    //Organise l’en-tête (ex: date + badge de statut) en ligne
  statusBadge: {
    backgroundColor: '#B2DFDB',
    color: '#004D40',
  paddingVertical: 5,
  paddingHorizontal: 16,
  borderRadius: 12,
  fontSize: 13,
  fontWeight: '600',
  alignSelf: 'flex-start',
},

    //Texte principal du trajet (ex: "Rue de la Gare → Clinique XYZ")
  routeText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#00695C',
    marginBottom: 12,
  },
  
  //Texte d'état sous forme de ligne simple
  statusText: {
    fontSize: 13,
    fontWeight: '500',
    marginTop: 5,
    marginLeft: 5,
    color: '#00695C',
  },
  
  //Texte affiché quand il n'y a aucune course
  emptyText: {
    marginTop: 20,
    marginLeft: 5,
    color: '#9E9E9E',
    fontSize: 14,
    textAlign: 'center',
  },

  //Texte secondaire dans une carte, comme l'adresse
  routeSection: {
  marginLeft: 5,
  marginBottom: 4,
  fontSize: 12,
  color: '#212121',
},


//Variante de timeText pour version plus visible
timeEnhanced: {
  fontSize: 14,
  fontWeight: '600',
  marginTop: 5,
  marginLeft:5,
  color: '#00695C',
},


});


