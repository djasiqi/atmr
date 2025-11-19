// styles/tripCardStyles.ts

import { StyleSheet } from 'react-native';

// ✅ Palette épurée et élégante (cohérente avec le login et mission)
const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
  placeholder: "#91A59D",
};

export const tripCardStyles = StyleSheet.create({
  // ✅ Style global de la carte de course avec design épuré (inspiré du login)
  cardContainer: {
    backgroundColor: palette.card,
    borderRadius: 24,
    padding: 24,
    marginHorizontal: 20,
    marginVertical: 8,
    shadowColor: "rgba(16,39,30,0.12)",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.14,
    shadowRadius: 20,
    elevation: 8,
    borderWidth: 1,
    borderColor: palette.border,
  },

  // ✅ Titre de section avec style épuré et élégant
  sectionHeader: {
    fontSize: 20,
    fontWeight: '700',
    paddingTop: 20,
    paddingBottom: 12,
    marginHorizontal: 20,
    color: palette.text,
    letterSpacing: -0.3,
  },

  // ✅ Texte d'horaire avec style épuré
  timeText: {
    fontSize: 14,
    fontWeight: '600',
    marginHorizontal: 20,
    color: palette.secondary,
    letterSpacing: 0.1,
  },

  // ✅ Badge de statut avec style épuré et élégant
  statusBadge: {
    backgroundColor: "rgba(10,127,89,0.12)",
    color: palette.accent,
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    fontSize: 13,
    fontWeight: '700',
    alignSelf: 'flex-start',
    marginTop: 12,
    borderWidth: 1,
    borderColor: "rgba(10,127,89,0.2)",
    letterSpacing: 0.2,
  },

  // ✅ Texte principal du trajet avec typographie élégante
  routeText: {
    fontSize: 16,
    fontWeight: '600',
    color: palette.text,
    marginBottom: 10,
    lineHeight: 24,
    letterSpacing: -0.2,
  },
  
  // ✅ Texte d'état avec style épuré
  statusText: {
    fontSize: 14,
    fontWeight: '600',
    marginTop: 8,
    marginLeft: 0,
    color: palette.secondary,
    letterSpacing: 0.1,
  },
  
  // ✅ Texte affiché quand il n'y a aucune course (style épuré)
  emptyText: {
    marginTop: 24,
    marginHorizontal: 20,
    color: palette.secondary,
    fontSize: 15,
    textAlign: 'center',
    lineHeight: 22,
  },

  // ✅ Texte secondaire dans une carte (adresse) avec style élégant
  routeSection: {
    marginLeft: 0,
    marginBottom: 8,
    marginTop: 4,
    fontSize: 15,
    color: palette.text,
    lineHeight: 22,
  },

  // ✅ Variante de timeText pour version plus visible (style épuré)
  timeEnhanced: {
    fontSize: 15,
    fontWeight: '600',
    marginTop: 8,
    marginLeft: 0,
    color: palette.text,
    letterSpacing: 0.1,
  },


});


