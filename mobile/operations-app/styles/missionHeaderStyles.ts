import { StyleSheet } from 'react-native';

// ✅ Palette épurée et élégante (cohérente avec le login)
const palette = {
  background: "#F5F7F6",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
};

export const styles = StyleSheet.create({
  // ✅ Bloc principal de l'en-tête avec style épuré
  container: { 
    paddingHorizontal: 28,
    paddingTop: 32,
    paddingBottom: 24,
    backgroundColor: palette.background,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },

  // ✅ Ligne de bienvenue avec typographie élégante
  title: { 
    fontSize: 28,
    fontWeight: '700',
    color: palette.text,
    marginBottom: 8,
    letterSpacing: -0.5,
  },

  // ✅ Texte de date ou info complémentaire
  subtitle: { 
    fontSize: 15,
    color: palette.secondary,
    marginBottom: 8,
    lineHeight: 22,
  },

  // ✅ Statut du chauffeur avec style élégant
  status: { 
    fontSize: 14,
    color: palette.accent,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
});
