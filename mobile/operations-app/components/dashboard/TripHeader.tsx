// components/dashboard/TripHeader.tsx
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

// ✅ Palette épurée et élégante (cohérente avec le login et mission)
const palette = {
  background: "#F5F7F6",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
};

export default function TripHeader({ date }: { date?: string | null }) {
  // ✅ Sécurité : garantir que date est toujours une string valide
  const safeDate = typeof date === "string" ? date : String(date ?? "");

  // ✅ Déterminer le titre selon l'heure
  const currentHour = new Date().getHours();
  const title = currentHour >= 19
    ? "Vos courses (aujourd'hui et demain)"
    : "Vos courses du jour";

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.subtitle}>{safeDate}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  // ✅ Container avec style épuré et élégant
  container: {
    width: '100%',
    paddingHorizontal: 28,
    paddingTop: 32,
    paddingBottom: 24,
    backgroundColor: palette.background,
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },
  // ✅ Titre avec typographie élégante
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: palette.text,
    marginBottom: 8,
    letterSpacing: -0.5,
  },
  // ✅ Sous-titre avec style épuré
  subtitle: {
    fontSize: 15,
    color: palette.secondary,
    marginBottom: 0,
    lineHeight: 22,
  },
});
