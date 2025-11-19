// components/dashboard/ChatHeader.tsx
import React from "react";
import { View, Text, StyleSheet } from "react-native";

// ✅ Palette épurée et élégante (cohérente avec le login et autres pages)
const palette = {
  background: "#F5F7F6",
  text: "#15362B",
  secondary: "#5F7369",
  border: "rgba(15,54,43,0.08)",
};

export default function ChatHeader() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Équipe</Text>
      <Text style={styles.subtitle}>
        Discutez avec votre équipe en temps réel
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  // ✅ Container avec style épuré et élégant
  container: {
    width: "100%",
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
    fontWeight: "700",
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

