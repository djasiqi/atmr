// components/chat/Avatar.tsx
// Composant Avatar pour afficher la photo ou les initiales (style WhatsApp)

import React from "react";
import { View, Text, Image, StyleSheet } from "react-native";

interface Props {
  photo?: string | null;
  name?: string | null;
  size?: number;
}

// Couleurs pour les avatars (générées à partir du nom)
const AVATAR_COLORS = [
  "#0A7F59", // Vert principal
  "#5F7369", // Gris-vert
  "#8B4513", // Marron
  "#4169E1", // Bleu royal
  "#DC143C", // Rouge
  "#FF6347", // Tomate
  "#32CD32", // Vert lime
  "#FFD700", // Or
  "#9370DB", // Violet
  "#20B2AA", // Turquoise
];

// Fonction pour obtenir une couleur basée sur le nom
const getColorForName = (name: string | null | undefined): string => {
  if (!name) return AVATAR_COLORS[0];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
};

// Fonction pour extraire les initiales
const getInitials = (name: string | null | undefined): string => {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) {
    // Première lettre du prénom + première lettre du nom
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  // Si un seul mot, prendre les 2 premières lettres
  return name.substring(0, 2).toUpperCase();
};

export default function Avatar({ photo, name, size = 32 }: Props) {
  const backgroundColor = getColorForName(name);
  const initials = getInitials(name);

  return (
    <View style={[styles.container, { width: size, height: size, borderRadius: size / 2 }]}>
      {photo ? (
        <Image
          source={{ uri: photo }}
          style={[styles.image, { width: size, height: size, borderRadius: size / 2 }]}
          resizeMode="cover"
        />
      ) : (
        <View style={[styles.initialsContainer, { backgroundColor, width: size, height: size, borderRadius: size / 2 }]}>
          <Text style={[styles.initials, { fontSize: size * 0.4 }]}>{initials}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    overflow: "hidden",
    backgroundColor: "#E0E0E0",
  },
  image: {
    width: "100%",
    height: "100%",
  },
  initialsContainer: {
    width: "100%",
    height: "100%",
    justifyContent: "center",
    alignItems: "center",
  },
  initials: {
    color: "#FFFFFF",
    fontWeight: "600",
    textAlign: "center",
  },
});

