import { StyleSheet } from "react-native";

// ✅ Palette épurée et élégante (cohérente avec le login)
const palette = {
  background: "#F5F7F6",
  card: "#FFFFFF",
  text: "#15362B",
  secondary: "#5F7369",
  accent: "#0A7F59",
  border: "rgba(15,54,43,0.08)",
  placeholder: "#91A59D",
};

export const styles = StyleSheet.create({
  // ✅ Container avec style épuré et élégant (inspiré du login)
  containerEnhanced: {
    backgroundColor: palette.card,
    borderRadius: 24,
    padding: 28,
    marginHorizontal: 20,
    marginVertical: 16,
    marginBottom: 75, // Marge supplémentaire pour éviter que la barre d'onglets ne cache le contenu
    shadowColor: "rgba(16,39,30,0.12)",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.14,
    shadowRadius: 24,
    elevation: 8,
    borderWidth: 1,
    borderColor: palette.border,
  },

  // ✅ Header row avec style épuré
  headerRowEnhanced: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 20,
    marginTop: 0,
  },

  // ✅ Nom du client avec typographie élégante
  clientName: {
    fontWeight: "700",
    fontSize: 22,
    color: palette.text,
    flex: 1,
    marginRight: 12,
    letterSpacing: -0.3,
  },

  // ✅ Badge de statut avec style épuré
  statusBadgeContainer: {
    backgroundColor: "rgba(10,127,89,0.12)",
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    minWidth: 90,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(10,127,89,0.2)",
  },

  statusBadgeText: {
    color: palette.accent,
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.2,
  },

  departRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 2,
    marginTop: 2,
    gap: 8,
  },

  // AJOUTE BIEN rowBetween ICI ⬇️
  rowBetween: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 2,
    marginBottom: 0,
  },

  routeSection: {
    marginTop: 8,
    marginBottom: 8,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: "#E0E0E0",
  },

  // ✅ Info avec style épuré
  infoEnhanced: {
    fontSize: 13,
    fontWeight: "600",
    color: palette.secondary,
    marginTop: 4,
    letterSpacing: 0.2,
  },

  // ✅ Texte de détail avec style élégant
  detailText: {
    fontSize: 15,
    color: palette.text,
    marginLeft: 0,
    marginBottom: 6,
    marginTop: 4,
    flexShrink: 1,
    lineHeight: 22,
  },

  timeRow: {
    flexDirection: "row",
    alignItems: "center",
    marginLeft: 12,
    minWidth: 60,
  },

  // ✅ Heure avec style épuré
  timeEnhanced: {
    fontWeight: "600",
    fontSize: 15,
    color: palette.text,
    marginLeft: 4,
    letterSpacing: 0.1,
  },

  // ✅ Section métadonnées avec espacement élégant
  metaInfoSection: {
    marginTop: 8,
    marginBottom: 12,
    gap: 10,
  },

  // ✅ Section informations médicales avec style épuré
  medicalInfoSection: {
    backgroundColor: "rgba(10,127,89,0.06)",
    borderLeftWidth: 3,
    borderLeftColor: palette.accent,
    borderRadius: 14,
    padding: 16,
    marginTop: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: palette.border,
  },

  medicalTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: palette.text,
    marginBottom: 8,
    letterSpacing: 0.2,
  },

  medicalDetail: {
    fontSize: 14,
    color: palette.secondary,
    marginLeft: 0,
    marginBottom: 4,
    lineHeight: 20,
  },

  // ✅ Section chaise roulante avec style épuré
  wheelchairSection: {
    backgroundColor: "rgba(255,193,7,0.08)",
    borderLeftWidth: 3,
    borderLeftColor: "#FFC107",
    borderRadius: 14,
    padding: 16,
    marginTop: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "rgba(255,193,7,0.15)",
  },

  wheelchairAlert: {
    fontSize: 14,
    fontWeight: "700",
    color: "#8B6914",
    marginBottom: 4,
    letterSpacing: 0.1,
  },

  // ✅ Notes avec style épuré
  notesEnhanced: {
    fontSize: 14,
    color: palette.secondary,
    fontStyle: "italic",
    marginTop: 8,
    lineHeight: 20,
  },

  // ✅ Actions row avec style épuré et élégant
  actionsRowEnhanced: {
    flexDirection: "row",
    flexWrap: "nowrap",
    justifyContent: "space-between",
    alignItems: "stretch",
    marginTop: 20,
    gap: 10,
  },

  // ✅ Bouton d'action avec style élégant (inspiré du login)
  actionItemEnhanced: {
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.accent,
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 12,
    flex: 1,
    flexBasis: 0,
    flexGrow: 1,
    flexShrink: 1,
    marginVertical: 0,
    marginHorizontal: 0,
    shadowColor: palette.accent,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.24,
    shadowRadius: 12,
    elevation: 6,
  },

  // ✅ Label d'action avec typographie élégante
  actionLabel: {
    fontSize: 12,
    color: "#FFFFFF",
    marginTop: 6,
    textAlign: "center",
    fontWeight: "600",
    letterSpacing: 0.3,
  },
});
