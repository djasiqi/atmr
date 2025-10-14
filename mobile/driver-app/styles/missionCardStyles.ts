import { StyleSheet } from "react-native";

export const styles = StyleSheet.create({
  containerEnhanced: {
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 18,
    marginHorizontal: 12,
    marginVertical: 12,
    marginBottom: 75, // Marge supplémentaire pour éviter que la barre d'onglets ne cache le contenu
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
    borderWidth: 1,
    borderColor: "#E0E0E0",
  },

  headerRowEnhanced: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
    marginTop: 2,
  },

  clientName: {
    fontWeight: "700",
    fontSize: 17,
    color: "#104F55",
    flex: 1,
    marginRight: 10,
  },

  statusBadgeContainer: {
    backgroundColor: "#B2DFDB",
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 13,
    minWidth: 80,
    alignItems: "center",
  },

  statusBadgeText: {
    color: "#00796B",
    fontSize: 14,
    fontWeight: "700",
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

  infoEnhanced: {
    fontSize: 14,
    fontWeight: "600",
    color: "#009688",
    marginTop: 3,
  },

  detailText: {
    fontSize: 13,
    color: "#232323",
    marginLeft: 5,
    marginBottom: 3,
    flexShrink: 1,
  },

  timeRow: {
    flexDirection: "row",
    alignItems: "center",
    marginLeft: 12,
    minWidth: 60,
  },

  timeEnhanced: {
    fontWeight: "600",
    fontSize: 15,
    color: "#666",
    marginLeft: 2,
  },

  metaInfoSection: {
    marginTop: 4,
    marginBottom: 6,
    gap: 8,
  },

  // Section informations médicales
  medicalInfoSection: {
    backgroundColor: "#E3F2FD",
    borderLeftWidth: 4,
    borderLeftColor: "#00796B",
    borderRadius: 8,
    padding: 10,
    marginTop: 8,
    marginBottom: 4,
  },

  medicalTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "#004085",
    marginBottom: 6,
  },

  medicalDetail: {
    fontSize: 13,
    color: "#004085",
    marginLeft: 4,
    marginBottom: 2,
  },

  // Section chaise roulante
  wheelchairSection: {
    backgroundColor: "#FFF3CD",
    borderLeftWidth: 4,
    borderLeftColor: "#FFC107",
    borderRadius: 8,
    padding: 10,
    marginTop: 4,
    marginBottom: 4,
  },

  wheelchairAlert: {
    fontSize: 14,
    fontWeight: "700",
    color: "#856404",
    marginBottom: 2,
  },

  notesEnhanced: {
    fontSize: 13,
    color: "#616161",
    fontStyle: "italic",
    marginTop: 4,
  },

  actionsRowEnhanced: {
    flexDirection: "row",
    flexWrap: "nowrap",
    justifyContent: "space-evenly",
    alignItems: "stretch",
    marginTop: 12,
    gap: 8,
  },

  actionItemEnhanced: {
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#00796B",
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 8,
    flex: 1,
    flexBasis: 0,
    flexGrow: 1,
    flexShrink: 1,
    marginVertical: 0,
    marginHorizontal: 0,
    shadowColor: "#00796B",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 4,
    elevation: 2,
  },

  actionLabel: {
    fontSize: 11,
    color: "#FFFFFF",
    marginTop: 4,
    textAlign: "center",
    fontWeight: "600",
  },
});
