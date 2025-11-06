// src/styles/loginStyles.ts
import { StyleSheet } from "react-native";

export const loginStyles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },

  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingVertical: 16,
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },

  header: {
    marginBottom: 32,
    alignItems: "center",
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
    color: "#104F55",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    fontWeight: "500",
    color: "#666666",
  },

  form: {
    width: "100%",
  },

  input: {
    height: 48,
    width: "100%",
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 8,
    paddingHorizontal: 16,
    fontSize: 16,
    color: "#000000",
    backgroundColor: "#FFFFFF",
    marginBottom: 16,
  },

  // === container du mot de passe + œil ===
  passwordContainer: {
    position: "relative",
    width: "100%",
    marginBottom: 16,
  },
  inputPassword: {
    height: 48,
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 8,
    paddingHorizontal: 16,
    fontSize: 16,
    color: "#000000",
    backgroundColor: "#FFFFFF",
  },
  eyeButton: {
    position: "absolute",
    right: 8,
    top: 0,
    bottom: 15,
    justifyContent: "center",
    paddingHorizontal: 8,
  },
  eyeIcon: {
    fontSize: 20,
    color: "#666666",
  },

  forgotContainer: {
    alignSelf: "flex-end",
    marginBottom: 24,
  },
  forgotText: {
    fontSize: 14,
    fontWeight: "500",
    color: "#0A7EA4",
  },

  loginButton: {
    backgroundColor: "#00796B",
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: "center",
    shadowColor: "#00796B",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 4,
    elevation: 2,
  },
  loginButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  secondaryButton: {
    marginTop: 20,
    alignItems: "center",
    paddingVertical: 12,
  },
  secondaryButtonText: {
    color: "#0A7EA4",
    fontSize: 15,
    fontWeight: "600",
    textDecorationLine: "underline",
  },
});
