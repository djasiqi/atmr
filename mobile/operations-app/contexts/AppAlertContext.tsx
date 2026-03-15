/**
 * Contexte pour afficher des alertes modales au design plateforme (sobre, professionnel).
 * Remplace Alert.alert pour une cohérence visuelle.
 */
import React, { createContext, useCallback, useContext, useState } from "react";
import {
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

type AlertItem = {
  id: number;
  title: string;
  message: string;
  onDismiss?: () => void;
};

type AppAlertContextValue = {
  showAlert: (title: string, message: string, onDismiss?: () => void) => void;
};

const AppAlertContext = createContext<AppAlertContextValue | null>(null);

let nextId = 0;

export function AppAlertProvider({ children }: { children: React.ReactNode }) {
  const [current, setCurrent] = useState<AlertItem | null>(null);

  const showAlert = useCallback(
    (title: string, message: string, onDismiss?: () => void) => {
      const id = ++nextId;
      setCurrent({ id, title, message, onDismiss });
    },
    []
  );

  const dismiss = useCallback(() => {
    setCurrent((prev) => {
      prev?.onDismiss?.();
      return null;
    });
  }, []);

  return (
    <AppAlertContext.Provider value={{ showAlert }}>
      {children}
      {current && (
        <Modal
          visible
          transparent
          animationType="fade"
          onRequestClose={dismiss}
        >
          <Pressable style={styles.overlay} onPress={dismiss}>
            <Pressable style={styles.card} onPress={(e) => e.stopPropagation()}>
              <Text style={styles.title}>{current.title}</Text>
              <Text style={styles.message}>{current.message}</Text>
              <TouchableOpacity
                style={styles.button}
                onPress={dismiss}
                activeOpacity={0.85}
              >
                <Text style={styles.buttonText}>OK</Text>
              </TouchableOpacity>
            </Pressable>
          </Pressable>
        </Modal>
      )}
    </AppAlertContext.Provider>
  );
}

export function useAppAlert(): AppAlertContextValue {
  const ctx = useContext(AppAlertContext);
  if (!ctx) {
    return {
      showAlert: (title: string, message: string, _onDismiss?: () => void) => {
        // Fallback si hors provider (ne devrait pas arriver)
        const { Alert } = require("react-native");
        Alert.alert(title, message);
      },
    };
  }
  return ctx;
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.52)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 14,
    padding: 22,
    width: "100%",
    maxWidth: 340,
    elevation: 6,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
  },
  title: {
    fontSize: 17,
    fontWeight: "600",
    color: "#1A1A1A",
    marginBottom: 10,
    textAlign: "center",
    letterSpacing: -0.2,
  },
  message: {
    fontSize: 14,
    color: "#525252",
    lineHeight: 20,
    marginBottom: 18,
    textAlign: "center",
  },
  button: {
    backgroundColor: "#00796B",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "600",
  },
});
