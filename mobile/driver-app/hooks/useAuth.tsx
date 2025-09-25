import React, {
  useState,
  useEffect,
  useCallback,
  createContext,
  useContext,
  ReactNode,
} from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  loginDriver,
  fetchDriverProfile,
  AuthResponse,
  Driver,
} from "@/services/api";


// --- Clef de stockage unique pour le token (alignée backend/front) ---
const TOKEN_KEY = "token";

interface AuthContextType {
  driver: Driver | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  loading: boolean;
  isAuthenticated: boolean;
  refreshProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [driver, setDriver] = useState<Driver | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Persistance centralisée du token
  const setTokenSafe = useCallback(async (value: string | null) => {
    setToken(value);
    if (value) {
      await AsyncStorage.setItem(TOKEN_KEY, value);
    } else {
      await AsyncStorage.removeItem(TOKEN_KEY);
    }
  }, []);

  const loadDriver = useCallback(async () => {
    setLoading(true);
    try {
      const storedToken = await AsyncStorage.getItem(TOKEN_KEY);
      if (storedToken) {
        setToken(storedToken);
        try {
          const driverProfile = await fetchDriverProfile();
          // Log utile en dev; silencieux en prod si besoin
          console.log(
            `✅ Chauffeur: ${driverProfile.first_name} ${driverProfile.last_name} (ID: ${driverProfile.id})`
          );
          setDriver(driverProfile);
        } catch (e) {
          console.warn("❌ Token invalide ou profil inaccessible :", e);
          await setTokenSafe(null);
          setDriver(null);
        }
      } else {
        // Pas de token stocké
        setDriver(null);
      }
    } catch (error) {
      console.error("❌ Erreur chargement profil chauffeur :", error);
      await setTokenSafe(null);
      setDriver(null);
    } finally {
      setLoading(false);
    }
  }, [setTokenSafe]);

  useEffect(() => {
    loadDriver();
  }, [loadDriver]);

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true);
    try {
      const response: AuthResponse = await loginDriver(email, password);
      await setTokenSafe(response.token);
      const driverProfile = await fetchDriverProfile();
      setDriver(driverProfile);
    } catch (error) {
      console.error("Erreur lors du login :", error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [setTokenSafe]);

  const logout = useCallback(async () => {
    await setTokenSafe(null);
    setDriver(null);
  }, [setTokenSafe]);

  const refreshProfile = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const driverProfile = await fetchDriverProfile();
      setDriver(driverProfile);
    } catch (error) {
      console.error("Erreur lors du rafraîchissement du profil :", error);
    } finally {
      setLoading(false);
    }
  }, [token]);

  const contextValue: AuthContextType = {
    driver,
    token,
    login,
    logout,
    loading,
    isAuthenticated: Boolean(token && driver),
    refreshProfile,
  };

  return (
    <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth doit être utilisé au sein d’un AuthProvider");
  }
  return context;
};
