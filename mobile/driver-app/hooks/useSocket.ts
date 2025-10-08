import { useEffect, useRef, useState } from "react";
import { connectSocket, getSocket } from "@/services/socket";
import * as Notifications from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import axios from "axios";
import api from "@/services/api";
import type { Socket } from "socket.io-client";

const refreshAccessToken = async (): Promise<string | null> => {
  const refreshToken = await AsyncStorage.getItem("refresh_token");
  if (!refreshToken) return null;

  try {
    const url = `${api.defaults.baseURL}/auth/refresh-token`; // baseURL DOIT contenir /api
    const response = await axios.post(url, null, {
      headers: { Authorization: `Bearer ${refreshToken}` },
    });

    const newToken =
      (response.data && (response.data.access_token || response.data.token)) ||
      null;

    if (newToken) {
      await AsyncStorage.setItem("token", newToken);
      return newToken;
    }
    return null;
  } catch (err) {
    console.warn("❌ Erreur lors du refresh token :", err);
    return null;
  }
};

export const useSocket = (
  onNewBooking?: (data: any) => void,
  onTeamMessage?: (msg: any) => void
): Socket | null => {
  const [socketInstance, setSocketInstance] = useState<Socket | null>(null);
  const isMountedRef = useRef(true);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef<number>(2000); // 2s -> 4s -> 8s ... max 30s
  const reconnectingRef = useRef<Promise<void> | null>(null);

  useEffect(() => {
    let activeSocket: Socket | null = null;
    isMountedRef.current = true;

    // ✅ (Re)bind propre des handlers pour un socket donné
    const bindHandlers = (s: Socket) => {
      // Toujours nettoyer avant de binder pour éviter les doublons
      s.off("connect");
      s.off("disconnect");
      s.off("connect_error");
      s.off("reconnect");
      s.off("new_booking");
      s.off("team_chat_message");
      s.off("error");
      s.off("unauthorized");

      s.on("connect", () => {
        console.log("✅ WebSocket connecté au serveur");
        // reset backoff
        backoffRef.current = 2000;
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        s.emit("join_driver_room");
      });

      s.on("disconnect", () => {
        console.log("⚠️ WebSocket déconnecté");
      });

      s.on("connect_error", (err) => {
        console.warn("❗ Erreur de connexion WebSocket :", err?.message || err);
        scheduleReconnection(s);
      });

      s.on("reconnect", () => {
        console.log("🔄 Reconnexion WebSocket");
        s.emit("join_driver_room");
      });

      s.on("new_booking", async (data: any) => {
        console.log("📦 Nouvelle mission reçue :", data);
        try {
          await Notifications.scheduleNotificationAsync({
            content: {
              title: "🚗 Nouvelle mission",
              body: `${data.pickup_location} → ${data.dropoff_location}`,
              sound: "default",
            },
            trigger: null,
          });
        } catch (err) {
          console.warn("⚠️ Erreur notification :", err);
        }
        onNewBooking?.(data);
      });

      s.on("team_chat_message", (message: any) => {
        console.log("💬 Message reçu :", message);
        onTeamMessage?.(message);
      });

      s.on("error", (data: any) => {
        console.error("❌ Erreur Socket.IO:", data);
      });

      s.on("unauthorized", async (data: any) => {
        console.error("❌ Non autorisé:", data);
        scheduleReconnection(s);
      });
    };

    // ✅ Reconnexion avec backoff exponentiel et refresh token
    const scheduleReconnection = (current: Socket) => {
      if (reconnectTimerRef.current || !isMountedRef.current) return;

      const delay = Math.min(backoffRef.current, 30000); // cap 30s
      console.log(`⏳ Reconnexion planifiée dans ${Math.round(delay / 1000)}s`);

      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        if (!isMountedRef.current) return;

        // Dédupliquer
        if (!reconnectingRef.current) {
          reconnectingRef.current = (async () => {
            try {
              const newToken = await refreshAccessToken();
              if (!newToken) {
                // pas de refresh possible → on laissera Socket.IO tenter ses reconnections auto
                backoffRef.current = Math.min(backoffRef.current * 2, 30000);
                return;
              }
              try {
                current.disconnect();
              } catch {}
              const fresh = await connectSocket(newToken);
              if (fresh && isMountedRef.current) {
                activeSocket = fresh;
                setSocketInstance(fresh);
                bindHandlers(fresh);
                console.log("🔐 Socket recréée avec nouveau token");
              }
              // reset backoff si succès
              backoffRef.current = 2000;
            } catch (e) {
              console.warn("♻️ Reconnexion échouée, on re-tentera.", e);
              backoffRef.current = Math.min(backoffRef.current * 2, 30000);
            } finally {
              reconnectingRef.current = null;
            }
          })();
        }
      }, delay);
    };

    const initializeSocket = async () => {
      let token = await AsyncStorage.getItem("token");

      if (!token) {
        token = await refreshAccessToken();
      }

      if (!token) {
        console.warn("❌ Aucun token disponible pour le WebSocket.");
        return;
      }

      activeSocket = await connectSocket(token);
      if (!activeSocket) return;

      setSocketInstance(activeSocket);

      bindHandlers(activeSocket);
    };

    initializeSocket();

    return () => {
      isMountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      try {
        const s = getSocket();
        s?.off(); // retire tous les listeners
      } catch {}
    };
  }, []);

  return socketInstance;
};
