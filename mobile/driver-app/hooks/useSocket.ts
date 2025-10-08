// hooks/useSocket.ts
import { useEffect, useRef, useState } from "react";
import { connectSocket, getSocket } from "@/services/socket";
import * as Notifications from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import type { Socket } from "socket.io-client";

const TOKEN_KEY = "token";

export const useSocket = (
  onNewBooking?: (data: any) => void,
  onTeamMessage?: (msg: any) => void
): Socket | null => {
  const [socketInstance, setSocketInstance] = useState<Socket | null>(null);
  const isMountedRef = useRef(true);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef<number>(2000); // 2s -> 4s -> 8s ... (max 30s)

  useEffect(() => {
    isMountedRef.current = true;

    const bindHandlers = (s: Socket) => {
      // Nettoyage pour éviter les doublons
      s.off("connect");
      s.off("disconnect");
      s.off("connect_error");
      s.off("reconnect");
      s.off("new_booking");
      s.off("team_chat_message");
      s.off("error");
      s.off("unauthorized");

      s.on("connect", () => {
        console.log("✅ WebSocket connecté");
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

      s.on("connect_error", (err: any) => {
        console.warn("❗ socket connect_error:", err?.message || err);
        scheduleReconnection();
      });

      s.on("reconnect", () => {
        console.log("🔄 Reconnexion WebSocket");
        s.emit("join_driver_room");
      });

      s.on("new_booking", async (data: any) => {
        console.log("📦 Nouvelle mission :", data);
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
        console.log("💬 Message équipe :", message);
        onTeamMessage?.(message);
      });

      s.on("error", (data: any) => {
        console.error("❌ Erreur Socket.IO:", data);
      });

      // Si le serveur nous dit "unauthorized" → on ne tente PAS de refresh
      s.on("unauthorized", async (data: any) => {
        console.error("❌ Socket unauthorized:", data);
        // Option : purger le token si tu veux forcer un relogin
        // await AsyncStorage.removeItem(TOKEN_KEY);
        scheduleReconnection();
      });
    };

    const scheduleReconnection = () => {
      if (reconnectTimerRef.current || !isMountedRef.current) return;

      const delay = Math.min(backoffRef.current, 30000);
      console.log(`⏳ Reconnexion socket dans ${Math.round(delay / 1000)}s`);

      reconnectTimerRef.current = setTimeout(async () => {
        reconnectTimerRef.current = null;
        if (!isMountedRef.current) return;

        const token = await AsyncStorage.getItem(TOKEN_KEY);
        if (!token) {
          console.warn("🔒 Aucun token — arrêt des tentatives socket.");
          return;
        }

        try {
          const fresh = await connectSocket(token).catch(() => null);
          if (fresh && isMountedRef.current) {
            setSocketInstance(fresh);
            bindHandlers(fresh);
            backoffRef.current = 2000; // reset si succès
          }
        } catch (e) {
          console.warn("♻️ Reconnexion échouée, on re-tentera.", e);
          backoffRef.current = Math.min(backoffRef.current * 2, 30000);
          scheduleReconnection();
        }
      }, delay);
    };

    // Initialisation
    (async () => {
      const token = await AsyncStorage.getItem(TOKEN_KEY);
      if (!token) {
        console.warn("🔒 Aucun token — socket non initialisé.");
        return;
      }
      try {
        const s = await connectSocket(token).catch(() => null);
        if (!s || !isMountedRef.current) {
          // échec initial → planifie une reconnexion
          scheduleReconnection();
          return;
        }
        setSocketInstance(s);
        bindHandlers(s);
      } catch {
        // fallback: planifier une reconnexion
        scheduleReconnection();
      }
    })();

    // Cleanup
    return () => {
      isMountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      try {
        const s = getSocket();
        s?.off();
        s?.disconnect();
      } catch {}
    };
  }, []);

  return socketInstance;
};
