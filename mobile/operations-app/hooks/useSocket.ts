// hooks/useSocket.ts
import { useEffect, useRef, useState } from "react";
import { connectSocket, getSocket } from "@/services/socket";
import * as Notifications from "expo-notifications";
import { secureStorage } from "@/services/storage";
import type { Socket } from "socket.io-client";

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
      s.off("booking_updated"); // ✅ FIX: Ajouter le nettoyage pour booking_updated
      s.off("team_chat_message");
      s.off("error");
      s.off("unauthorized");

      s.on("connect", () => {
        console.log(JSON.stringify({
          event: "socket_connect",
          timestamp: new Date().toISOString()
        }));
        backoffRef.current = 2000;
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        s.emit("join_driver_room");
      });

      s.on("disconnect", () => {
        console.log(JSON.stringify({
          event: "socket_disconnect",
          timestamp: new Date().toISOString()
        }));
      });

      s.on("connect_error", (err: any) => {
        console.warn(JSON.stringify({
          event: "socket_connect_error",
          error: err?.message || String(err),
          timestamp: new Date().toISOString()
        }));
        scheduleReconnection();
      });

      s.on("reconnect", (attempt) => {
        console.log(JSON.stringify({
          event: "socket_reconnect",
          attempt: attempt,
          timestamp: new Date().toISOString()
        }));
        s.emit("join_driver_room");
      });

      // ✅ Listener pong pour heartbeat applicatif
      s.on("pong", (data: any) => {
        console.log(JSON.stringify({
          event: "heartbeat_pong",
          timestamp: data?.timestamp,
          received_at: new Date().toISOString()
        }));
      });

      s.on("new_booking", async (data: any) => {
        console.log(JSON.stringify({
          event: "new_booking",
          booking_id: data?.id,
          timestamp: new Date().toISOString()
        }));
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
          console.warn(JSON.stringify({
            event: "notification_error",
            error: err instanceof Error ? err.message : String(err),
            timestamp: new Date().toISOString()
          }));
        }
        onNewBooking?.(data);
      });

      // ✅ FIX: Écouter aussi "booking_updated" pour compatibilité (même handler)
      s.on("booking_updated", async (data: any) => {
        console.log(JSON.stringify({
          event: "booking_updated",
          booking_id: data?.id,
          status: data?.status,
          timestamp: new Date().toISOString()
        }));
        try {
          await Notifications.scheduleNotificationAsync({
            content: {
              title: "🔄 Mission mise à jour",
              body: `${data.pickup_location} → ${data.dropoff_location}`,
              sound: "default",
            },
            trigger: null,
          });
        } catch (err) {
          console.warn(JSON.stringify({
            event: "notification_error",
            error: err instanceof Error ? err.message : String(err),
            timestamp: new Date().toISOString()
          }));
        }
        onNewBooking?.(data);
      });

      s.on("team_chat_message", (message: any) => {
        console.log(JSON.stringify({
          event: "team_chat_message",
          sender_id: message?.sender_id,
          timestamp: new Date().toISOString()
        }));
        onTeamMessage?.(message);
      });

      s.on("error", (data: any) => {
        console.error(JSON.stringify({
          event: "socket_error",
          error: data?.error || String(data),
          timestamp: new Date().toISOString()
        }));
      });

      // Si le serveur nous dit "unauthorized" → on ne tente PAS de refresh
      s.on("unauthorized", async (data: any) => {
        console.error(JSON.stringify({
          event: "socket_unauthorized",
          error: data?.error || String(data),
          timestamp: new Date().toISOString()
        }));
        // Option : purger le token si tu veux forcer un relogin
        // await secureStorage.removeAccessToken();
        scheduleReconnection();
      });
    };

    const scheduleReconnection = () => {
      if (reconnectTimerRef.current || !isMountedRef.current) return;

      const delay = Math.min(backoffRef.current, 30000);
      console.log(JSON.stringify({
        event: "socket_reconnect_scheduled",
        delay_ms: delay,
        timestamp: new Date().toISOString()
      }));

      reconnectTimerRef.current = setTimeout(async () => {
        reconnectTimerRef.current = null;
        if (!isMountedRef.current) return;

        // ✅ FIX: Utiliser secureStorage.getAccessToken() au lieu d'AsyncStorage
        const token = await secureStorage.getAccessToken();
        if (!token) {
          console.warn(JSON.stringify({
            event: "socket_reconnect_aborted",
            reason: "no_token",
            timestamp: new Date().toISOString()
          }));
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
          console.warn(JSON.stringify({
            event: "socket_reconnect_failed",
            error: e instanceof Error ? e.message : String(e),
            next_attempt_ms: Math.min(backoffRef.current * 2, 30000),
            timestamp: new Date().toISOString()
          }));
          backoffRef.current = Math.min(backoffRef.current * 2, 30000);
          scheduleReconnection();
        }
      }, delay);
    };

    // Initialisation
    (async () => {
      // ✅ FIX: Utiliser secureStorage.getAccessToken() au lieu d'AsyncStorage
      const token = await secureStorage.getAccessToken();
      if (!token) {
        console.warn(JSON.stringify({
          event: "socket_init_aborted",
          reason: "no_token",
          timestamp: new Date().toISOString()
        }));
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
