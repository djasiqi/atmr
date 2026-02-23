// hooks/useSocket.ts
import { useEffect, useRef, useState } from "react";
import { connectSocket } from "@/services/socket";

import { secureStorage } from "@/services/storage";
import { useAuth } from "@/hooks/useAuth";
import { getLogger } from "@/utils/logger";
import type { Socket } from "socket.io-client";

const log = getLogger("SocketHook");

export const useSocket = (
  onNewBooking?: (data: any) => void,
  onTeamMessage?: (msg: any) => void
): Socket | null => {
  const { driver } = useAuth();
  const driverIdRef = useRef<number | undefined>(driver?.id);
  driverIdRef.current = driver?.id;

  log.info("hook executed", {
    timestamp: new Date().toISOString(),
    hasOnNewBooking: !!onNewBooking,
    hasOnTeamMessage: !!onTeamMessage
  });
  
  const [socketInstance, setSocketInstance] = useState<Socket | null>(null);
  const isMountedRef = useRef(true);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef<number>(5000); // ✅ Augmenté de 2s à 5s -> 10s -> 20s ... (max 60s)
  const lastReconnectAttemptRef = useRef<number>(0); // ✅ Cooldown entre reconnexions

  useEffect(() => {
    log.info("useEffect start", {
      timestamp: new Date().toISOString()
    });
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

      s.on("connect", async () => {
        log.info("socket connect", {
          event: "socket_connect",
          timestamp: new Date().toISOString()
        });
        backoffRef.current = 5000; // ✅ Reset à 5s au lieu de 2s
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        s.emit("join_driver_room");
        
        // ✅ P2-1: Resync queue GPS au reconnect
        try {
          const { syncLocationQueue } = await import("@/services/locationQueue");
          await syncLocationQueue(s);
        } catch (error) {
          log.error("resync queue gps failed", { error });
        }
      });

      s.on("disconnect", () => {
        log.info("socket disconnect", {
          event: "socket_disconnect",
          timestamp: new Date().toISOString()
        });
      });

      s.on("connect_error", (err: any) => {
        log.warn("socket connect error", {
          event: "socket_connect_error",
          error: err?.message || String(err),
          timestamp: new Date().toISOString()
        });
        scheduleReconnection();
      });

      s.on("reconnect", (attempt) => {
        log.info("socket reconnect", {
          event: "socket_reconnect",
          attempt,
          timestamp: new Date().toISOString()
        });
        s.emit("join_driver_room");
      });

      // ✅ Listener pong pour heartbeat applicatif
      s.on("pong", (data: any) => {
        log.info("heartbeat pong", {
          event: "heartbeat_pong",
          timestamp: data?.timestamp,
          received_at: new Date().toISOString()
        });
      });

      s.on("new_booking", async (data: any) => {
        log.info("new booking", {
          event: "new_booking",
          booking_id: data?.id,
          timestamp: new Date().toISOString()
        });
        // Push notification is already sent by backend — no local notification needed
        onNewBooking?.(data);
      });

      // Écouter "booking_updated" pour rafraîchir l'UI
      s.on("booking_updated", async (data: any) => {
        log.info("booking updated", {
          event: "booking_updated",
          booking_id: data?.id,
          status: data?.status,
          timestamp: new Date().toISOString()
        });
        // Push notification is already sent by backend — no local notification needed
        onNewBooking?.(data);
      });

      // Mission retirée (réassignée) → rafraîchir les courses
      s.on("booking_reassigned", async (data: any) => {
        log.info("booking reassigned", {
          event: "booking_reassigned",
          booking_id: data?.booking_id,
          new_driver_id: data?.new_driver_id,
          timestamp: new Date().toISOString()
        });
        // Push notification is already sent by backend — no local notification needed
        onNewBooking?.(data);
      });

      s.on("team_chat_message", (message: any) => {
        log.info("team chat message", {
          event: "team_chat_message",
          sender_id: message?.sender_id,
          timestamp: new Date().toISOString()
        });
        onTeamMessage?.(message);
      });

      s.on("error", (data: any) => {
        log.error("socket error", {
          event: "socket_error",
          error: data?.error || String(data),
          timestamp: new Date().toISOString()
        });
      });

      // Si le serveur nous dit "unauthorized" → on ne tente PAS de refresh
      s.on("unauthorized", async (data: any) => {
        log.error("socket unauthorized", {
          event: "socket_unauthorized",
          error: data?.error || String(data),
          timestamp: new Date().toISOString()
        });
        // Option : purger le token si tu veux forcer un relogin
        // await secureStorage.removeAccessToken();
        scheduleReconnection();
      });
    };

    const scheduleReconnection = () => {
      if (reconnectTimerRef.current || !isMountedRef.current) return;

      // ✅ Cooldown minimum de 5s entre tentatives de reconnexion
      const now = Date.now();
      const timeSinceLastAttempt = now - lastReconnectAttemptRef.current;
      const cooldownMs = 5000;
      
      if (timeSinceLastAttempt < cooldownMs) {
        const waitTime = cooldownMs - timeSinceLastAttempt;
        log.info("reconnect cooldown", {
          event: "socket_reconnect_cooldown",
          wait_ms: waitTime,
          timestamp: new Date().toISOString()
        });
        reconnectTimerRef.current = setTimeout(() => {
          reconnectTimerRef.current = null;
          scheduleReconnection();
        }, waitTime);
        return;
      }

      const delay = Math.min(backoffRef.current, 60000); // ✅ Augmenté max de 30s à 60s
      log.info("reconnect scheduled", {
        event: "socket_reconnect_scheduled",
        delay_ms: delay,
        timestamp: new Date().toISOString()
      });

      reconnectTimerRef.current = setTimeout(async () => {
        reconnectTimerRef.current = null;
        lastReconnectAttemptRef.current = Date.now(); // ✅ Enregistrer la tentative
        if (!isMountedRef.current) return;

        // ✅ FIX: Utiliser secureStorage.getAccessToken() au lieu d'AsyncStorage
        const token = await secureStorage.getAccessToken();
        if (!token) {
          log.warn("reconnect aborted (no token)", {
            event: "socket_reconnect_aborted",
            reason: "no_token",
            timestamp: new Date().toISOString()
          });
          return;
        }

        try {
          const fresh = await connectSocket(token).catch(() => null);
          if (fresh && isMountedRef.current) {
            setSocketInstance(fresh);
            bindHandlers(fresh);
            backoffRef.current = 5000; // ✅ Reset à 5s si succès (au lieu de 2s)
          }
        } catch (e) {
          log.warn("reconnect failed", {
            event: "socket_reconnect_failed",
            error: e instanceof Error ? e.message : String(e),
            next_attempt_ms: Math.min(backoffRef.current * 2, 60000),
            timestamp: new Date().toISOString()
          });
          backoffRef.current = Math.min(backoffRef.current * 2, 60000); // ✅ Max 60s (au lieu de 30s)
          scheduleReconnection();
        }
      }, delay);
    };

    // Initialisation
    (async () => {
      // ✅ FIX: Utiliser secureStorage.getAccessToken() au lieu d'AsyncStorage
      const token = await secureStorage.getAccessToken();
      log.info("token check", {
        hasToken: !!token,
        tokenLength: token?.length || 0,
        timestamp: new Date().toISOString()
      });
      
      if (!token) {
        log.warn("init aborted (no token)", {
          event: "socket_init_aborted",
          reason: "no_token",
          timestamp: new Date().toISOString()
        });
        return;
      }
      
      log.info("attempting to connect socket");
      try {
        const s = await connectSocket(token).catch((err) => {
          log.error("connectSocket failed", { err });
          return null;
        });
        
        log.info("connection result", {
          success: !!s,
          connected: s?.connected,
          id: s?.id,
        });
        
        if (!s || !isMountedRef.current) {
          // échec initial → planifie une reconnexion
          log.warn("connection failed, scheduling reconnection");
          scheduleReconnection();
          return;
        }
        setSocketInstance(s);
        bindHandlers(s);
        log.info("socket initialized and handlers bound");
      } catch (err) {
        // fallback: planifier une reconnexion
        log.error("unexpected error", { err });
        scheduleReconnection();
      }
    })();

    return () => {
      isMountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };
  }, []);

  return socketInstance;
};
