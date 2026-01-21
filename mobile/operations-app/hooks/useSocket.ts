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
  console.log("🔵 [useSocket] Hook exécuté (mount ou update)", {
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
    console.log("🔵 [useSocket] useEffect START", {
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
        console.log(JSON.stringify({
          event: "socket_connect",
          timestamp: new Date().toISOString()
        }));
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
          console.error("❌ [useSocket] Erreur resync queue GPS:", error);
        }
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
          const bookingId = data?.booking_id ?? data?.id;
          const pickup = data?.pickup_address ?? data?.pickup_location;
          const dropoff = data?.dropoff_address ?? data?.dropoff_location;
          const route =
            pickup && dropoff ? `${pickup} → ${dropoff}` : pickup || dropoff || "";

          await Notifications.scheduleNotificationAsync({
            content: {
              title: "Nouvelle course assignée",
              body: bookingId
                ? `Vous êtes assigné à la course #${bookingId}${route ? ` — ${route}` : ""}.`
                : `Vous avez une nouvelle course assignée${route ? ` — ${route}` : ""}.`,
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
          const bookingId = data?.booking_id ?? data?.id;
          const status = String(data?.status || "").toLowerCase();
          const statusLabelMap: Record<string, string> = {
            assigned: "assignée",
            en_route: "en route",
            in_progress: "à bord",
            completed: "terminée",
            return_completed: "retour terminé",
            canceled: "annulée",
            cancelled: "annulée",
          };
          const statusLabel = statusLabelMap[status];

          const changes = data?.changes;
          const parts: string[] = [];
          const add = (s?: string) => {
            if (s) parts.push(s);
          };

          const fmtHHmm = (v: any): string | null => {
            if (!v) return null;
            const s = String(v);
            if (s.includes("T") && s.length >= 16) {
              const hhmm = s.replace("Z", "").slice(11, 16);
              return hhmm.length === 5 ? hhmm : null;
            }
            return null;
          };

          const short = (v: any, maxLen = 32): string | null => {
            if (v == null) return null;
            const s = String(v).replace(/\s+/g, " ").trim();
            if (!s) return null;
            return s.length > maxLen ? `${s.slice(0, maxLen - 1)}…` : s;
          };

          const timeFrom = changes?.scheduled_time?.from;
          const timeTo = changes?.scheduled_time?.to;
          const hhmmFrom = fmtHHmm(timeFrom);
          const hhmmTo = fmtHHmm(timeTo);
          if (hhmmFrom && hhmmTo && hhmmFrom !== hhmmTo) {
            add(`Horaire : ${hhmmFrom} → ${hhmmTo}`);
          }

          const pFrom = short(changes?.pickup_location?.from);
          const pTo = short(changes?.pickup_location?.to);
          if (pFrom && pTo && pFrom !== pTo) add(`Départ : ${pFrom} → ${pTo}`);
          else if (pTo && !pFrom) add(`Départ : ${pTo}`);

          const dFrom = short(changes?.dropoff_location?.from);
          const dTo = short(changes?.dropoff_location?.to);
          if (dFrom && dTo && dFrom !== dTo) add(`Destination : ${dFrom} → ${dTo}`);
          else if (dTo && !dFrom) add(`Destination : ${dTo}`);

          if (changes?.notes) add("Info : mise à jour");

          // ✅ Pro: limiter à 2 changements + "+N autres modifications"
          const maxItems = 2;
          const head = parts.slice(0, maxItems);
          const remaining = parts.length - head.length;
          const summary =
            head.join(" • ") +
            (remaining > 0
              ? remaining === 1
                ? " • +1 autre modification"
                : ` • +${remaining} autres modifications`
              : "");

          await Notifications.scheduleNotificationAsync({
            content: {
              title: "Course mise à jour",
              body: bookingId
                ? `Course #${bookingId} — ${summary || "mise à jour"}${statusLabel ? ` (statut : ${statusLabel})` : ""}.`
                : `Une course a été ${summary || "mise à jour"}${statusLabel ? ` (statut : ${statusLabel})` : ""}.`,
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

      // ✅ Mission retirée (réassignée) → notifier + inviter à rafraîchir
      s.on("booking_reassigned", async (data: any) => {
        console.log(JSON.stringify({
          event: "booking_reassigned",
          booking_id: data?.booking_id,
          new_driver_id: data?.new_driver_id,
          timestamp: new Date().toISOString()
        }));
        try {
          const bookingId = data?.booking_id ?? data?.id;
          await Notifications.scheduleNotificationAsync({
            content: {
              title: "Course réassignée",
              body: bookingId
                ? `La course #${bookingId} a été réassignée. Vos courses vont être mises à jour.`
                : "Une course a été réassignée. Vos courses vont être mises à jour.",
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

      // ✅ Cooldown minimum de 5s entre tentatives de reconnexion
      const now = Date.now();
      const timeSinceLastAttempt = now - lastReconnectAttemptRef.current;
      const cooldownMs = 5000;
      
      if (timeSinceLastAttempt < cooldownMs) {
        const waitTime = cooldownMs - timeSinceLastAttempt;
        console.log(JSON.stringify({
          event: "socket_reconnect_cooldown",
          wait_ms: waitTime,
          timestamp: new Date().toISOString()
        }));
        reconnectTimerRef.current = setTimeout(() => {
          reconnectTimerRef.current = null;
          scheduleReconnection();
        }, waitTime);
        return;
      }

      const delay = Math.min(backoffRef.current, 60000); // ✅ Augmenté max de 30s à 60s
      console.log(JSON.stringify({
        event: "socket_reconnect_scheduled",
        delay_ms: delay,
        timestamp: new Date().toISOString()
      }));

      reconnectTimerRef.current = setTimeout(async () => {
        reconnectTimerRef.current = null;
        lastReconnectAttemptRef.current = Date.now(); // ✅ Enregistrer la tentative
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
            backoffRef.current = 5000; // ✅ Reset à 5s si succès (au lieu de 2s)
          }
        } catch (e) {
          console.warn(JSON.stringify({
            event: "socket_reconnect_failed",
            error: e instanceof Error ? e.message : String(e),
            next_attempt_ms: Math.min(backoffRef.current * 2, 60000),
            timestamp: new Date().toISOString()
          }));
          backoffRef.current = Math.min(backoffRef.current * 2, 60000); // ✅ Max 60s (au lieu de 30s)
          scheduleReconnection();
        }
      }, delay);
    };

    // Initialisation
    (async () => {
      // ✅ FIX: Utiliser secureStorage.getAccessToken() au lieu d'AsyncStorage
      const token = await secureStorage.getAccessToken();
      console.log("[useSocket] 🔑 Token check:", {
        hasToken: !!token,
        tokenLength: token?.length || 0,
        timestamp: new Date().toISOString()
      });
      
      if (!token) {
        console.warn(JSON.stringify({
          event: "socket_init_aborted",
          reason: "no_token",
          timestamp: new Date().toISOString()
        }));
        return;
      }
      
      console.log("[useSocket] 🔌 Attempting to connect socket...");
      try {
        const s = await connectSocket(token).catch((err) => {
          console.error("[useSocket] ❌ connectSocket failed:", err);
          return null;
        });
        
        console.log("[useSocket] Socket connection result:", {
          success: !!s,
          connected: s?.connected,
          id: s?.id,
        });
        
        if (!s || !isMountedRef.current) {
          // échec initial → planifie une reconnexion
          console.warn("[useSocket] ⚠️ Socket connection failed, scheduling reconnection");
          scheduleReconnection();
          return;
        }
        setSocketInstance(s);
        bindHandlers(s);
        console.log("[useSocket] ✅ Socket initialized and handlers bound");
      } catch (err) {
        // fallback: planifier une reconnexion
        console.error("[useSocket] ❌ Unexpected error:", err);
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
