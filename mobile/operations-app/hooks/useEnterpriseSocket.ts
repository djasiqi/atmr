// hooks/useEnterpriseSocket.ts
import { useEffect, useRef, useState } from "react";
import { connectSocket, getSocket } from "@/services/socket";
import type { Socket } from "socket.io-client";
import { secureStorage } from "@/services/storage";
import { getLogger } from "@/utils/logger";

const log = getLogger("EntSocket");

export const useEnterpriseSocket = (
  onTeamMessage?: (msg: any) => void
): Socket | null => {
  const [socketInstance, setSocketInstance] = useState<Socket | null>(null);
  const isMountedRef = useRef(true);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef<number>(2000);

  useEffect(() => {
    isMountedRef.current = true;

    const bindHandlers = (s: Socket) => {
      s.off("connect");
      s.off("disconnect");
      s.off("connect_error");
      s.off("reconnect");
      s.off("team_chat_message");
      s.off("typing_start");
      s.off("typing_stop");
      s.off("error");
      s.off("unauthorized");

      s.on("connect", () => {
        log.success("enterprise socket connected", {});
        backoffRef.current = 2000;
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        // Rejoindre la room entreprise
        s.emit("join_company_room");
      });

      s.on("disconnect", () => {
        log.info("enterprise socket disconnected", {});
      });

      s.on("connect_error", (err: any) => {
        log.warn("enterprise socket connect_error", { message: err?.message || err });
        scheduleReconnection();
      });

      s.on("reconnect", () => {
        log.info("enterprise socket reconnected", {});
        s.emit("join_company_room");
      });

      s.on("team_chat_message", (message: any) => {
        log.info("team message received", { message });
        onTeamMessage?.(message);
      });

      s.on("typing_start", () => {
        // Géré par le composant parent
      });

      s.on("typing_stop", () => {
        // Géré par le composant parent
      });

      s.on("error", (data: any) => {
        log.error("enterprise socket error", { data });
      });

      s.on("unauthorized", async (data: any) => {
        log.error("enterprise socket unauthorized", { data });
        scheduleReconnection();
      });
    };

    const scheduleReconnection = () => {
      if (reconnectTimerRef.current || !isMountedRef.current) return;

      const delay = Math.min(backoffRef.current, 30000);
      log.info("enterprise socket reconnecting", { delaySeconds: Math.round(delay / 1000) });

      reconnectTimerRef.current = setTimeout(async () => {
        reconnectTimerRef.current = null;
        if (!isMountedRef.current) return;

        const token = await secureStorage.getEnterpriseToken();
        if (!token) {
          log.warn("no enterprise token, stopping socket attempts", {});
          return;
        }

        try {
          const fresh = await connectSocket(token, "enterprise").catch(() => null);
          if (fresh && isMountedRef.current) {
            setSocketInstance(fresh);
            bindHandlers(fresh);
            backoffRef.current = 2000;
          }
        } catch (e) {
          log.warn("enterprise reconnection failed, will retry", { error: e });
          backoffRef.current = Math.min(backoffRef.current * 2, 30000);
          scheduleReconnection();
        }
      }, delay);
    };

    // Initialisation
    (async () => {
      const token = await secureStorage.getEnterpriseToken();
      if (!token) {
        log.warn("no enterprise token, socket not initialized", {});
        return;
      }
      try {
        const s = await connectSocket(token, "enterprise").catch(() => null);
        if (!s || !isMountedRef.current) {
          scheduleReconnection();
          return;
        }
        setSocketInstance(s);
        bindHandlers(s);
      } catch {
        scheduleReconnection();
      }
    })();

    return () => {
      isMountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      // Retirer les handlers pour éviter les fuites mémoire (le socket global reste connecté)
      if (socketInstance) {
        socketInstance.off("connect");
        socketInstance.off("disconnect");
        socketInstance.off("connect_error");
        socketInstance.off("reconnect");
        socketInstance.off("team_chat_message");
        socketInstance.off("typing_start");
        socketInstance.off("typing_stop");
        socketInstance.off("error");
        socketInstance.off("unauthorized");
      }
    };
  }, [onTeamMessage]);

  return socketInstance;
};

