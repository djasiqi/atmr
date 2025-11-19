// hooks/useEnterpriseSocket.ts
import { useEffect, useRef, useState } from "react";
import { connectSocket, getSocket } from "@/services/socket";
import type { Socket } from "socket.io-client";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { ENTERPRISE_TOKEN_KEY } from "@/services/enterpriseAuth";

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
        console.log("✅ Enterprise Socket connecté");
        backoffRef.current = 2000;
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        // Rejoindre la room entreprise
        s.emit("join_company_room");
      });

      s.on("disconnect", () => {
        console.log("⚠️ Enterprise Socket déconnecté");
      });

      s.on("connect_error", (err: any) => {
        console.warn("❗ Enterprise socket connect_error:", err?.message || err);
        scheduleReconnection();
      });

      s.on("reconnect", () => {
        console.log("🔄 Reconnexion Enterprise Socket");
        s.emit("join_company_room");
      });

      s.on("team_chat_message", (message: any) => {
        console.log("💬 Message équipe (enterprise):", message);
        onTeamMessage?.(message);
      });

      s.on("typing_start", () => {
        // Géré par le composant parent
      });

      s.on("typing_stop", () => {
        // Géré par le composant parent
      });

      s.on("error", (data: any) => {
        console.error("❌ Erreur Enterprise Socket.IO:", data);
      });

      s.on("unauthorized", async (data: any) => {
        console.error("❌ Enterprise Socket unauthorized:", data);
        scheduleReconnection();
      });
    };

    const scheduleReconnection = () => {
      if (reconnectTimerRef.current || !isMountedRef.current) return;

      const delay = Math.min(backoffRef.current, 30000);
      console.log(`⏳ Reconnexion enterprise socket dans ${Math.round(delay / 1000)}s`);

      reconnectTimerRef.current = setTimeout(async () => {
        reconnectTimerRef.current = null;
        if (!isMountedRef.current) return;

        const token = await AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY);
        if (!token) {
          console.warn("🔒 Aucun token entreprise — arrêt des tentatives socket.");
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
          console.warn("♻️ Reconnexion enterprise échouée, on re-tentera.", e);
          backoffRef.current = Math.min(backoffRef.current * 2, 30000);
          scheduleReconnection();
        }
      }, delay);
    };

    // Initialisation
    (async () => {
      const token = await AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY);
      if (!token) {
        console.warn("🔒 Aucun token entreprise — socket non initialisé.");
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

    // Cleanup
    return () => {
      isMountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      // Ne pas déconnecter le socket car il peut être utilisé ailleurs
    };
  }, [onTeamMessage]);

  return socketInstance;
};

