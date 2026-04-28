// hooks/useEnterpriseSocket.ts
import { useEffect, useRef, useState } from "react";
import {
  connectSocket,
  disconnectSocket,
  getSocketRole,
  onTeamChatMessage,
} from "@/services/socket";
import type { Socket } from "socket.io-client";
import { secureStorage } from "@/services/storage";
import { getLogger } from "@/utils/logger";

const log = getLogger("EntSocket");

export const useEnterpriseSocket = (
  onTeamMessage?: (msg: any) => void
): Socket | null => {
  const [socketInstance, setSocketInstance] = useState<Socket | null>(null);
  const isMountedRef = useRef(true);
  const onTeamMessageRef = useRef<typeof onTeamMessage>(onTeamMessage);

  useEffect(() => {
    onTeamMessageRef.current = onTeamMessage;
  }, [onTeamMessage]);

  useEffect(() => {
    isMountedRef.current = true;
    let unsubscribeTeamMessage: (() => void) | null = null;

    // Initialisation
    (async () => {
      const token = await secureStorage.getEnterpriseToken();
      if (!token) {
        log.warn("enterprise socket init aborted (no token)");
        return;
      }
      try {
        const s = await connectSocket(token, "enterprise");
        if (!s || !isMountedRef.current) {
          return;
        }
        setSocketInstance(s);
        unsubscribeTeamMessage = onTeamChatMessage((payload) => {
          onTeamMessageRef.current?.(payload);
        });
      } catch {
        log.warn("enterprise socket init failed");
      }
    })();

    return () => {
      isMountedRef.current = false;
      unsubscribeTeamMessage?.();
      if (getSocketRole() === "enterprise") {
        disconnectSocket();
      }
    };
  }, []);

  return socketInstance;
};

