// hooks/useSocket.ts
import { useEffect, useRef, useState } from "react";
import {
  connectSocket,
  onBookingNew,
  onBookingUpdated,
  onBookingReassigned,
  onTeamChatMessage,
} from "@/services/socket";
import { secureStorage } from "@/services/storage";
import { getLogger } from "@/utils/logger";
import type { Socket } from "socket.io-client";

const log = getLogger("SocketHook");

export const useSocket = (
  onNewBooking?: (data: any) => void,
  onTeamMessage?: (msg: any) => void
): Socket | null => {
  const [socketInstance, setSocketInstance] = useState<Socket | null>(null);
  const onNewBookingRef = useRef<typeof onNewBooking>(onNewBooking);
  const onTeamMessageRef = useRef<typeof onTeamMessage>(onTeamMessage);

  useEffect(() => {
    onNewBookingRef.current = onNewBooking;
    onTeamMessageRef.current = onTeamMessage;
  }, [onNewBooking, onTeamMessage]);

  useEffect(() => {
    let mounted = true;
    const unsubscribers: Array<() => void> = [];

    (async () => {
      const token = await secureStorage.getAccessToken();
      if (!token) {
        log.warn("socket hook init aborted (no token)");
        return;
      }

      try {
        const s = await connectSocket(token, "driver");
        if (!mounted) return;
        if (!s) {
          log.warn("socket hook init failed (null socket)");
          return;
        }

        setSocketInstance(s);

        unsubscribers.push(
          onBookingNew((payload) => {
            onNewBookingRef.current?.(payload);
          })
        );
        unsubscribers.push(
          onBookingUpdated((payload) => {
            onNewBookingRef.current?.(payload);
          })
        );
        unsubscribers.push(
          onBookingReassigned((payload) => {
            onNewBookingRef.current?.(payload);
          })
        );
        unsubscribers.push(
          onTeamChatMessage((payload) => {
            onTeamMessageRef.current?.(payload);
          })
        );
      } catch (err) {
        log.error("socket hook init error", { err });
      }
    })();

    return () => {
      mounted = false;
      unsubscribers.forEach((unsubscribe) => {
        try {
          unsubscribe();
        } catch {
          // no-op
        }
      });
    };
  }, []);

  return socketInstance;
};
