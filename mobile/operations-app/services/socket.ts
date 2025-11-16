// services/socket.ts
import { io, type Socket } from "socket.io-client";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { baseURL } from "./api"; // ← réutilise l’URL déjà déduite (Expo dev/prod)

type SocketRole = "driver" | "enterprise";

// Flask-SocketIO vit à la racine (/socket.io). On enlève le suffixe /api ou /api/vX.
const SOCKET_ORIGIN = baseURL.replace(/\/api(?:\/v\d+)?$/, "");
const IS_SECURE = SOCKET_ORIGIN.startsWith("https://");

// (Optionnel) logs verbeux en dev pour socket.io
let enableSocketIODebug = () => {};
try {
  enableSocketIODebug = () =>
    require("debug").enable("socket.io-client:*,engine.io-client:*");
} catch {}

let socket: Socket | null = null;
let socketRole: SocketRole | null = null;
let connectPromise: Promise<Socket> | null = null;

const IS_DEV = __DEV__;

function buildOptions(token: string) {
  const base = {
    path: "/socket.io", // ⚠️ sans slash final
    auth: { token },
    extraHeaders: { Authorization: `Bearer ${token}` },
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    forceNew: true,
    transports: IS_SECURE ? ["websocket"] : ["websocket", "polling"],
    upgrade: true,
    rememberUpgrade: true,
    secure: IS_SECURE,
  };
  // En dev non sécurisé (HTTP), on garde polling en secours
  if (IS_DEV && !IS_SECURE) {
    return { ...base, transports: ["websocket", "polling"] };
  }
  return base;
}

export async function connectSocket(
  token: string,
  role: SocketRole = "driver"
): Promise<Socket | null> {
  if (!token) {
    console.warn("❌ Aucun token fourni à connectSocket");
    return null;
  }

  if (socket && socketRole && socketRole !== role) {
    try {
      socket.off();
      socket.disconnect();
    } catch {}
    socket = null;
    connectPromise = null;
  }

  if (socket?.connected && socketRole === role) {
    return socket;
  }
  if (connectPromise && socketRole === role) {
    return connectPromise;
  }
  if (IS_DEV) enableSocketIODebug();

  socketRole = role;

  connectPromise = new Promise<Socket>((resolve, reject) => {
    try {
      // ⚠️ Utiliser l’origine sans /api sinon 404 sur /api/socket.io
      socket = io(SOCKET_ORIGIN, buildOptions(token));

      socket.on("connect", () => {
        console.log("✅ Socket connecté, sid:", socket?.id);
        if (socketRole === "driver") {
          +joinDriverRoom().catch(() => {});
        } else if (socketRole === "enterprise") {
          +joinCompanyRoom().catch(() => {});
        }
        resolve(socket as Socket);
      });

      socket.on("disconnect", (reason) => {
        console.log("🔌 Socket déconnecté:", reason);
        connectPromise = null;
      });

      socket.on("connect_error", (err: any) => {
        console.error("❗ connect_error:", err?.message || err);
        connectPromise = null;
        reject(err);
      });

      socket.on("unauthorized", (data: any) => {
        console.error("⛔ unauthorized:", data);
      });

      socket.on("error", (e: any) => {
        console.error("❌ socket error:", e);
      });

      socket.on("connected", (data: any) => {
        console.log("🤝 handshake OK:", data);
      });
    } catch (e) {
      connectPromise = null;
      reject(e);
    }
  });

  return connectPromise;
}

export function getSocket(): Socket | null {
  return socket;
}

export function disconnectSocket() {
  try {
    socket?.off();
    socket?.disconnect();
  } finally {
    socket = null;
    socketRole = null;
    connectPromise = null;
  }
}

// Helpers côté driver
export async function joinDriverRoom() {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    console.warn("⚠️ Socket non connecté, impossible de rejoindre la room");
    return;
  }

  try {
    const idStr = await AsyncStorage.getItem("driver_id");
    const driver_id = idStr ? Number(idStr) : undefined;
    // ✅ FIX: Validation stricte du driver_id
    if (driver_id && Number.isFinite(driver_id) && driver_id > 0) {
      s?.emit("join_driver_room", { driver_id });
      console.log(`📍 join_driver_room émis avec driver_id=${driver_id}`);
    } else {
      s?.emit("join_driver_room");
      console.log("📍 join_driver_room émis sans driver_id (fallback JWT)");
    }
  } catch {
    s?.emit("join_driver_room");
    console.log(
      "📍 join_driver_room émis sans driver_id (erreur AsyncStorage)"
    );
  }
}

export async function joinCompanyRoom() {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    console.warn("⚠️ Socket non connecté, impossible de rejoindre la room entreprise");
    return;
  }
  s.emit("join_company");
  console.log("🏢 join_company émis");
}

export async function sendDriverLocation(payload: {
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp?: number | string;
}) {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    console.warn(
      "⚠️ Socket non connecté, impossible d'envoyer la localisation"
    );
    return;
  }

  try {
    const idStr = await AsyncStorage.getItem("driver_id");
    const driver_id = idStr ? Number(idStr) : undefined;
    const body =
      driver_id && Number.isFinite(driver_id) && driver_id > 0
        ? { ...payload, driver_id }
        : payload;
    s?.emit("driver_location", body);
    const has_driver_id =
      "driver_id" in (body as Record<string, unknown>) &&
      typeof (body as any).driver_id === "number";
    console.log(`📍 driver_location émis:`, {
      has_driver_id,
      lat: payload.latitude,
      lon: payload.longitude,
    });
  } catch {
    s?.emit("driver_location", payload);
    console.log("📍 driver_location émis sans driver_id (erreur)");
  }
}
