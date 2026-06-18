import { Platform } from "react-native";
import * as Haptics from "expo-haptics";
import { getExpoNotificationsModule } from "../../core/notifications/expoNotificationsCompat";

const COOLDOWN_MS = 2500;
let lastPlayedAt = 0;

export function isUrgentHubMessage(payload: Record<string, unknown>): boolean {
  const priority = String(payload.priority ?? "").toLowerCase();
  if (priority === "urgent") return true;
  const messageType = String(payload.message_type ?? "").toLowerCase();
  if (messageType === "system") {
    const content = String(payload.content ?? "");
    if (content.startsWith("⚠")) return true;
  }
  const alertType = String(payload.alert_type ?? "");
  return alertType.startsWith("driver_hub_");
}

function playWebUrgentBeep(): void {
  if (typeof globalThis === "undefined") return;
  const win = globalThis as typeof globalThis & {
    AudioContext?: typeof AudioContext;
    webkitAudioContext?: typeof AudioContext;
  };
  const Ctx = win.AudioContext ?? win.webkitAudioContext;
  if (!Ctx) return;
  try {
    const ctx = new Ctx();
    const tones: { freq: number; start: number; dur: number }[] = [
      { freq: 880, start: 0, dur: 0.14 },
      { freq: 880, start: 0.18, dur: 0.14 },
      { freq: 1175, start: 0.36, dur: 0.22 },
    ];
    for (const tone of tones) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.value = tone.freq;
      gain.gain.value = 0.22;
      osc.connect(gain);
      gain.connect(ctx.destination);
      const t0 = ctx.currentTime + tone.start;
      osc.start(t0);
      osc.stop(t0 + tone.dur);
    }
    window.setTimeout(() => {
      void ctx.close().catch(() => undefined);
    }, 900);
  } catch {
    /* ignore autoplay / audio restrictions */
  }
}

async function playNativeUrgentAlert(): Promise<void> {
  await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => undefined);
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return;
  await Notifications.scheduleNotificationAsync({
    content: {
      title: "🚨 Alerte urgente",
      body: "Signalement chauffeur",
      sound: "default",
      priority: Notifications.AndroidNotificationPriority.MAX,
      data: { type: "urgent_alert" },
    },
    trigger: null,
  }).catch(() => undefined);
}

/** Alerte sonore + haptique pour signalement urgence (anti-spam 2,5 s). */
export async function playUrgentAlertSound(): Promise<void> {
  const now = Date.now();
  if (now - lastPlayedAt < COOLDOWN_MS) return;
  lastPlayedAt = now;

  if (Platform.OS === "web") {
    playWebUrgentBeep();
    return;
  }
  await playNativeUrgentAlert();
}
