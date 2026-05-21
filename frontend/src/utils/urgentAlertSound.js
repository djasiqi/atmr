const COOLDOWN_MS = 2500;
let lastPlayedAt = 0;

export function isUrgentHubMessage(payload) {
  if (!payload || typeof payload !== 'object') return false;
  const priority = String(payload.priority ?? '').toLowerCase();
  if (priority === 'urgent') return true;
  const messageType = String(payload.message_type ?? '').toLowerCase();
  if (messageType === 'system') {
    const content = String(payload.content ?? '');
    if (content.startsWith('⚠')) return true;
  }
  const alertType = String(payload.alert_type ?? '');
  return alertType.startsWith('driver_hub_');
}

function playWebUrgentBeep() {
  if (typeof window === 'undefined') return;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  try {
    const ctx = new Ctx();
    const tones = [
      { freq: 880, start: 0, dur: 0.14 },
      { freq: 880, start: 0.18, dur: 0.14 },
      { freq: 1175, start: 0.36, dur: 0.22 },
    ];
    tones.forEach(({ freq, start, dur }) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.value = freq;
      gain.gain.value = 0.22;
      osc.connect(gain);
      gain.connect(ctx.destination);
      const t0 = ctx.currentTime + start;
      osc.start(t0);
      osc.stop(t0 + dur);
    });
    window.setTimeout(() => {
      ctx.close().catch(() => {});
    }, 900);
  } catch {
    /* autoplay policy */
  }
}

export function playUrgentAlertSound() {
  const now = Date.now();
  if (now - lastPlayedAt < COOLDOWN_MS) return;
  lastPlayedAt = now;
  playWebUrgentBeep();
}

export function bindUrgentAlertSoundListeners(socket) {
  if (!socket || typeof socket.on !== 'function') return;

  const onUrgentAlert = () => playUrgentAlertSound();
  const onTeamChat = (payload) => {
    if (isUrgentHubMessage(payload)) playUrgentAlertSound();
  };

  socket.off('urgent_alert', onUrgentAlert);
  socket.off('team_chat_message', onTeamChat);
  socket.on('urgent_alert', onUrgentAlert);
  socket.on('team_chat_message', onTeamChat);
}
