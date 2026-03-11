import { getMessaging, setBackgroundMessageHandler } from '@react-native-firebase/messaging';
import notifee, { AndroidImportance } from '@notifee/react-native';

// ✅ Notifee: onBackgroundEvent DOIT être enregistré AVANT toute autre exécution.
// Sinon "no background event handler has been set" et les actions (Mission Bar) ne marchent pas en background.
// On enregistre un noop immédiatement pour éviter le warning, puis le handler complet.
notifee.onBackgroundEvent(async () => {});
try {
  const { registerNotifeeBackgroundHandler } = require('./services/missionBarBackground');
  registerNotifeeBackgroundHandler();
} catch (e) {
  if (__DEV__) {
    console.warn('[Notifee] background handler registration failed', e?.message || e);
  }
}

let channelReady = false;
async function ensureChannel() {
  if (channelReady) return;
  await notifee.createChannel({
    id: 'missions_v2',
    name: 'Missions',
    importance: AndroidImportance.HIGH,
    sound: 'default',
  });
  channelReady = true;
}

// Android data-only messages trigger this handler when app is killed/background.
// iOS does NOT execute JS in killed state — system tray handles display natively.
const messaging = getMessaging();
setBackgroundMessageHandler(messaging, async (remoteMessage) => {
  try {
    const { data } = remoteMessage;
    if (!data) return;

    if (data.type === 'silent_update') return;

    await ensureChannel();

    await notifee.displayNotification({
      title: data.title || 'Liri Opérations',
      body: data.body || '',
      data,
      android: {
        channelId: data.channelId || 'missions_v2',
        importance: AndroidImportance.HIGH,
        sound: 'default',
        pressAction: { id: 'default' },
      },
    });
  } catch {
    // Ne pas planter le background handler — Android le tuerait comme ANR
  }
});

import 'expo-router/entry';
