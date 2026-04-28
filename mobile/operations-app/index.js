import './polyfills';

import { getMessaging, setBackgroundMessageHandler } from '@react-native-firebase/messaging';
import { getApps } from '@react-native-firebase/app';
import notifee, { AndroidImportance } from '@notifee/react-native';
import { DevSettings, Platform } from 'react-native';

try {
  const originalDevReload = DevSettings?.reload?.bind(DevSettings);
  if (originalDevReload) {
    DevSettings.reload = (...args) => {
      const stackText = String(new Error('devsettings reload stack').stack ?? '');
      const shouldIgnoreAutoReload =
        __DEV__ &&
        Platform.OS === 'android' &&
        (stackText.includes('registerBundleEntryPoints') ||
          stackText.includes('registerBundle') ||
          stackText.includes('index.bundle'));
      if (shouldIgnoreAutoReload) {
        return;
      }
      return originalDevReload(...args);
    };
  }
} catch {}

// ✅ Notifee: onBackgroundEvent DOIT être enregistré AVANT toute autre exécution.
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

if (Platform.OS !== 'web' && getApps().length > 0) {
  try {
    const messaging = getMessaging();
    setBackgroundMessageHandler(messaging, async (remoteMessage) => {
      try {
        const { data } = remoteMessage;
        if (!data) return;
        if (data.type === 'silent_update') return;
        await ensureChannel();
        // ID déterministe : évite doublons quand plusieurs FCM arrivent (ex: 2 DeviceTokens même appareil)
        const notifId = `mission_${data.booking_id || data.trace_id || Date.now()}_${data.type || "push"}`.replace(/\s/g, "_");
        await notifee.displayNotification({
          id: notifId,
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
  } catch (e) {
    if (__DEV__) {
      console.warn('[FCM] setBackgroundMessageHandler non enregistré', e?.message || e);
    }
  }
}

try {
  require('expo-router/entry');
} catch (e) {
  throw e;
}
