import messaging from '@react-native-firebase/messaging';
import notifee, { AndroidImportance } from '@notifee/react-native';

async function ensureChannel() {
  await notifee.createChannel({
    id: 'missions_v2',
    name: 'Missions',
    importance: AndroidImportance.HIGH,
    sound: 'default',
  });
}

// Android data-only messages trigger this handler when app is killed/background.
// iOS does NOT execute JS in killed state — system tray handles display natively.
messaging().setBackgroundMessageHandler(async (remoteMessage) => {
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
});

import 'expo-router/entry';
