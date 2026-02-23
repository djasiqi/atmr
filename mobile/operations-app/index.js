import messaging from '@react-native-firebase/messaging';
import notifee, { AndroidImportance } from '@notifee/react-native';

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
messaging().setBackgroundMessageHandler(async (remoteMessage) => {
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
