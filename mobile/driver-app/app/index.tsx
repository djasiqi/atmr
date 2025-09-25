// app/index.tsx
import { useEffect, useState } from 'react';
import { router } from 'expo-router';
import { View, Text } from 'react-native';

export default function IndexRedirect() {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      router.replace('/(tabs)/mission');
    }, 50); // léger délai pour laisser le layout se monter

    return () => clearTimeout(timer);
  }, []);

  return (
    <View>
      <Text>Chargement...</Text>
    </View>
  );
}
