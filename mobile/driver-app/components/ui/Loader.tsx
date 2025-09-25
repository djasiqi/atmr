import React from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { useColorScheme } from '@/hooks/useColorScheme';

export const Loader: React.FC = () => {
  const colorScheme = useColorScheme();

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color={colorScheme === 'dark' ? '#ffffff' : '#000000'} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});
