// components/dashboard/TripHeader.tsx
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function TripHeader({ date }: { date: string }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Vos courses du jour</Text>
      <Text style={styles.subtitle}>{date}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%', //
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 16,
    backgroundColor: '#F7F9FB',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    color: '#004D40',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 14,
    color: '#555',
    marginBottom: 5,
  },
});
