// src/config/env.ts
import Constants from 'expo-constants';

const get = (name: string) =>
  process.env[name as keyof typeof process.env] ??
  (Constants.expoConfig?.extra as any)?.[name];

const required = (name: string) => {
  const val = get(name);
  if (val && String(val).trim().length > 0) return String(val);
  const msg = `[ENV] Missing ${name}. Ajoute ${name} dans .env.local (ou dans EAS Secrets).`;
  if (__DEV__) throw new Error(msg);
  console.warn(msg);
  return '';
};

export const GOOGLE_API_KEY = required('EXPO_PUBLIC_GOOGLE_API_KEY');            // Directions API (REST)
export const ANDROID_MAPS_API_KEY = required('EXPO_PUBLIC_ANDROID_MAPS_API_KEY'); // Maps SDK Android (natif)
