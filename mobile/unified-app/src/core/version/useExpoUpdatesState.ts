import { useUpdates } from "expo-updates";

/** Hook expo-updates — import nommé requis (Updates.useUpdates n'existe pas). */
export function useExpoUpdatesState() {
  return useUpdates();
}
