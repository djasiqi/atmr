/**
 * @deprecated Conservé pour rétrocompat — l'implémentation vit désormais dans
 * `services/batteryOptimization.ts` (UX guidée pour Samsung One UI / Doze).
 */
export {
  checkBatteryOptimizationStatus,
  openBatteryOptimizationSettings,
  openBatteryOptimizationSettingsScreen,
  openDriverBatteryUnrestrictedSettings,
  requestIgnoreBatteryOptimizations,
} from "./services/batteryOptimization";
