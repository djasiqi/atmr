/**
 * Chaînes fr-CH gelées pour la production (GPS, FGS, permissions OS, compliance Play).
 * Source de vérité alignée avec les modales disclosure et `ops-readiness/play-release/bg-location-justification.txt`.
 */
export const PRODUCTION_LOCALE = {
  appDisplayName: "Lirie",

  /** Notification persistante FGS Android (mission / présence). */
  fgsNotificationTitle: "Lirie est active",
  fgsNotificationBodyMission: "Mission en cours — localisation active",
  fgsNotificationBodyPresence: "Disponibilité active — localisation en cours",
  /** Notification FGS stable (présence ↔ live sans mutation). */
  fgsNotificationBodyStable: "Localisation ATMR active",

  /** Dialogues système iOS / plugin expo-location (alignés modales in-app). */
  iosLocationWhenInUse:
    "La localisation permet le suivi temps réel des missions en cours et, lorsque vous êtes en service, la visibilité sur la carte dispatch de votre entreprise.",
  iosLocationAlways:
    "La localisation en arrière-plan permet le suivi des missions lorsque l'écran est verrouillé et, lorsque vous êtes en service, la visibilité flotte pour le dispatch.",
  iosLocationAlwaysAndWhenInUse:
    "La localisation en arrière-plan permet le suivi des missions lorsque l'écran est verrouillé et, lorsque vous êtes en service, la visibilité flotte pour le dispatch.",
  androidLocationAlwaysAndWhenInUse:
    "La localisation permet le suivi opérationnel des missions en cours, y compris lorsque l'écran est verrouillé.",

  iosMicrophone:
    "Le micro sert à enregistrer des messages vocaux dans le chat.",
  iosPhotoLibrary: "L'accès aux photos permet d'envoyer des pièces jointes dans le chat.",
  iosCamera: "L'accès à la caméra permet de prendre des photos pour les missions.",

  /** Canaux notifications Android (réglages système). */
  notificationChannelGeneral: "Général",
  notificationChannelUrgent: "Urgent",
  notificationChannelMissionUpdates: "Mises à jour mission",
  notificationChannelChat: "Messages",
  notificationChannelSilentSync: "Synchronisation silencieuse",
  notificationChannelLockScreen: "Écran verrouillé",
  notificationChannelMissionActive: "Mission active",
} as const;
