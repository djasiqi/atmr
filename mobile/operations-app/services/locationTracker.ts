// services/locationTracker.ts
// Tracker GPS adaptatif avec fréquence variable selon mouvement

import * as Location from "expo-location";
import { getDistanceInMeters } from "./location";
import { sendDriverLocation } from "./location";
import { type DriverLocationPayload } from "./api";
import { getSocket } from "./socket";

/**
 * Tracker GPS adaptatif qui ajuste la fréquence selon la vitesse.
 * 
 * - En mouvement (> 3.6 km/h) : 5s
 * - Immobile : 30s
 * - Batterie < 20% : 60s (mode économie)
 */
export class AdaptiveLocationTracker {
  private locationSub: Location.LocationSubscription | null = null;
  private updateInterval: number = 5000; // 5s par défaut
  private lastPosition: Location.LocationObject | null = null;
  private lastSpeed: number = 0;
  private lastSentAt: number = 0;
  private isTracking: boolean = false;
  private batteryCheckInterval: ReturnType<typeof setInterval> | null = null;
  private trackingTimeout: ReturnType<typeof setTimeout> | null = null;

  // Seuils de vitesse (m/s)
  private readonly SPEED_THRESHOLD_MOVING = 1.0; // 3.6 km/h
  private readonly INTERVAL_MOVING_MS = 5000; // 5s en mouvement
  private readonly INTERVAL_STATIONARY_MS = 30000; // 30s immobile
  private readonly INTERVAL_BATTERY_LOW_MS = 60000; // 60s si batterie < 20%

  /**
   * Démarrer le tracking adaptatif.
   */
  async startTracking(): Promise<void> {
    if (this.isTracking) {
      console.log("[AdaptiveLocationTracker] ⚠️ Tracking déjà actif");
      return;
    }

    // Demander permissions
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== "granted") {
      throw new Error("Permission de localisation refusée");
    }

    this.isTracking = true;
    this.lastSentAt = Date.now();

    // Démarrer vérification batterie (toutes les 60s)
    this.startBatteryMonitoring();

    // Démarrer le tracking
    this.scheduleNextUpdate();
    console.log("[AdaptiveLocationTracker] ✅ Tracking adaptatif démarré");
  }

  /**
   * Arrêter le tracking.
   */
  stopTracking(): void {
    if (!this.isTracking) {
      return;
    }

    this.isTracking = false;

    if (this.locationSub) {
      try {
        this.locationSub.remove();
      } catch (e) {
        console.warn("[AdaptiveLocationTracker] Erreur arrêt subscription:", e);
      }
      this.locationSub = null;
    }

    if (this.trackingTimeout) {
      clearTimeout(this.trackingTimeout);
      this.trackingTimeout = null;
    }

    if (this.batteryCheckInterval) {
      clearInterval(this.batteryCheckInterval);
      this.batteryCheckInterval = null;
    }

    console.log("[AdaptiveLocationTracker] ⏹️ Tracking arrêté");
  }

  /**
   * Programmer la prochaine mise à jour.
   */
  private scheduleNextUpdate(): void {
    if (!this.isTracking) {
      return;
    }

    // Calculer vitesse moyenne depuis dernière position
    const speed = this.calculateSpeed();

    // Adapter fréquence selon vitesse
    if (speed > this.SPEED_THRESHOLD_MOVING) {
      // En mouvement (> 3.6 km/h)
      this.updateInterval = this.INTERVAL_MOVING_MS;
    } else {
      // Immobile
      this.updateInterval = this.INTERVAL_STATIONARY_MS;
    }

    // Vérifier batterie et ajuster si nécessaire
    this.checkBatteryLevel().then((batteryLevel) => {
      if (batteryLevel !== null && batteryLevel < 0.2) {
        // < 20% : mode économie
        this.updateInterval = this.INTERVAL_BATTERY_LOW_MS;
        console.log(
          `[AdaptiveLocationTracker] 🔋 Batterie faible (${(batteryLevel * 100).toFixed(0)}%), fréquence réduite à ${this.updateInterval}ms`
        );
      }
    });

    // Programmer prochaine mise à jour
    this.trackingTimeout = setTimeout(() => {
      this.updateLocation();
    }, this.updateInterval);
  }

  /**
   * Mettre à jour la position.
   */
  private async updateLocation(): Promise<void> {
    if (!this.isTracking) {
      return;
    }

    try {
      // Récupérer position actuelle
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced, // Bon compromis précision/batterie
      });

      // Calculer vitesse
      const speed = this.calculateSpeedFromLocation(location);
      this.lastSpeed = speed;

      // Vérifier si on doit envoyer (distance ou temps)
      const shouldSend = this.shouldSendLocation(location);

      if (shouldSend) {
        await this.sendLocation(location);
        this.lastSentAt = Date.now();
      }

      this.lastPosition = location;

      // Programmer prochaine mise à jour
      this.scheduleNextUpdate();
    } catch (error) {
      console.error("[AdaptiveLocationTracker] ❌ Erreur mise à jour position:", error);
      // Réessayer après un délai
      this.trackingTimeout = setTimeout(() => {
        this.updateLocation();
      }, this.updateInterval);
    }
  }

  /**
   * Calculer vitesse moyenne depuis dernière position.
   */
  private calculateSpeed(): number {
    if (!this.lastPosition) {
      return 0;
    }

    // Utiliser vitesse du GPS si disponible
    if (this.lastPosition.coords.speed !== null && this.lastPosition.coords.speed > 0) {
      return this.lastPosition.coords.speed; // m/s
    }

    // Sinon, calculer depuis distance/temps
    return this.lastSpeed; // Garder dernière vitesse calculée
  }

  /**
   * Calculer vitesse depuis une nouvelle position.
   */
  private calculateSpeedFromLocation(location: Location.LocationObject): number {
    if (!this.lastPosition) {
      return 0;
    }

    // Utiliser vitesse GPS si disponible
    if (location.coords.speed !== null && location.coords.speed > 0) {
      return location.coords.speed; // m/s
    }

    // Calculer depuis distance/temps
    const distance = getDistanceInMeters(
      this.lastPosition.coords.latitude,
      this.lastPosition.coords.longitude,
      location.coords.latitude,
      location.coords.longitude
    );

    const timeDelta = ((location.timestamp || Date.now()) - (this.lastPosition.timestamp || Date.now())) / 1000; // secondes

    if (timeDelta <= 0) {
      return 0;
    }

    return distance / timeDelta; // m/s
  }

  /**
   * Déterminer si on doit envoyer la position.
   */
  private shouldSendLocation(location: Location.LocationObject): boolean {
    const now = Date.now();
    const timeSinceLastSend = now - this.lastSentAt;

    // Toujours envoyer si > 60s depuis dernier envoi (heartbeat)
    if (timeSinceLastSend >= 60000) {
      return true;
    }

    if (!this.lastPosition) {
      return true; // Première position
    }

    // Vérifier distance
    const distance = getDistanceInMeters(
      this.lastPosition.coords.latitude,
      this.lastPosition.coords.longitude,
      location.coords.latitude,
      location.coords.longitude
    );

    // Envoyer si déplacement > 10m
    return distance >= 10;
  }

  /**
   * Envoyer la position au serveur.
   */
  private async sendLocation(location: Location.LocationObject): Promise<void> {
    const { latitude, longitude, speed, heading, accuracy } = location.coords;

    const payload: DriverLocationPayload = {
      latitude: Number(latitude),
      longitude: Number(longitude),
      speed: speed ? Number(speed) : undefined,
      heading: heading ? Number(heading) : undefined,
      accuracy: accuracy ? Number(accuracy) : undefined,
      timestamp: location.timestamp || Date.now(),
    };

    try {
      // Essayer Socket.IO d'abord (plus efficace)
      const socket = getSocket();
      if (socket && socket.connected) {
        socket.emit("driver_location", payload);
        console.log(
          `[AdaptiveLocationTracker] 📍 Position envoyée via Socket.IO (speed=${(this.lastSpeed * 3.6).toFixed(1)} km/h, interval=${this.updateInterval}ms)`
        );
      } else {
        // Fallback HTTP
        await sendDriverLocation(payload);
        console.log(
          `[AdaptiveLocationTracker] 📍 Position envoyée via HTTP (speed=${(this.lastSpeed * 3.6).toFixed(1)} km/h, interval=${this.updateInterval}ms)`
        );
      }
    } catch (error) {
      console.error("[AdaptiveLocationTracker] ❌ Erreur envoi position:", error);
      // Ne pas bloquer le tracking en cas d'erreur
    }
  }

  /**
   * Démarrer monitoring batterie.
   */
  private startBatteryMonitoring(): void {
    // Vérifier batterie toutes les 60s
    this.batteryCheckInterval = setInterval(() => {
      this.checkBatteryLevel().then((batteryLevel) => {
        if (batteryLevel !== null) {
          console.log(
            `[AdaptiveLocationTracker] 🔋 Niveau batterie: ${(batteryLevel * 100).toFixed(0)}%`
          );
        }
      });
    }, 60000);
  }

  /**
   * Vérifier niveau batterie (si API disponible).
   */
  private async checkBatteryLevel(): Promise<number | null> {
    try {
      // Expo Battery API (si disponible)
      const Battery = require("expo-battery");
      if (Battery && Battery.getBatteryLevelAsync) {
        const batteryLevel = await Battery.getBatteryLevelAsync();
        return batteryLevel; // 0.0 à 1.0
      }
    } catch (error) {
      // API batterie non disponible (normal sur certaines plateformes)
      // console.log("[AdaptiveLocationTracker] API batterie non disponible");
    }
    return null;
  }

  /**
   * Obtenir statistiques du tracker.
   */
  getStats(): {
    isTracking: boolean;
    updateInterval: number;
    lastSpeed: number;
    lastSentAt: number;
  } {
    return {
      isTracking: this.isTracking,
      updateInterval: this.updateInterval,
      lastSpeed: this.lastSpeed,
      lastSentAt: this.lastSentAt,
    };
  }
}

// Instance globale (singleton)
let globalTracker: AdaptiveLocationTracker | null = null;

/**
 * Obtenir l'instance globale du tracker.
 */
export function getAdaptiveLocationTracker(): AdaptiveLocationTracker {
  if (!globalTracker) {
    globalTracker = new AdaptiveLocationTracker();
  }
  return globalTracker;
}

/**
 * Démarrer le tracking adaptatif (helper).
 */
export async function startAdaptiveLocationTracking(): Promise<void> {
  const tracker = getAdaptiveLocationTracker();
  await tracker.startTracking();
}

/**
 * Arrêter le tracking adaptatif (helper).
 */
export function stopAdaptiveLocationTracking(): void {
  const tracker = getAdaptiveLocationTracker();
  tracker.stopTracking();
}

