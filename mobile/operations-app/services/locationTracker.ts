// services/locationTracker.ts
// Tracker GPS adaptatif avec fréquence variable selon mouvement

import * as Location from "expo-location";
import { getLogger } from "@/utils/logger";
import { getDistanceInMeters } from "./location";
import { sendDriverLocation } from "./location";
import { type DriverLocationPayload } from "./api";
import { getSocket, getSocketRole } from "./socket";

const log = getLogger("Tracker");

/**
 * Tracker GPS adaptatif qui ajuste la fréquence selon la vitesse.
 * 
 * - En mouvement (> 3.6 km/h) : 5s
 * - Immobile : 30s
 * - Batterie < 20% : 60s (mode économie)
 */
export type GpsStatus = "active" | "disabled" | "unavailable" | "unknown";

export class AdaptiveLocationTracker {
  private locationSub: Location.LocationSubscription | null = null;
  private updateInterval: number = 5000; // 5s par défaut
  private lastPosition: Location.LocationObject | null = null;
  private lastSpeed: number = 0;
  private lastSentAt: number = 0;
  private isTracking: boolean = false;
  private batteryCheckInterval: ReturnType<typeof setInterval> | null = null;
  private trackingTimeout: ReturnType<typeof setTimeout> | null = null;

  private _gpsStatus: GpsStatus = "unknown";
  private gpsStatusListeners: Set<(status: GpsStatus) => void> = new Set();

  onGpsStatusChange(listener: (status: GpsStatus) => void): () => void {
    this.gpsStatusListeners.add(listener);
    listener(this._gpsStatus);
    return () => this.gpsStatusListeners.delete(listener);
  }

  get gpsStatus(): GpsStatus {
    return this._gpsStatus;
  }

  private setGpsStatus(status: GpsStatus): void {
    if (status === this._gpsStatus) return;
    this._gpsStatus = status;
    this.gpsStatusListeners.forEach((fn) => fn(status));
  }

  // Seuils de vitesse (m/s)
  private readonly SPEED_THRESHOLD_MOVING = 1.0; // 3.6 km/h
  // Config via env: EXPO_PUBLIC_GPS_FAST_MS=5000, EXPO_PUBLIC_GPS_SLOW_MS=30000
  private readonly INTERVAL_MOVING_MS =
    parseInt(process.env.EXPO_PUBLIC_GPS_FAST_MS ?? "5000", 10) || 5000;
  private readonly INTERVAL_STATIONARY_MS =
    parseInt(process.env.EXPO_PUBLIC_GPS_SLOW_MS ?? "30000", 10) || 30000;
  private readonly INTERVAL_BATTERY_LOW_MS = 60000; // 60s si batterie < 20%

  /**
   * Démarrer le tracking adaptatif.
   */
  async startTracking(): Promise<void> {
    if (this.isTracking) {
      log.warn("tracking already active");
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
    log.success("adaptive tracking started");
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
        log.warn("stop subscription error", { error: e });
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

    log.success("tracking stopped");
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
        log.info("low battery interval reduced", {
          batteryLevel: (batteryLevel * 100).toFixed(0),
          intervalMs: this.updateInterval,
        });
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
      const enabled = await Location.hasServicesEnabledAsync();
      if (!enabled) {
        this.setGpsStatus("disabled");
        log.warn("location services disabled, retrying later");
        this.scheduleNextUpdate();
        return;
      }

      let location: Location.LocationObject | null = null;

      try {
        location = await Location.getCurrentPositionAsync({
          accuracy: Location.Accuracy.Balanced,
          timeInterval: 10000,
        });
      } catch {
        log.warn("getCurrentPositionAsync failed, trying last known position");
        location = await Location.getLastKnownPositionAsync();
      }

      if (!location) {
        this.setGpsStatus("unavailable");
        log.warn("no location available, retrying later");
        this.scheduleNextUpdate();
        return;
      }

      this.setGpsStatus("active");

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
      log.error("position update failed", { error });
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
      if (socket && socket.connected && getSocketRole() === "driver") {
        socket.emit("driver_location", payload);
        log.success("position sent", {
          via: "socket",
          speedKmh: (this.lastSpeed * 3.6).toFixed(1),
          intervalMs: this.updateInterval,
        });
      } else {
        // Fallback HTTP
        await sendDriverLocation(payload);
        log.success("position sent", {
          via: "http",
          speedKmh: (this.lastSpeed * 3.6).toFixed(1),
          intervalMs: this.updateInterval,
        });
      }
    } catch (error) {
      log.error("send position failed", { error });
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
          log.info("battery level", { level: (batteryLevel * 100).toFixed(0) });
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

