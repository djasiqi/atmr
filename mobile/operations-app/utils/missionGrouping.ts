// utils/missionGrouping.ts
import { Booking } from "@/services/api";

/**
 * Normalise une adresse pour la comparaison (enlève les espaces, met en minuscule, etc.)
 */
export function normalizeAddress(address: string): string {
  if (!address) return "";
  return address
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[.,]/g, "")
    .trim()
    .substring(0, 50); // Prendre les 50 premiers caractères pour la comparaison
}

/**
 * Type pour représenter un groupe de missions
 */
export interface MissionGroup {
  id: string; // Identifiant unique du groupe (basé sur l'adresse)
  location: string; // Adresse normalisée
  locationDisplay: string; // Adresse d'affichage (première adresse du groupe)
  type: "pickup" | "dropoff"; // Type de groupement
  missions: Booking[];
  isGrouped: boolean; // true si plusieurs missions, false si une seule
}

/**
 * Groupe les missions par point de départ (pickup_location)
 * AVEC vérification de l'intervalle de 5 minutes entre les missions
 */
export function groupMissionsByPickup(missions: Booking[]): MissionGroup[] {
  const groupsMap = new Map<string, Booking[]>();
  const FIVE_MINUTES_MS = 5 * 60 * 1000; // 5 minutes en millisecondes

  // Trier les missions par heure d'abord
  const sortedMissions = [...missions].sort(
    (a, b) =>
      new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
  );

  // Grouper les missions par adresse ET intervalle de 5min
  sortedMissions.forEach((mission) => {
    if (!mission.pickup_location) return;

    const normalized = normalizeAddress(mission.pickup_location);
    const missionTime = new Date(mission.scheduled_time).getTime();

    // Chercher un groupe existant avec la même adresse ET dans les 5 minutes
    let groupKey: string | null = null;
    
    groupsMap.forEach((missionsInGroup, key) => {
      if (key.startsWith(`pickup_${normalized}_`)) {
        // Vérifier si cette mission est dans les 5 minutes de la dernière mission du groupe
        const lastMissionInGroup = missionsInGroup[missionsInGroup.length - 1];
        const lastMissionTime = new Date(lastMissionInGroup.scheduled_time).getTime();
        const timeDiff = missionTime - lastMissionTime;

        if (timeDiff >= 0 && timeDiff <= FIVE_MINUTES_MS) {
          groupKey = key;
        }
      }
    });

    // Si aucun groupe trouvé dans les 5min, créer un nouveau groupe
    if (!groupKey) {
      groupKey = `pickup_${normalized}_${missionTime}`;
      groupsMap.set(groupKey, []);
    }

    groupsMap.get(groupKey)!.push(mission);
  });

  // Convertir en array de groupes
  const groups: MissionGroup[] = [];
  groupsMap.forEach((missionsInGroup, groupKey) => {
    const displayAddress = missionsInGroup[0]?.pickup_location || "";
    const normalized = groupKey.split("_")[1] || "";

    groups.push({
      id: groupKey,
      location: normalized,
      locationDisplay: displayAddress,
      type: "pickup",
      missions: missionsInGroup,
      isGrouped: missionsInGroup.length > 1,
    });
  });

  // Trier les groupes par heure de la première mission
  return groups.sort((a, b) => {
    const timeA = new Date(a.missions[0]?.scheduled_time || 0).getTime();
    const timeB = new Date(b.missions[0]?.scheduled_time || 0).getTime();
    return timeA - timeB;
  });
}

/**
 * Groupe les missions par point d'arrivée (dropoff_location)
 * AVEC vérification de l'intervalle de 5 minutes entre les missions
 * Utile pour les cas où plusieurs clients vont au même endroit
 */
export function groupMissionsByDropoff(missions: Booking[]): MissionGroup[] {
  const groupsMap = new Map<string, Booking[]>();
  const FIVE_MINUTES_MS = 5 * 60 * 1000; // 5 minutes en millisecondes

  // Trier les missions par heure d'abord
  const sortedMissions = [...missions].sort(
    (a, b) =>
      new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
  );

  // Grouper les missions par adresse ET intervalle de 5min
  sortedMissions.forEach((mission) => {
    if (!mission.dropoff_location) return;

    const normalized = normalizeAddress(mission.dropoff_location);
    const missionTime = new Date(mission.scheduled_time).getTime();

    // Chercher un groupe existant avec la même adresse ET dans les 5 minutes
    let groupKey: string | null = null;
    
    groupsMap.forEach((missionsInGroup, key) => {
      if (key.startsWith(`dropoff_${normalized}_`)) {
        // Vérifier si cette mission est dans les 5 minutes de la dernière mission du groupe
        const lastMissionInGroup = missionsInGroup[missionsInGroup.length - 1];
        const lastMissionTime = new Date(lastMissionInGroup.scheduled_time).getTime();
        const timeDiff = missionTime - lastMissionTime;

        if (timeDiff >= 0 && timeDiff <= FIVE_MINUTES_MS) {
          groupKey = key;
        }
      }
    });

    // Si aucun groupe trouvé dans les 5min, créer un nouveau groupe
    if (!groupKey) {
      groupKey = `dropoff_${normalized}_${missionTime}`;
      groupsMap.set(groupKey, []);
    }

    groupsMap.get(groupKey)!.push(mission);
  });

  // Convertir en array de groupes
  const groups: MissionGroup[] = [];
  groupsMap.forEach((missionsInGroup, groupKey) => {
    const displayAddress = missionsInGroup[0]?.dropoff_location || "";
    const normalized = groupKey.split("_")[1] || "";

    groups.push({
      id: groupKey,
      location: normalized,
      locationDisplay: displayAddress,
      type: "dropoff",
      missions: missionsInGroup,
      isGrouped: missionsInGroup.length > 1,
    });
  });

  return groups.sort((a, b) => {
    const timeA = new Date(a.missions[0]?.scheduled_time || 0).getTime();
    const timeB = new Date(b.missions[0]?.scheduled_time || 0).getTime();
    return timeA - timeB;
  });
}

/**
 * Organise les missions pour l'affichage, en priorisant le groupement par pickup
 * Retourne une liste plate avec des indicateurs de groupe
 */
export interface DisplayMission {
  mission: Booking;
  missionNumber: number; // Numéro dans le groupe (1, 2, 3...)
  groupInfo: {
    isGrouped: boolean;
    groupId: string;
    groupLocation: string;
    groupLocationDisplay: string;
    groupType: "pickup" | "dropoff";
    groupSize: number;
    isFirstInGroup: boolean; // true si c'est la première mission du groupe
  };
}

export function organizeMissionsForDisplay(missions: Booking[]): DisplayMission[] {
  if (missions.length === 0) return [];

  // Grouper par pickup (priorité)
  const pickupGroups = groupMissionsByPickup(missions);

  const result: DisplayMission[] = [];
  let globalMissionNumber = 1;

  pickupGroups.forEach((group) => {
    group.missions.forEach((mission, indexInGroup) => {
      result.push({
        mission,
        missionNumber: globalMissionNumber++,
        groupInfo: {
          isGrouped: group.isGrouped,
          groupId: group.id,
          groupLocation: group.location,
          groupLocationDisplay: group.locationDisplay,
          groupType: group.type,
          groupSize: group.missions.length,
          isFirstInGroup: indexInGroup === 0,
        },
      });
    });
  });

  return result;
}

/**
 * Filtre les missions pour n'afficher que celles qui sont actives ou proches
 * - En cours (in_progress, en_route)
 * - Assignées pour aujourd'hui
 * - À partir de 19h00, affiche aussi les courses du lendemain
 */
export function filterActiveMissions(missions: Booking[]): Booking[] {
  const now = new Date();
  const currentHour = now.getHours();
  const todayStart = new Date(now);
  todayStart.setHours(0, 0, 0, 0);
  const todayEnd = new Date(now);
  todayEnd.setHours(23, 59, 59, 999);

  // Si après 19h00, étendre jusqu'à demain 23h59
  const endOfPeriod = currentHour >= 19
    ? new Date(todayEnd.getTime() + 24 * 60 * 60 * 1000) // +1 jour
    : todayEnd;

  return missions.filter((mission) => {
    const status = mission.status?.toLowerCase() || "";
    const scheduledTime = new Date(mission.scheduled_time).getTime();

    // Toujours afficher les missions en cours ou en route
    if (status === "in_progress" || status === "en_route") {
      return true;
    }

    // Afficher les missions assignées d'aujourd'hui (ou demain si après 19h)
    if (status === "assigned") {
      return scheduledTime >= todayStart.getTime() && scheduledTime <= endOfPeriod.getTime();
    }

    return false;
  });
}

/**
 * Filtre pour n'afficher QUE le prochain groupe de missions
 * 
 * Logique :
 * - Affiche uniquement la prochaine mission OU le prochain groupe de missions
 * - Critères de groupement :
 *   1. Même adresse de pickup (normalisée)
 *   2. Écart de temps ≤ 5 minutes entre les courses
 * - Si une course d'un groupe passe "en_route", TOUTES les courses du groupe restent affichées
 * 
 * Exemples :
 * - Course A 10h00, Course B 11h00 → Affiche uniquement Course A
 * - Course A 10h00, Course B 10h00 (même lieu) → Affiche A et B (groupe)
 * - Course A 10h00, Course B 10h05 (même lieu) → Affiche A et B (groupe)
 * - Course A "en_route", Course B "assigned" (même groupe) → Affiche A ET B
 * - Après complétion de A et B → Affiche la course suivante
 */
export function filterNextMissionsOnly(missions: Booking[]): Booking[] {
  if (missions.length === 0) return [];

  // 1. Trouver les missions en cours ou en route
  const inProgressOrEnRoute = missions.filter((m) => {
    const status = m.status?.toLowerCase() || "";
    return status === "in_progress" || status === "en_route";
  });

  // 2. Si des missions sont en cours/en route, on doit inclure TOUT leur groupe
  if (inProgressOrEnRoute.length > 0) {
    // Construire un Set des pickups et times des missions actives
    const activeGroupKeys = new Set<string>();
    const GROUPING_WINDOW_MS = 5 * 60 * 1000; // 5 minutes

    inProgressOrEnRoute.forEach((activeMission) => {
      const activePickup = normalizeAddress(activeMission.pickup_location || "");
      const activeTime = new Date(activeMission.scheduled_time).getTime();
      
      // Trouver toutes les missions du même groupe (même pickup, dans les 5 min)
      missions.forEach((mission) => {
        const missionPickup = normalizeAddress(mission.pickup_location || "");
        const missionTime = new Date(mission.scheduled_time).getTime();
        const timeDiff = Math.abs(missionTime - activeTime);

        // Si même pickup ET dans les 5 minutes → même groupe
        if (missionPickup === activePickup && timeDiff <= GROUPING_WINDOW_MS) {
          const groupKey = `${missionPickup}_${Math.floor(missionTime / GROUPING_WINDOW_MS)}`;
          activeGroupKeys.add(groupKey);
        }
      });
    });

    // Filtrer pour garder toutes les missions des groupes actifs
    const groupedMissions = missions.filter((mission) => {
      const status = mission.status?.toLowerCase() || "";
      // Exclure les missions completed ou cancelled
      if (status === "completed" || status === "cancelled" || status === "canceled") {
        return false;
      }

      const missionPickup = normalizeAddress(mission.pickup_location || "");
      const missionTime = new Date(mission.scheduled_time).getTime();
      const groupKey = `${missionPickup}_${Math.floor(missionTime / GROUPING_WINDOW_MS)}`;

      return activeGroupKeys.has(groupKey);
    });

    return groupedMissions;
  }

  // 3. Si aucune mission en cours, trier les missions assignées par heure
  const assignedMissions = missions
    .filter((m) => m.status?.toLowerCase() === "assigned")
    .sort(
      (a, b) =>
        new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
    );

  if (assignedMissions.length === 0) return [];

  // 4. Prendre la première mission (la plus proche dans le temps)
  const firstMission = assignedMissions[0];
  const firstMissionTime = new Date(firstMission.scheduled_time).getTime();
  const firstPickup = normalizeAddress(firstMission.pickup_location || "");

  // 5. Trouver toutes les missions dans le même groupe
  const GROUPING_WINDOW_MS = 5 * 60 * 1000; // 5 minutes

  const nextGroup = assignedMissions.filter((mission) => {
    const missionTime = new Date(mission.scheduled_time).getTime();
    const missionPickup = normalizeAddress(mission.pickup_location || "");
    const timeDiff = Math.abs(missionTime - firstMissionTime);

    // Grouper si :
    // - Même adresse de pickup (normalisée) ET
    // - Dans les 5 minutes de la première mission
    return missionPickup === firstPickup && timeDiff <= GROUPING_WINDOW_MS;
  });

  return nextGroup;
}

/**
 * Coordonnées pour la prochaine destination (priorité aux coords backend)
 */
export interface NextDestinationCoords {
  latitude: number;
  longitude: number;
}

/**
 * Trouve les coordonnées de la prochaine destination à afficher sur la carte.
 * Priorité : mission en cours (in_progress) > mission en route (en_route) > mission assignée.
 * Utilise pickup_lat/lon et dropoff_lat/lon du backend quand disponibles.
 */
export function getNextDestinationCoords(missions: Booking[]): NextDestinationCoords | null {
  if (missions.length === 0) return null;

  const inProgress = missions.find((m) => m.status?.toLowerCase() === "in_progress");
  if (inProgress?.dropoff_lat != null && inProgress?.dropoff_lon != null) {
    return { latitude: inProgress.dropoff_lat, longitude: inProgress.dropoff_lon };
  }

  const enRoute = missions.find((m) => m.status?.toLowerCase() === "en_route");
  if (enRoute?.pickup_lat != null && enRoute?.pickup_lon != null) {
    return { latitude: enRoute.pickup_lat, longitude: enRoute.pickup_lon };
  }

  const assigned = missions
    .filter((m) => m.status?.toLowerCase() === "assigned")
    .sort(
      (a, b) =>
        new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
    );

  if (assigned.length > 0 && assigned[0].pickup_lat != null && assigned[0].pickup_lon != null) {
    return { latitude: assigned[0].pickup_lat, longitude: assigned[0].pickup_lon };
  }

  const firstWithCoords = missions.find(
    (m) =>
      (m.status?.toLowerCase() === "in_progress" && m.dropoff_lat != null && m.dropoff_lon != null) ||
      (m.status?.toLowerCase() === "en_route" && m.pickup_lat != null && m.pickup_lon != null) ||
      (m.pickup_lat != null && m.pickup_lon != null)
  );
  if (firstWithCoords?.dropoff_lat != null && firstWithCoords?.dropoff_lon != null) {
    return { latitude: firstWithCoords.dropoff_lat, longitude: firstWithCoords.dropoff_lon };
  }
  if (firstWithCoords?.pickup_lat != null && firstWithCoords?.pickup_lon != null) {
    return { latitude: firstWithCoords.pickup_lat, longitude: firstWithCoords.pickup_lon };
  }

  return null;
}

/**
 * Trouve la prochaine destination à afficher sur la carte
 * Priorité : mission en cours (in_progress) > mission assignée la plus proche
 */
export function getNextDestination(missions: Booking[]): string | null {
  if (missions.length === 0) return null;

  // Chercher une mission en cours
  const inProgress = missions.find(
    (m) => m.status?.toLowerCase() === "in_progress"
  );
  if (inProgress?.dropoff_location) {
    return inProgress.dropoff_location;
  }

  // Chercher une mission en route vers le pickup
  const enRoute = missions.find((m) => m.status?.toLowerCase() === "en_route");
  if (enRoute?.pickup_location) {
    return enRoute.pickup_location;
  }

  // Sinon, prendre la mission assignée la plus proche dans le temps
  const assigned = missions
    .filter((m) => m.status?.toLowerCase() === "assigned")
    .sort(
      (a, b) =>
        new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
    );

  if (assigned.length > 0 && assigned[0].pickup_location) {
    return assigned[0].pickup_location;
  }

  // Fallback : première mission avec pickup_location
  const firstWithPickup = missions.find((m) => m.pickup_location);
  return firstWithPickup?.pickup_location || null;
}

