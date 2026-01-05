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
 */
export function groupMissionsByPickup(missions: Booking[]): MissionGroup[] {
  const groupsMap = new Map<string, Booking[]>();

  // Grouper les missions par adresse de départ normalisée
  missions.forEach((mission) => {
    if (!mission.pickup_location) return;

    const normalized = normalizeAddress(mission.pickup_location);
    if (!groupsMap.has(normalized)) {
      groupsMap.set(normalized, []);
    }
    groupsMap.get(normalized)!.push(mission);
  });

  // Convertir en array de groupes
  const groups: MissionGroup[] = [];
  groupsMap.forEach((missionsInGroup, normalizedAddress) => {
    // Trouver l'adresse d'affichage (la première du groupe)
    const displayAddress = missionsInGroup[0]?.pickup_location || normalizedAddress;

    groups.push({
      id: `pickup_${normalizedAddress}`,
      location: normalizedAddress,
      locationDisplay: displayAddress,
      type: "pickup",
      missions: missionsInGroup.sort(
        (a, b) =>
          new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
      ),
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
 * Utile pour les cas où plusieurs clients vont au même endroit
 */
export function groupMissionsByDropoff(missions: Booking[]): MissionGroup[] {
  const groupsMap = new Map<string, Booking[]>();

  missions.forEach((mission) => {
    if (!mission.dropoff_location) return;

    const normalized = normalizeAddress(mission.dropoff_location);
    if (!groupsMap.has(normalized)) {
      groupsMap.set(normalized, []);
    }
    groupsMap.get(normalized)!.push(mission);
  });

  const groups: MissionGroup[] = [];
  groupsMap.forEach((missionsInGroup, normalizedAddress) => {
    const displayAddress = missionsInGroup[0]?.dropoff_location || normalizedAddress;

    groups.push({
      id: `dropoff_${normalizedAddress}`,
      location: normalizedAddress,
      locationDisplay: displayAddress,
      type: "dropoff",
      missions: missionsInGroup.sort(
        (a, b) =>
          new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
      ),
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

