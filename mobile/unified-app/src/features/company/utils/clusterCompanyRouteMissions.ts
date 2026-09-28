type RouteMission = {
  trip_flags?: Record<string, boolean | number | string | null> | null;
};

function routeGroupId(mission: RouteMission): string | null {
  const raw = mission.trip_flags?.route_group_id;
  if (typeof raw !== "string") return null;
  const value = raw.trim();
  return value || null;
}

function legNumber(mission: RouteMission): number {
  const raw = mission.trip_flags?.leg_number;
  const value = typeof raw === "number" ? raw : Number(raw);
  return Number.isFinite(value) ? value : 0;
}

/** Même regroupement que la table web : un parcours, segments dans l'ordre. */
export function clusterCompanyRouteMissions<T extends RouteMission>(missions: T[]): T[] {
  if (!Array.isArray(missions) || missions.length === 0) return missions;
  const byGroup = new Map<string, T[]>();
  missions.forEach((mission) => {
    const groupId = routeGroupId(mission);
    if (!groupId) return;
    const members = byGroup.get(groupId);
    if (members) members.push(mission);
    else byGroup.set(groupId, [mission]);
  });
  if (byGroup.size === 0) return missions;

  const seen = new Set<string>();
  const result: T[] = [];
  missions.forEach((mission) => {
    const groupId = routeGroupId(mission);
    if (!groupId) {
      result.push(mission);
      return;
    }
    if (seen.has(groupId)) return;
    seen.add(groupId);
    const members = [...(byGroup.get(groupId) ?? [])].sort(
      (left, right) => legNumber(left) - legNumber(right)
    );
    result.push(...members);
  });
  return result;
}
