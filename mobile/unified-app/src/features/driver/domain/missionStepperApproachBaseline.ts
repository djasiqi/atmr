export type StepperApproachSegment = "pickup" | "dropoff";

const baselineMetersByKey = new Map<string, number>();

const MIN_BASELINE_METERS = 50;

function baselineKey(missionId: number, segment: StepperApproachSegment): string {
  return `${missionId}:${segment}`;
}

export function resetStepperApproachBaseline(
  missionId: number,
  segment: StepperApproachSegment
): void {
  baselineMetersByKey.delete(baselineKey(missionId, segment));
}

export function clearStepperApproachBaseline(
  missionId: number,
  segment: StepperApproachSegment
): void {
  baselineMetersByKey.delete(baselineKey(missionId, segment));
}

export function clearAllStepperApproachBaselines(missionId: number): void {
  baselineMetersByKey.delete(baselineKey(missionId, "pickup"));
  baselineMetersByKey.delete(baselineKey(missionId, "dropoff"));
}

/**
 * Distance de référence figée au début de chaque segment (première mesure).
 */
export function resolveStepperApproachBaseline(
  missionId: number,
  segment: StepperApproachSegment,
  currentMeters: number
): number {
  const key = baselineKey(missionId, segment);
  const stored = baselineMetersByKey.get(key);
  if (stored != null && stored > 0) {
    return stored;
  }
  const baseline = Math.max(currentMeters, MIN_BASELINE_METERS);
  baselineMetersByKey.set(key, baseline);
  return baseline;
}

export function resetDriverStepperApproachBaselinesForTests(): void {
  baselineMetersByKey.clear();
}
