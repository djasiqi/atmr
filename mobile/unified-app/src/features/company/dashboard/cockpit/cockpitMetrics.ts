type CockpitMetricEvent =
  | "sheet_open_ms"
  | "camera_recenter_ms"
  | "marker_to_action_ms"
  | "marker_rerender"
  | "cluster_recalc"
  | "fps_sample"
  | "density_level"
  | "routes_visible"
  | "orchestration_frame_ms";

type Sample = { t: number; v: number };

const samples: Partial<Record<CockpitMetricEvent, Sample[]>> = {};
const MAX_SAMPLES = 48;

function pushSample(key: CockpitMetricEvent, value: number) {
  const list = samples[key] ?? [];
  list.push({ t: Date.now(), v: value });
  if (list.length > MAX_SAMPLES) list.shift();
  samples[key] = list;
}

export function recordCockpitMetric(key: CockpitMetricEvent, value: number) {
  if (!__DEV__ && process.env.EXPO_PUBLIC_COCKPIT_METRICS !== "1") return;
  pushSample(key, value);
}

export function recordCockpitDuration(key: Exclude<CockpitMetricEvent, "marker_rerender" | "cluster_recalc" | "fps_sample">, startMs: number) {
  recordCockpitMetric(key, Math.max(0, Date.now() - startMs));
}

export function getCockpitMetricSnapshot(): Record<string, { avg: number; count: number }> {
  const out: Record<string, { avg: number; count: number }> = {};
  for (const [key, list] of Object.entries(samples)) {
    if (!list?.length) continue;
    const avg = list.reduce((s, x) => s + x.v, 0) / list.length;
    out[key] = { avg: Math.round(avg), count: list.length };
  }
  return out;
}
