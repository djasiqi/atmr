import { getPerfActiveContext } from "./perfActiveContext";

export type PerfMetricCategory =
  | "notify"
  | "invalidate"
  | "socket_channel"
  | "js_long_task"
  | "heap"
  | "http"
  | "page_load"
  | "context_switch"
  | "fleet_map"
  | "react_query_refetch"
  | "message_send"
  | "chat_cache_mismatch"
  | "thread_cache"
  | "thread_runtime"
  | "tap"
  | "screen_render"
  | "query_cache"
  | "api_roundtrip"
  | "mission_details";

type BucketKey = string;

type Bucket = {
  count: number;
  sumMs: number;
  maxMs: number;
  /** Ring buffer for percentile estimates (capped). */
  samples: number[];
};

const MAX_SAMPLES_PER_BUCKET = 500;

const buckets = new Map<BucketKey, Bucket>();

function bucketKey(category: PerfMetricCategory, subKey: string): BucketKey {
  const ctx = getPerfActiveContext();
  return `${category}|${subKey}|${ctx.role}|${ctx.screen}`;
}

function touchBucket(key: BucketKey, durationMs: number, countIncrement = 1): void {
  let bucket = buckets.get(key);
  if (!bucket) {
    bucket = { count: 0, sumMs: 0, maxMs: 0, samples: [] };
    buckets.set(key, bucket);
  }
  bucket.count += countIncrement;
  if (durationMs > 0) {
    bucket.sumMs += durationMs;
    if (durationMs > bucket.maxMs) bucket.maxMs = durationMs;
    if (bucket.samples.length < MAX_SAMPLES_PER_BUCKET) {
      bucket.samples.push(durationMs);
    }
  }
}

export function recordPerfBucket(
  category: PerfMetricCategory,
  subKey: string,
  durationMs = 0,
  countIncrement = 1
): void {
  touchBucket(bucketKey(category, subKey), durationMs, countIncrement);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)] ?? 0;
}

export type PerfBucketReportRow = {
  category: PerfMetricCategory;
  sub_key: string;
  role: string;
  screen: string;
  count: number;
  sum_ms: number;
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
};

export function buildPerfInstrumentationReport(topN = 10): {
  generated_at: string;
  rows: PerfBucketReportRow[];
  top_by_count: PerfBucketReportRow[];
  top_by_sum_ms: PerfBucketReportRow[];
} {
  const rows: PerfBucketReportRow[] = [];
  for (const [key, bucket] of buckets) {
    const [category, subKey, role, screen] = key.split("|") as [
      PerfMetricCategory,
      string,
      string,
      string,
    ];
    const sorted = [...bucket.samples].sort((a, b) => a - b);
    rows.push({
      category,
      sub_key: subKey,
      role,
      screen,
      count: bucket.count,
      sum_ms: bucket.sumMs,
      avg_ms: bucket.count > 0 && bucket.sumMs > 0 ? bucket.sumMs / bucket.count : 0,
      p50_ms: percentile(sorted, 50),
      p95_ms: percentile(sorted, 95),
      max_ms: bucket.maxMs,
    });
  }
  const topByCount = [...rows].sort((a, b) => b.count - a.count).slice(0, topN);
  const topBySum = [...rows].sort((a, b) => b.sum_ms - a.sum_ms).slice(0, topN);
  return {
    generated_at: new Date().toISOString(),
    rows,
    top_by_count: topByCount,
    top_by_sum_ms: topBySum,
  };
}

export function resetPerfInstrumentationStoreForTests(): void {
  buckets.clear();
}

export function getPerfInstrumentationBucketCountForTests(): number {
  return buckets.size;
}
