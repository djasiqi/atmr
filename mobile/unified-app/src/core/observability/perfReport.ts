import { buildPerfInstrumentationReport, type PerfBucketReportRow } from "./perfInstrumentationStore";

export type PerfReportMarkdownOptions = {
  topN?: number;
  title?: string;
};

function formatRow(row: PerfBucketReportRow): string {
  return [
    `| ${row.category} | ${row.sub_key} | ${row.role} | ${row.screen} |`,
    ` ${row.count} | ${row.sum_ms.toFixed(1)} | ${row.p50_ms.toFixed(1)} |`,
    ` ${row.p95_ms.toFixed(1)} | ${row.max_ms.toFixed(1)} |`,
  ].join("");
}

export function buildPerfReportMarkdown(options: PerfReportMarkdownOptions = {}): string {
  const topN = options.topN ?? 10;
  const report = buildPerfInstrumentationReport(topN);
  const title = options.title ?? "Rapport perf unified-app (Sprint 0C)";
  const header = [
    `# ${title}`,
    ``,
    `Généré : ${report.generated_at}`,
    ``,
    `## Top ${topN} par occurrences`,
    ``,
    `| catégorie | sous-clé | rôle | écran | count | sum_ms | p50 | p95 | max |`,
    `| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |`,
    ...report.top_by_count.map(formatRow),
    ``,
    `## Top ${topN} par temps cumulé`,
    ``,
    `| catégorie | sous-clé | rôle | écran | count | sum_ms | p50 | p95 | max |`,
    `| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |`,
    ...report.top_by_sum_ms.map(formatRow),
    ``,
  ];
  return header.join("\n");
}

export function dumpPerfReportToConsole(): void {
  const json = buildPerfInstrumentationReport(10);
   
  console.info("[perf-report]", JSON.stringify(json, null, 2));
   
  console.info(buildPerfReportMarkdown());
}

export function getPerfReportSnapshot(topN = 10): ReturnType<typeof buildPerfInstrumentationReport> {
  return buildPerfInstrumentationReport(topN);
}

declare global {
   
  var __dumpPerfReport__: (() => void) | undefined;
   
  var __getPerfReport__: ((topN?: number) => ReturnType<typeof buildPerfInstrumentationReport>) | undefined;
}

if (typeof __DEV__ !== "undefined" && __DEV__) {
  globalThis.__dumpPerfReport__ = dumpPerfReportToConsole;
  globalThis.__getPerfReport__ = getPerfReportSnapshot;
}
