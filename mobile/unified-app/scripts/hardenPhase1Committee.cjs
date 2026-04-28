#!/usr/bin/env node
/* eslint-env node */
/* global __dirname */

const fs = require("fs");
const path = require("path");

const DRY_RUN = process.argv.includes("--dry-run");
const STRICT = process.argv.includes("--strict") || process.argv.includes("--strict-release");
const STRICT_RELEASE = process.argv.includes("--strict-release");
const TODAY = new Date().toISOString().slice(0, 10);

const DRIVER_MATRIX_PATH = path.resolve(
  __dirname,
  "../docs/migration/DRIVER_PARITY_EXECUTION_MATRIX.csv"
);
const COMPANY_MATRIX_PATH = path.resolve(
  __dirname,
  "../docs/migration/COMPANY_PARITY_EXECUTION_MATRIX.csv"
);
const PHASE1_AUTH_PATH = path.resolve(
  __dirname,
  "../docs/migration/PHASE1_CLOSE_AUTHORIZATION.md"
);
const PHASE1_BLOCKERS_PATH = path.resolve(
  __dirname,
  "../docs/migration/PHASE1_COMMITTEE_BLOCKERS.md"
);
const EVIDENCE_DIR = path.resolve(__dirname, "../docs/migration/phase1_evidence");

const EVIDENCE_FILES = {
  g2: "realtime_reconnect.log.md",
  g3Android: "tracking_background_android.log.md",
  g3Ios: "tracking_background_ios.log.md",
  g4: "quick_actions_campaign.csv.md",
  g5: "offline_replay_session.log.md",
  g6: "resume_after_kill.log.md",
  g7: "auth_refresh_session.log.md",
  missionLifecycle: "mission_lifecycle_convergence.log.md",
  company: "company_jtbd_execution.md",
  rollback: "rollback_drill.log.md",
  dispatchSemantics: "dispatch_semantics_validation.md",
};

function readFileSafe(filePath) {
  if (!fs.existsSync(filePath)) return "";
  return fs.readFileSync(filePath, "utf8");
}

function parseBool(value) {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase();
  if (["true", "yes", "y", "ok", "pass", "passed", "ready"].includes(normalized)) return true;
  if (["false", "no", "n", "ko", "fail", "failed", "not_ready"].includes(normalized)) return false;
  return null;
}

function parseNumber(value) {
  const normalized = String(value ?? "")
    .trim()
    .replace(",", ".");
  if (!normalized) return null;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseKvMarkdown(content) {
  const map = {};
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const kv = line.match(/^-?\s*([A-Za-z0-9_]+)\s*:\s*(.*)$/);
    if (!kv) continue;
    map[kv[1]] = kv[2].trim();
  }
  return map;
}

function parseCsvBlock(content) {
  const blockMatch = content.match(/```csv([\s\S]*?)```/m);
  if (!blockMatch) return [];
  const lines = blockMatch[1]
    .trim()
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const values = splitCsvLine(lines[i]);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

function splitCsvLine(line) {
  const out = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      const next = line[i + 1];
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (char === "," && !inQuotes) {
      out.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  out.push(current);
  return out;
}

function toCsvLine(values) {
  return values
    .map((value) => {
      const input = value == null ? "" : String(value);
      if (input.includes(",") || input.includes('"') || input.includes("\n")) {
        return `"${input.replace(/"/g, '""')}"`;
      }
      return input;
    })
    .join(",");
}

function readCsvMatrix(csvPath) {
  const content = readFileSafe(csvPath).trim();
  if (!content) throw new Error(`CSV matrix missing or empty: ${csvPath}`);
  const lines = content.split(/\r?\n/);
  const header = splitCsvLine(lines[0]);
  const rows = lines.slice(1).filter(Boolean).map((line) => splitCsvLine(line));
  return { header, rows };
}

function ensureColumns(matrix, additionalColumns) {
  additionalColumns.forEach((column) => {
    if (matrix.header.includes(column)) return;
    matrix.header.push(column);
    matrix.rows.forEach((row) => row.push(""));
  });
}

function rowToObject(header, row) {
  const obj = {};
  header.forEach((column, index) => {
    obj[column] = row[index] ?? "";
  });
  return obj;
}

function updateRowBySujet(matrix, sujet, updater) {
  const sujetIndex = matrix.header.indexOf("Sujet");
  if (sujetIndex === -1) return false;
  let updated = false;
  matrix.rows.forEach((row) => {
    if ((row[sujetIndex] ?? "").trim() !== sujet) return;
    const rowObj = rowToObject(matrix.header, row);
    const patch = updater(rowObj);
    if (!patch || typeof patch !== "object") return;
    Object.entries(patch).forEach(([key, value]) => {
      const index = matrix.header.indexOf(key);
      if (index !== -1) row[index] = value;
    });
    updated = true;
  });
  return updated;
}

function serializeCsv(matrix) {
  const lines = [toCsvLine(matrix.header)];
  matrix.rows.forEach((row) => {
    const normalized = matrix.header.map((_, idx) => row[idx] ?? "");
    lines.push(toCsvLine(normalized));
  });
  return `${lines.join("\n")}\n`;
}

function evaluateEvidence() {
  const g2 = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.g2)));
  const g3a = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.g3Android)));
  const g3i = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.g3Ios)));
  const g4Csv = parseCsvBlock(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.g4)));
  const g5 = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.g5)));
  const g6 = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.g6)));
  const g7 = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.g7)));
  const lifecycle = parseKvMarkdown(
    readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.missionLifecycle))
  );
  const company = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.company)));
  const rollback = parseKvMarkdown(readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.rollback)));
  const dispatchSemantics = parseKvMarkdown(
    readFileSafe(path.join(EVIDENCE_DIR, EVIDENCE_FILES.dispatchSemantics))
  );

  function trackingSideEval(side) {
    const points = parseNumber(side.points_collected);
    const ackRate = parseNumber(side.ack_rate);
    const p95 = parseNumber(side.p95_latency_ms);
    const resultFlag = parseBool(side.result);
    const ready =
      resultFlag === true &&
      points !== null &&
      points >= 30 &&
      ackRate !== null &&
      ackRate >= 0.99 &&
      p95 !== null &&
      p95 <= 8000;
    const hasSignal = resultFlag !== null || points !== null || ackRate !== null || p95 !== null;
    const failedThreshold = hasSignal && !ready;
    return { ready, hasSignal, failedThreshold };
  }

  function boolGroupEval(keys, payload) {
    const values = keys.map((key) => parseBool(payload[key]));
    const ready = values.every((value) => value === true);
    const hasSignal = values.some((value) => value !== null);
    const failedThreshold = hasSignal && values.some((value) => value === false);
    return { ready, hasSignal, failedThreshold };
  }

  const quickActionRuns = g4Csv.filter((row) => row.run_id).length;
  const quickActionSuccess = g4Csv.filter((row) => parseBool(row.result) === true).length;
  const quickActionRate = quickActionRuns > 0 ? quickActionSuccess / quickActionRuns : 0;
  const quickActionReady = quickActionRuns >= 30 && quickActionRate >= 0.95;
  const quickActionHasSignal = quickActionRuns > 0;
  const quickActionFailedThreshold = quickActionHasSignal && !quickActionReady;

  const g2Eval = boolGroupEval(
    [
      "socket_reconnect_lt_5s",
      "no_mission_stale_after_reconnect",
      "event_drop_rate_lte_baseline_w0",
      "polling_fallback_auto_enabled",
      "strict_validation_socket_rest_ui",
    ],
    g2
  );
  const g5Eval = boolGroupEval(
    [
      "no_transition_duplication",
      "transition_order_preserved",
      "no_transition_loss",
      "backend_convergence_lt_10s",
      "convergence_confirmed_via_rest_or_socket",
    ],
    g5
  );
  const g6Eval = boolGroupEval(
    [
      "app_killed_during_mission",
      "session_restored",
      "mission_state_reconciled",
      "queue_flushed",
      "backend_state_converged_lt_10s",
    ],
    g6
  );
  const g7Eval = boolGroupEval(
    [
      "token_expired_mid_mission",
      "refresh_auto_success",
      "switch_company_driver_company_success",
      "crash_free_session",
      "backend_metrics_attached",
    ],
    g7
  );
  const lifecycleEval = boolGroupEval(
    [
      "assigned_to_accepted",
      "accepted_to_started",
      "started_to_completed",
      "completed_to_synced_backend",
      "local_mission_state",
      "backend_canonical_state",
    ],
    lifecycle
  );

  const s1Eval = boolGroupEval(["S1_create_schedule_visible_dispatch"], company);
  const s2Eval = boolGroupEval(["S2_urgent_propagation"], company);
  const s3Eval = boolGroupEval(["S3_cancel_reason_code_note"], company);
  const s4aEval = boolGroupEval(["S4a_realtime_ride_updated"], company);
  const s4bEval = boolGroupEval(["S4b_realtime_ride_cancelled"], company);
  const s4cEval = boolGroupEval(["S4c_realtime_delay_invalidated"], company);
  const s5Eval = boolGroupEval(["S5_socket_reconnect_convergence"], company);
  const s6CompanyEval = boolGroupEval(["S6_dispatch_semantics"], company);
  const s6DispatchEval = boolGroupEval(
    [
      "ui_dispatch_mode",
      "ui_dispatch_state",
      "ui_optimizer_state",
      "socket_payload_consistent_with_rest_snapshot",
      "dashboard_kpi_coherent_with_canonical_backend_state",
    ],
    dispatchSemantics
  );

  const checks = {
    g2StrictValidation: g2Eval.ready,
    trackingAndroidReady: trackingSideEval(g3a).ready,
    trackingIosReady: trackingSideEval(g3i).ready,
    quickActionReady,
    offlineReplayReady: g5Eval.ready,
    resumeReady: g6Eval.ready,
    authRefreshReady: g7Eval.ready,
    missionLifecycleReady: lifecycleEval.ready,
    s1Ready: s1Eval.ready,
    s2Ready: s2Eval.ready,
    s3Ready: s3Eval.ready,
    s4aReady: s4aEval.ready,
    s4bReady: s4bEval.ready,
    s4cReady: s4cEval.ready,
    s5Ready: s5Eval.ready,
    s6SemanticsReady: s6CompanyEval.ready && s6DispatchEval.ready,
  };

  const trackingAndroid = trackingSideEval(g3a);
  const trackingIos = trackingSideEval(g3i);
  const rollbackSeconds = parseNumber(rollback.measured_execution_seconds);
  const rollbackEval = boolGroupEval(
    [
      "feature_flag_off_propagated_lt_30s",
      "legacy_socket_rooms_restored",
      "polling_fallback_active",
      "existing_sessions_not_interrupted",
      "no_active_mission_loss",
      "no_client_crash",
    ],
    rollback
  );
  checks.rollbackReady =
    rollbackEval.ready &&
    rollbackSeconds !== null &&
    rollbackSeconds <= 120;

  checks.g3Ready = checks.trackingAndroidReady && checks.trackingIosReady;
  checks.companyReady =
    checks.s1Ready &&
    checks.s2Ready &&
    checks.s3Ready &&
    checks.s4aReady &&
    checks.s4bReady &&
    checks.s4cReady &&
    checks.s5Ready &&
    checks.s6SemanticsReady;

  return {
    checks,
    metrics: {
      quickActionRuns,
      quickActionSuccess,
      quickActionRate,
      rollbackSeconds,
    },
    signals: {
      g2: g2Eval,
      g3Android: trackingAndroid,
      g3Ios: trackingIos,
      g4: {
        ready: quickActionReady,
        hasSignal: quickActionHasSignal,
        failedThreshold: quickActionFailedThreshold,
      },
      g5: g5Eval,
      g6: g6Eval,
      g7: g7Eval,
      lifecycle: lifecycleEval,
      s1: s1Eval,
      s2: s2Eval,
      s3: s3Eval,
      s4a: s4aEval,
      s4b: s4bEval,
      s4c: s4cEval,
      s5: s5Eval,
      s6Company: s6CompanyEval,
      s6Dispatch: s6DispatchEval,
      rollback: rollbackEval,
    },
  };
}

function applyEvidenceToMatrices(evidence) {
  const { checks, signals } = evidence;
  const driver = readCsvMatrix(DRIVER_MATRIX_PATH);
  const company = readCsvMatrix(COMPANY_MATRIX_PATH);

  ensureColumns(driver, [
    "evidence_link",
    "evidence_date",
    "validated_by",
    "line_id",
    "required_evidence",
    "validation_rule",
    "computed_runtime_proof",
    "computed_gate_result",
    "blocking_scope",
  ]);
  ensureColumns(company, [
    "evidence_link",
    "evidence_date",
    "validated_by",
    "line_id",
    "required_evidence",
    "validation_rule",
    "computed_runtime_proof",
    "computed_gate_result",
    "blocking_scope",
  ]);

  const ruleResults = [];
  const driverRules = [
    {
      sujet: "Mission active",
      lineId: "DRV-P0-MISSION-ACTIVE",
      requiredEvidence: ["mission_lifecycle_convergence.log.md"],
      validationRule: "Transitions mission lifecycle alignees backend",
      blockingScope: "g5,g6",
      strictScope: "driver_p0,g5,g6",
      gate: "gLifecycle",
      ready: checks.missionLifecycleReady,
      signal: signals.lifecycle,
    },
    {
      sujet: "Quick action push",
      lineId: "DRV-P0-QUICK-ACTION",
      requiredEvidence: ["quick_actions_campaign.csv.md"],
      validationRule: "success_rate >= 95% et N >= 30",
      blockingScope: "g4",
      strictScope: "driver_p0,g4",
      gate: "g4",
      ready: checks.quickActionReady,
      signal: signals.g4,
    },
    {
      sujet: "Silent background mission refresh",
      lineId: "DRV-P0-REALTIME-RECONNECT",
      requiredEvidence: ["realtime_reconnect.log.md"],
      validationRule: "socket<5s + no stale + fallback + socket/rest/ui",
      blockingScope: "g2",
      strictScope: "driver_p0,g2",
      gate: "g2",
      ready: checks.g2StrictValidation,
      signal: signals.g2,
    },
    {
      sujet: "Silent background garanties mitigation",
      lineId: "DRV-P0-AUTH-REFRESH",
      requiredEvidence: ["auth_refresh_session.log.md"],
      validationRule: "refresh auto + switch context + crash free",
      blockingScope: "g7",
      strictScope: "driver_p0,g7",
      gate: "g7",
      ready: checks.authRefreshReady,
      signal: signals.g7,
    },
    {
      sujet: "TRACK-cadence-mission",
      lineId: "DRV-P0-TRACK-CADENCE",
      requiredEvidence: ["tracking_background_android.log.md", "tracking_background_ios.log.md"],
      validationRule: "ack_rate >= 99% + p95 <= 8s + points >= 30",
      blockingScope: "g3",
      strictScope: "driver_p0,g3",
      gate: "g3",
      ready: checks.g3Ready,
      signal: {
        ready: checks.g3Ready,
        hasSignal: signals.g3Android.hasSignal || signals.g3Ios.hasSignal,
        failedThreshold: signals.g3Android.failedThreshold || signals.g3Ios.failedThreshold,
      },
    },
    {
      sujet: "TRACK-retry-ack",
      lineId: "DRV-P0-TRACK-ACK",
      requiredEvidence: ["tracking_background_android.log.md", "tracking_background_ios.log.md"],
      validationRule: "ack_rate >= 99% + p95 <= 8s + dual platform",
      blockingScope: "g3",
      strictScope: "driver_p0,g3",
      gate: "g3",
      ready: checks.g3Ready,
      signal: {
        ready: checks.g3Ready,
        hasSignal: signals.g3Android.hasSignal || signals.g3Ios.hasSignal,
        failedThreshold: signals.g3Android.failedThreshold || signals.g3Ios.failedThreshold,
      },
    },
    {
      sujet: "TRACK-offline-persistence",
      lineId: "DRV-P0-OFFLINE-REPLAY",
      requiredEvidence: ["offline_replay_session.log.md"],
      validationRule: "no duplication + no loss + convergence<10s",
      blockingScope: "g5",
      strictScope: "driver_p0,g5",
      gate: "g5",
      ready: checks.offlineReplayReady,
      signal: signals.g5,
    },
    {
      sujet: "TRACK-resume-convergence",
      lineId: "DRV-P0-RESUME-KILL",
      requiredEvidence: ["resume_after_kill.log.md"],
      validationRule: "resume + queue flush + backend convergence<10s",
      blockingScope: "g6",
      strictScope: "driver_p0,g6",
      gate: "g6",
      ready: checks.resumeReady,
      signal: signals.g6,
    },
  ];

  const companyRules = [
    {
      sujet: "S1-create-schedule-visible",
      lineId: "CMP-P0-S1",
      requiredEvidence: ["company_jtbd_execution.md"],
      validationRule: "S1 create/schedule valide",
      blockingScope: "company_p0",
      strictScope: "company_p0",
      gate: "company",
      ready: checks.s1Ready,
      signal: signals.s1,
    },
    {
      sujet: "S2-urgent-propagation",
      lineId: "CMP-P0-S2",
      requiredEvidence: ["company_jtbd_execution.md"],
      validationRule: "S2 urgent propagation valide",
      blockingScope: "company_p0",
      strictScope: "company_p0",
      gate: "company",
      ready: checks.s2Ready,
      signal: signals.s2,
    },
    {
      sujet: "S3-cancel-parity",
      lineId: "CMP-P0-S3",
      requiredEvidence: ["company_jtbd_execution.md"],
      validationRule: "S3 cancel reason_code+note valide",
      blockingScope: "company_p0",
      strictScope: "company_p0",
      gate: "company",
      ready: checks.s3Ready,
      signal: signals.s3,
    },
    {
      sujet: "S4a-realtime-ride-updated",
      lineId: "CMP-P0-S4A",
      requiredEvidence: ["company_jtbd_execution.md"],
      validationRule: "S4a socket + REST + UI coherents",
      blockingScope: "company_p0",
      strictScope: "company_p0",
      gate: "company",
      ready: checks.s4aReady,
      signal: signals.s4a,
    },
    {
      sujet: "S4b-realtime-ride-cancelled",
      lineId: "CMP-P0-S4B",
      requiredEvidence: ["company_jtbd_execution.md"],
      validationRule: "S4b socket + REST + UI coherents",
      blockingScope: "company_p0",
      strictScope: "company_p0",
      gate: "company",
      ready: checks.s4bReady,
      signal: signals.s4b,
    },
    {
      sujet: "S4c-realtime-delay-invalidated",
      lineId: "CMP-P0-S4C",
      requiredEvidence: ["company_jtbd_execution.md"],
      validationRule: "S4c socket + REST + UI coherents",
      blockingScope: "company_p0",
      strictScope: "company_p0",
      gate: "company",
      ready: checks.s4cReady,
      signal: signals.s4c,
    },
    {
      sujet: "S5-reconnect-recovery",
      lineId: "CMP-P0-S5",
      requiredEvidence: ["company_jtbd_execution.md", "realtime_reconnect.log.md"],
      validationRule: "S5 reconnect convergence valide",
      blockingScope: "company_p0",
      strictScope: "company_p0,g2",
      gate: "company",
      ready: checks.s5Ready,
      signal: signals.s5,
    },
    {
      sujet: "S6-dispatch-semantics",
      lineId: "CMP-P0-S6",
      requiredEvidence: ["company_jtbd_execution.md", "dispatch_semantics_validation.md"],
      validationRule: "dispatch_mode/state/optimizer coherents sur 4 couches",
      blockingScope: "company_p0",
      strictScope: "company_p0",
      gate: "company",
      ready: checks.s6SemanticsReady,
      signal: {
        ready: checks.s6SemanticsReady,
        hasSignal: signals.s6Company.hasSignal || signals.s6Dispatch.hasSignal,
        failedThreshold: signals.s6Company.failedThreshold || signals.s6Dispatch.failedThreshold,
      },
    },
  ];

  function applyRuleToMatrix(matrix, rule) {
    updateRowBySujet(matrix, rule.sujet, (row) => {
      const computedRuntimeProof = rule.ready ? "scenario-validated" : "pending-evidence";
      const computedGateResult = rule.ready
        ? "ready-for-signature"
        : rule.signal.failedThreshold
          ? "failed-threshold"
          : rule.signal.hasSignal
            ? "pending-evidence"
            : "missing-proof";
      const computedStatus = rule.ready
        ? "ready-for-signature"
        : rule.signal.failedThreshold
          ? "failed-threshold"
          : rule.signal.hasSignal
            ? "pending-evidence"
            : "missing-proof";
      ruleResults.push({
        sujet: rule.sujet,
        lineId: rule.lineId,
        gate: rule.gate,
        ready: rule.ready,
        blockingScope: rule.blockingScope,
        strictScope: rule.strictScope,
        computedStatus,
      });
      return {
        line_id: rule.lineId,
        required_evidence: rule.requiredEvidence.join(";"),
        validation_rule: rule.validationRule,
        computed_runtime_proof: computedRuntimeProof,
        computed_gate_result: computedGateResult,
        blocking_scope: rule.blockingScope,
        evidence_link: rule.ready
          ? `docs/migration/phase1_evidence/${rule.requiredEvidence[0]}`
          : row.evidence_link ?? "",
        evidence_date: rule.ready ? TODAY : row.evidence_date ?? "",
        runtime_proof: rule.ready ? "scenario-validated" : row.runtime_proof,
        "Resultat gate": rule.ready ? "Ready for signature" : row["Resultat gate"],
        Statut: rule.ready ? "Ready for signature" : row.Statut,
      };
    });
  }

  driverRules.forEach((rule) => applyRuleToMatrix(driver, rule));
  companyRules.forEach((rule) => applyRuleToMatrix(company, rule));

  ruleResults.push({
    sujet: "Rollback drill",
    lineId: "OPS-P0-ROLLBACK",
    gate: "rollback",
    ready: checks.rollbackReady,
    blockingScope: "rollback",
    strictScope: "rollback",
    computedStatus: checks.rollbackReady
      ? "ready-for-signature"
      : signals.rollback.failedThreshold
        ? "failed-threshold"
        : signals.rollback.hasSignal
          ? "pending-evidence"
          : "missing-proof",
  });

  return { driver, company, ruleResults };
}

function buildPhase1AuthDoc(evaluation, ruleResults) {
  const { checks } = evaluation;
  function line(label, ready) {
    return `- ${label}: ${ready ? "READY" : "NOT_READY"}`;
  }

  const gateMap = {
    g2: "G2 realtime reconnect",
    g3: "G3 background tracking",
    g4: "G4 quick actions push",
    g5: "G5 offline replay",
    g6: "G6 resume-after-kill",
    g7: "G7 auth refresh",
    gLifecycle: "Mission lifecycle convergence",
    company: "Company JTBD S1-S6",
  };

  const gateStatus = {};
  Object.keys(gateMap).forEach((gateKey) => {
    const gateLines = ruleResults.filter((lineResult) => lineResult.gate === gateKey);
    gateStatus[gateKey] = gateLines.length > 0 && gateLines.every((lineResult) => lineResult.ready);
  });

  const blockingLines = ruleResults.filter((lineResult) => !lineResult.ready);
  const p0Lines = ruleResults.filter((lineResult) =>
    lineResult.strictScope.split(",").includes("driver_p0") ||
    lineResult.strictScope.split(",").includes("company_p0")
  );
  const p0Ready = p0Lines.filter((lineResult) => lineResult.ready).length;
  const p0Blocked = p0Lines.length - p0Ready;
  const gatesReady = Object.values(gateStatus).filter(Boolean).length;
  const gatesBlocked = Object.values(gateStatus).length - gatesReady + (checks.rollbackReady ? 0 : 1);

  const phase2Ready =
    gateStatus.g2 &&
    gateStatus.g3 &&
    gateStatus.g4 &&
    gateStatus.g5 &&
    gateStatus.g6 &&
    gateStatus.g7 &&
    gateStatus.gLifecycle &&
    gateStatus.company &&
    checks.rollbackReady;

  const blockerSection =
    blockingLines.length === 0
      ? "- Aucun blocage ligne-a-ligne."
      : blockingLines
          .map(
            (lineResult) =>
              `- ${lineResult.lineId} (${lineResult.sujet}) [${lineResult.blockingScope}] -> pending evidence`
          )
          .join("\n");

  return `# Phase 1 Close Authorization

Date: ${TODAY}
Source: docs/migration/phase1_evidence/

## Gates status

${line(gateMap.g2, gateStatus.g2)}
${line(gateMap.g3, gateStatus.g3)}
${line(gateMap.g4, gateStatus.g4)}
${line(gateMap.g5, gateStatus.g5)}
${line(gateMap.g6, gateStatus.g6)}
${line(gateMap.g7, gateStatus.g7)}
${line(gateMap.gLifecycle, gateStatus.gLifecycle)}
${line(gateMap.company, gateStatus.company)}
${line("Rollback drill <= 120s", checks.rollbackReady)}

## Line-level blockers

${blockerSection}

## Decision

- Authorized to start Phase 2: ${phase2Ready ? "YES" : "NO"}
- Overall status: ${phase2Ready ? "READY_FOR_SIGNATURE" : "HOLD_PENDING_FIELD_EVIDENCE"}

## Summary

- P0 lines: ${p0Lines.length} total / ${p0Ready} ready / ${p0Blocked} blocked
- Gates: ${gatesReady + (checks.rollbackReady ? 1 : 0)} ready / ${gatesBlocked} blocked

## Signatures

- Mobile Lead:
- Backend Lead:
- QA Device Owner:
- Product:
- Ops:
`;
}

function buildBlockersDoc(ruleResults) {
  const blocked = ruleResults.filter((lineResult) => !lineResult.ready);
  const body =
    blocked.length === 0
      ? "| line_id | sujet | blocking_scope | strict_scope | status |\n|---|---|---|---|---|\n| none | none | none | none | READY |\n"
      : [
          "| line_id | sujet | blocking_scope | strict_scope | status |",
          "|---|---|---|---|---|",
          ...blocked.map(
            (lineResult) =>
              `| ${lineResult.lineId} | ${lineResult.sujet} | ${lineResult.blockingScope} | ${lineResult.strictScope} | ${lineResult.computedStatus} |`
          ),
        ].join("\n");

  return `# Phase 1 Committee Blockers

Date: ${TODAY}

## Blocking lines

${body}
`;
}

function computeStrictFailures(ruleResults, overallStatus, signaturesFilled) {
  const strictScopes = new Set([
    "driver_p0",
    "company_p0",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "g7",
    "rollback",
  ]);
  const blockingStatuses = new Set([
    "pending-evidence",
    "failed-threshold",
    "missing-proof",
    "blocked",
  ]);

  const strictBlocking = ruleResults.filter((lineResult) => {
    const scopes = String(lineResult.strictScope || "")
      .split(",")
      .map((scope) => scope.trim())
      .filter(Boolean);
    const inScope = scopes.some((scope) => strictScopes.has(scope));
    return inScope && blockingStatuses.has(lineResult.computedStatus);
  });

  const failures = [];
  if (strictBlocking.length > 0) {
    failures.push({
      type: "blocking_lines",
      lines: strictBlocking,
    });
  }

  if (STRICT_RELEASE) {
    const authorizedStatus = new Set(["READY_FOR_SIGNATURE", "SIGNED"]);
    if (!authorizedStatus.has(overallStatus)) {
      failures.push({
        type: "authorization_status",
        status: overallStatus,
      });
    }
    if (!signaturesFilled) {
      failures.push({
        type: "missing_signatures",
      });
    }
  }

  return failures;
}

function main() {
  if (!fs.existsSync(EVIDENCE_DIR)) {
    console.error("phase1_evidence folder not found.");
    process.exit(1);
  }

  const evaluation = evaluateEvidence();
  const { driver, company, ruleResults } = applyEvidenceToMatrices(evaluation);
  const authDoc = buildPhase1AuthDoc(evaluation, ruleResults);
  const blockersDoc = buildBlockersDoc(ruleResults);
  const overallStatus = authDoc.includes("Overall status: READY_FOR_SIGNATURE")
    ? "READY_FOR_SIGNATURE"
    : authDoc.includes("Overall status: SIGNED")
      ? "SIGNED"
      : "HOLD_PENDING_FIELD_EVIDENCE";
  const signaturesFilled = false;
  const strictFailures = computeStrictFailures(ruleResults, overallStatus, signaturesFilled);

  if (DRY_RUN) {
    console.log("Phase 1 committee hardening dry-run");
    console.log(JSON.stringify({ checks: evaluation.checks, metrics: evaluation.metrics }, null, 2));
    console.log("Blocking lines:");
    ruleResults
      .filter((lineResult) => !lineResult.ready)
      .forEach((lineResult) => console.log(`- ${lineResult.lineId} (${lineResult.sujet})`));
    if (STRICT && strictFailures.length > 0) {
      console.log("STRICT MODE WOULD FAIL");
    }
    return;
  }

  fs.writeFileSync(DRIVER_MATRIX_PATH, serializeCsv(driver));
  fs.writeFileSync(COMPANY_MATRIX_PATH, serializeCsv(company));
  fs.writeFileSync(PHASE1_AUTH_PATH, authDoc);
  fs.writeFileSync(PHASE1_BLOCKERS_PATH, blockersDoc);

  console.log("Phase 1 committee hardening completed.");
  if (STRICT && strictFailures.length > 0) {
    console.error("");
    console.error("STRICT MODE FAILED");
    console.error("");
    const lineFailure = strictFailures.find((failure) => failure.type === "blocking_lines");
    if (lineFailure) {
      console.error("Blocking lines:");
      lineFailure.lines.forEach((lineResult) => {
        console.error(
          `- ${lineResult.blockingScope.toUpperCase()} / ${lineResult.sujet} / ${lineResult.computedStatus}`
        );
      });
    }
    if (STRICT_RELEASE) {
      if (strictFailures.some((failure) => failure.type === "authorization_status")) {
        console.error(`- RELEASE / phase1 authorization status / ${overallStatus}`);
      }
      if (strictFailures.some((failure) => failure.type === "missing_signatures")) {
        console.error("- RELEASE / signatures / missing-signatures");
      }
    }
    console.error("");
    console.error("See:");
    console.error("docs/migration/PHASE1_COMMITTEE_BLOCKERS.md");
    console.error("docs/migration/PHASE1_CLOSE_AUTHORIZATION.md");
    process.exit(1);
  }
}

main();

