export type GovernanceLot = "P0" | "P1" | "P2" | "P3" | "P4" | "Transverse";

export type GovernanceProgramStatus =
  | "cadrage"
  | "exécution"
  | "validation"
  | "pré_décommission"
  | "décision"
  | "décommissionnée"
  | "rollback_programme";

export type GovernanceLotMaturity =
  | "non_démarré"
  | "contrat_à_figer"
  | "implémentation"
  | "shadow"
  | "cohorte"
  | "gate_en_validation"
  | "validé_lot"
  | "rollback_only";

export type GovernanceMetricDefinition = {
  key: string;
  lot: GovernanceLot;
  label: string;
  formula: string;
  sourceOfTruth: string;
  measurementWindow: string;
  threshold: string;
  requiresProdControlledProof: boolean;
};

export const baselineMetrics: GovernanceMetricDefinition[] = [
  {
    key: "tracking_continuity_over_30m",
    lot: "P0",
    label: "Continuite tracking missions > 30 min",
    formula: "points_acked / points_eligible",
    sourceOfTruth: "driver_telemetry + backend ingest counters",
    measurementWindow: "14 jours glissants",
    threshold: ">= 99.5% et 0 perte silencieuse sur scenarios critiques",
    requiresProdControlledProof: true,
  },
  {
    key: "background_resume_success",
    lot: "P2",
    label: "Reprise session/background sans logout involontaire",
    formula: "1 - (logout_involontaires / sessions_resume)",
    sourceOfTruth: "auth logs + runtime resume telemetry",
    measurementWindow: "14 jours glissants",
    threshold: ">= baseline operations-app ; cible 100%",
    requiresProdControlledProof: true,
  },
  {
    key: "push_open_route_success",
    lot: "P1",
    label: "Ouverture mission via push (fg/bg/cold)",
    formula: "opens_to_expected_mission / push_opens",
    sourceOfTruth: "push telemetry received/opened/routed + mission detail fetch",
    measurementWindow: "7 jours glissants",
    threshold: ">= 99.0% sur scenarios critiques",
    requiresProdControlledProof: true,
  },
  {
    key: "mission_drift_persistent",
    lot: "P4",
    label: "Divergence mission persistante apres rattrapage",
    formula: "sessions_with_persistent_drift / sessions_with_reconnect",
    sourceOfTruth: "realtime.drift.detected + reconcile correlation backend",
    measurementWindow: "7 jours glissants",
    threshold: "0 sur scenarios critiques",
    requiresProdControlledProof: true,
  },
  {
    key: "offline_transition_integrity",
    lot: "P3",
    label: "Transitions offline sans double/perte",
    formula: "1 - ((double_transitions + lost_transitions) / transitions_total)",
    sourceOfTruth: "offline queue logs + backend idempotency outcomes",
    measurementWindow: "7 jours glissants",
    threshold: "100% sur scenarios de validation",
    requiresProdControlledProof: true,
  },
];

export type GovernanceContract = {
  lot: GovernanceLot;
  contractName: string;
  ownerContract: string;
  ownerImplementation: string;
  ownerValidationGate: string;
  targetSignatureDate: string;
  impactIfUnsigned: string;
};

export const governanceContracts: GovernanceContract[] = [
  {
    lot: "P0",
    contractName: "Tracking ingest/ACK and silent-loss definition",
    ownerContract: "Backend Lead",
    ownerImplementation: "Mobile Driver Lead",
    ownerValidationGate: "Quality Engineering Lead",
    targetSignatureDate: "2026-04-23",
    impactIfUnsigned: "P0 remains blocked in contrat_à_figer.",
  },
  {
    lot: "P1",
    contractName: "Driver push payloads and mission routing",
    ownerContract: "Product Driver PM",
    ownerImplementation: "Mobile Driver Lead",
    ownerValidationGate: "Operations QA Lead",
    targetSignatureDate: "2026-04-25",
    impactIfUnsigned: "Push lot cannot reach validé_lot.",
  },
  {
    lot: "P2",
    contractName: "Session invalidation, refresh and resume lifecycle",
    ownerContract: "Auth Platform Lead",
    ownerImplementation: "Mobile Platform Lead",
    ownerValidationGate: "SRE Runtime Lead",
    targetSignatureDate: "2026-04-24",
    impactIfUnsigned: "Resume orchestration remains non-certifiable.",
  },
  {
    lot: "P3",
    contractName: "Mission conflict, stale and reassignment policy",
    ownerContract: "Dispatch Product Lead",
    ownerImplementation: "Mobile Driver Lead",
    ownerValidationGate: "Operations QA Lead",
    targetSignatureDate: "2026-04-26",
    impactIfUnsigned: "Mission manager conflict rules are not opposable.",
  },
  {
    lot: "P4",
    contractName: "Realtime ordering contract and persistent drift definition",
    ownerContract: "Realtime Platform Lead",
    ownerImplementation: "Mobile Driver Lead",
    ownerValidationGate: "SRE Runtime Lead",
    targetSignatureDate: "2026-04-28",
    impactIfUnsigned: "P4 soak gate remains contestable.",
  },
];

export type GovernanceRegisterEntry = {
  line: number;
  lot: GovernanceLot;
  decision: string;
  owner: string;
  targetDate: string;
  decisionDate: string | null;
  fixedValue: string;
  evidenceLink: string;
};

export const placeholderRegister: GovernanceRegisterEntry[] = [
  {
    line: 0,
    lot: "Transverse",
    decision: "Version du document maitre",
    owner: "Program Manager",
    targetDate: "2026-04-16",
    decisionDate: "2026-04-16",
    fixedValue: "v1.0.0",
    evidenceLink: "runtimeGovernance/programGovernance.ts#L1",
  },
  {
    line: 1,
    lot: "P4",
    decision: "Drift persistant X/Y/Z",
    owner: "Realtime Platform Lead",
    targetDate: "2026-04-28",
    decisionDate: "2026-04-16",
    fixedValue: "X=120s; Y=3 resync cycles; Z=5 reconnect attempts",
    evidenceLink: "runtimeGovernance/programGovernance.ts#L140",
  },
  {
    line: 2,
    lot: "P0",
    decision: "Seuils tracking T/M/K/ack",
    owner: "Mobile Driver Lead",
    targetDate: "2026-04-24",
    decisionDate: "2026-04-16",
    fixedValue: "T=5m; M=120 missions; K=60 runs tunnel; ack>=99.5%",
    evidenceLink: "runtimeGovernance/programGovernance.ts#L26",
  },
  {
    line: 3,
    lot: "Transverse",
    decision: "Tailles minimales campagnes P1-P4",
    owner: "Quality Engineering Lead",
    targetDate: "2026-04-30",
    decisionDate: "2026-04-16",
    fixedValue: "P1>=300 pushes; P2>=500 resumes; P3>=120 suites; P4>=7 jours soak",
    evidenceLink: "runtimeGovernance/programGovernance.ts#L26",
  },
  {
    line: 4,
    lot: "Transverse",
    decision: "Liste scenarios critiques baseline",
    owner: "Program Manager",
    targetDate: "2026-04-22",
    decisionDate: "2026-04-16",
    fixedValue:
      "mission_longue_30m,tunnel_network,background_long_resume,push_cold_start,reconnect_cascade,offline_relaunch_replay",
    evidenceLink: "runtimeGovernance/programGovernance.ts#L26",
  },
  {
    line: 5,
    lot: "Transverse",
    decision: "Seuils rollback auto vs manuel",
    owner: "Incident Commander",
    targetDate: "2026-04-22",
    decisionDate: "2026-04-16",
    fixedValue: "auto_if_ack<98%_5m_or_crash>2%; else manual",
    evidenceLink: "runtimeGovernance/programGovernance.ts#L286",
  },
  {
    line: 6,
    lot: "P0",
    decision: "Definition perte silencieuse",
    owner: "Backend Lead",
    targetDate: "2026-04-23",
    decisionDate: "2026-04-16",
    fixedValue:
      "point eligible absent de ingest counters sans etat dropped/expired trace cote client",
    evidenceLink: "runtimeGovernance/programGovernance.ts#L26",
  },
];

export type GovernanceCadence = {
  rhythm: "weekly" | "bi_weekly" | "on_demand";
  objective: string;
  requiredInputs: string[];
};

export const governanceCadence: GovernanceCadence[] = [
  {
    rhythm: "weekly",
    objective: "Revue maturite P0-P4 et blocages actifs",
    requiredInputs: ["lot maturity board", "open blockers", "owner updates"],
  },
  {
    rhythm: "bi_weekly",
    objective: "Revue registre + contrats + annexe gates",
    requiredInputs: ["placeholder register", "contracts signatures", "gate evidence links"],
  },
  {
    rhythm: "on_demand",
    objective: "Gate workshop cible (ne remplace pas la bi-hebdo)",
    requiredInputs: ["gate template", "metric extracts", "rollback runbook excerpt"],
  },
];

export type GovernanceRollbackSheet = {
  lot: GovernanceLot;
  triggerSignal: string;
  maxRollbackDurationMinutes: number;
  requiresStoreRelease: boolean;
  userImpact: string;
  decisionOwner: string;
};

export const rollbackSheets: GovernanceRollbackSheet[] = [
  {
    lot: "P0",
    triggerSignal: "tracking_continuity_over_30m < 98% sur 5 minutes",
    maxRollbackDurationMinutes: 10,
    requiresStoreRelease: false,
    userImpact: "tracking degrade vers HTTP legacy sans interruption mission",
    decisionOwner: "Incident Commander",
  },
  {
    lot: "P2",
    triggerSignal: "logout involontaires > 0 sur campagne resume controlee",
    maxRollbackDurationMinutes: 15,
    requiresStoreRelease: false,
    userImpact: "resume orchestre desactive, fallback refresh reactif 401",
    decisionOwner: "Incident Commander",
  },
  {
    lot: "P4",
    triggerSignal: "drift persistant observe sur scenario critique",
    maxRollbackDurationMinutes: 15,
    requiresStoreRelease: false,
    userImpact: "realtime degrade vers polling prioritaire",
    decisionOwner: "Incident Commander",
  },
];

export type GovernanceGateTemplate = {
  lot: GovernanceLot;
  metric: string;
  sourceOfTruth: string;
  threshold: string;
  measurementWindow: string;
  proofLevel: "lab_only" | "lab_and_prod_controlled";
  ownerValidationGate: string;
  rollbackReference: string;
};

export const gateTemplates: GovernanceGateTemplate[] = [
  {
    lot: "P0",
    metric: "tracking_continuity_over_30m",
    sourceOfTruth: "driver_telemetry + backend ingest counters",
    threshold: ">=99.5%, no silent loss on critical scenarios",
    measurementWindow: "14 days rolling + tunnel campaign",
    proofLevel: "lab_and_prod_controlled",
    ownerValidationGate: "Quality Engineering Lead",
    rollbackReference: "rollbackSheets.P0",
  },
  {
    lot: "P1",
    metric: "push_open_route_success",
    sourceOfTruth: "push opened/routed telemetry",
    threshold: ">=99.0% on critical scenarios",
    measurementWindow: "7 days rolling + payload matrix test suite",
    proofLevel: "lab_and_prod_controlled",
    ownerValidationGate: "Operations QA Lead",
    rollbackReference: "driver_push_enabled",
  },
  {
    lot: "P2",
    metric: "background_resume_success",
    sourceOfTruth: "runtime resume telemetry + auth logs",
    threshold: "no involuntary logout on critical scenarios",
    measurementWindow: "7 days rolling + resume campaign",
    proofLevel: "lab_and_prod_controlled",
    ownerValidationGate: "SRE Runtime Lead",
    rollbackReference: "rollbackSheets.P2",
  },
  {
    lot: "P3",
    metric: "offline_transition_integrity",
    sourceOfTruth: "offline queue telemetry + backend idempotency",
    threshold: "100% on conflict simulation suite",
    measurementWindow: "120 conflict scenarios",
    proofLevel: "lab_and_prod_controlled",
    ownerValidationGate: "Operations QA Lead",
    rollbackReference: "serialized_transition_policy",
  },
  {
    lot: "P4",
    metric: "mission_drift_persistent",
    sourceOfTruth: "realtime drift telemetry + backend canonical state",
    threshold: "0 on critical reconnect scenarios",
    measurementWindow: "7 days soak",
    proofLevel: "lab_and_prod_controlled",
    ownerValidationGate: "SRE Runtime Lead",
    rollbackReference: "rollbackSheets.P4",
  },
];

export type PreDecommissionChecklistEntry = {
  key: string;
  mandatory: boolean;
  description: string;
};

export const preDecommissionChecklist: PreDecommissionChecklistEntry[] = [
  {
    key: "controlled_prod_cohort",
    mandatory: true,
    description: "Cohorte production limitee active et observee sur fenetre complete.",
  },
  {
    key: "rollback_exercise_complete",
    mandatory: true,
    description: "Rollback P0/P2/P4 execute avec delais max respectes.",
  },
  {
    key: "runbook_ready",
    mandatory: true,
    description: "Runbook incident valide et partage (decisionnaire + communication).",
  },
  {
    key: "observability_live_review",
    mandatory: true,
    description: "Dashboards, alertes et correlation IDs verifies.",
  },
  {
    key: "ops_support_signoff",
    mandatory: true,
    description: "Validation produit/ops/support recueillie et datee.",
  },
];

export function canOpenGoNoGoCommittee(input: {
  lots: Record<"P0" | "P1" | "P2" | "P3" | "P4", GovernanceLotMaturity>;
  allRegisterLinesClosed: boolean;
  preDecommissionChecklistCompleted: boolean;
}): boolean {
  const allLotsValidated = (["P0", "P1", "P2", "P3", "P4"] as const).every(
    (lot) => input.lots[lot] === "validé_lot"
  );
  return (
    allLotsValidated && input.allRegisterLinesClosed && input.preDecommissionChecklistCompleted
  );
}

// ─── Migration Program Gates (source de vérité — Annexe A) ──────────────────
//
// Ces gates sont les points de décision Go/No-Go du programme de migration
// operations-app → unified-app. Elles sont PLUS HAUTES que les gates de lot
// (P0–P4) et conditionnent les décisions de tracks A/B/C.
//
// Règle de mise à jour : tout changement de seuil ou d'owner se fait ICI
// (pas dans un tableau Notion/Word séparé). Le comité lit ce fichier comme
// source de vérité à chaque réunion bi-hebdo.

export type MigrationTrack = "A" | "B" | "C";

export type MigrationGateStatus =
  | "pending"          // pas encore évaluable (prérequis manquants)
  | "in_progress"      // mesures en cours
  | "ready_for_vote"   // toutes les métriques réunies, vote comité requis
  | "passed"           // gate franchie (décision datée)
  | "blocked"          // bloquée explicitement (incident ou manque contrat)
  | "rolled_back";     // gate franchie puis rollback déclenché

export type MigrationGate = {
  /** Identifiant canonique — utilisé dans les outils de governance et les runbooks. */
  id: string;
  /** Track du plan de migration concerné. */
  track: MigrationTrack;
  /** Description courte lisible par le comité. */
  description: string;
  /**
   * Lots P0–P4 qui doivent être en statut `validé_lot` avant que la gate
   * puisse être soumise au vote.
   */
  requiredLots: ("P0" | "P1" | "P2" | "P3" | "P4")[];
  /**
   * Critères additionnels non couverts par les lots (contrats backend,
   * KPIs prod, fenêtres de soak, UX signoff...).
   */
  additionalCriteria: string[];
  /** Owner unique responsable de présenter les preuves au comité. */
  owner: string;
  /** Date cible — format ISO 8601. "TBD" si non encore fixée. */
  targetDate: string;
  /** Budget de rollback maximal en minutes (référence pour le runbook). */
  rollbackBudgetMinutes: number;
  /** Statut courant de la gate. Mis à jour manuellement après chaque comité. */
  status: MigrationGateStatus;
  /**
   * Date de la décision (ISO 8601) si status === "passed" | "rolled_back".
   * null sinon.
   */
  decisionDate: string | null;
  /** Lien vers les preuves / dashboard / runbook de cette gate. */
  evidenceLink: string;
};

/**
 * Registre des gates de migration du programme.
 *
 * Ordre : Track A (chauffeur) → Track B (entreprise) → Track C (remplacement).
 * À l'intérieur d'un track : ordre chronologique.
 */
export const migrationGates: MigrationGate[] = [
  // ── Track A — Chauffeur ────────────────────────────────────────────────────
  {
    id: "DRIVER_SHADOW_READY",
    track: "A",
    description:
      "Le runtime chauffeur unified est éligible au shadow parallèle en production. " +
      "Tous les lots d'infrastructure (P0 tracking, P2 auth recovery) sont validés en lab. " +
      "Ingest HTTP local opérationnel, replay window 24 h, ACK staleness 75 s actifs.",
    requiredLots: ["P0", "P2"],
    additionalCriteria: [
      "Contrat backend driver_unified_enabled par cohorte signé (bootstrap.feature_flags format validé)",
      "Ingest adapter wired via MonitoringProvider (driverTelemetry sink)",
      "Auth exhaustion 5 tentatives + codes terminaux implémentés dans realtimeManager",
      "Replay window driverTrackingQueue = 24 h (vs 45 min ops)",
      "ACK staleness driverTrackingBridge SOCKET_STALE_MS = 75 s",
      "UX écran blocked contextualisée (reason=driver_gate) validée par produit/ops",
      "DriverUnifiedGateGuard actif dans /(app)/(driver)/_layout.tsx",
    ],
    owner: "Mobile Driver Lead",
    targetDate: "TBD",
    rollbackBudgetMinutes: 10,
    status: "in_progress",
    decisionDate: null,
    evidenceLink: "src/features/driver/runtimeGovernance/programGovernance.ts#DRIVER_SHADOW_READY",
  },
  {
    id: "DRIVER_PILOT_READY",
    track: "A",
    description:
      "Le runtime chauffeur unified est éligible au pilot contrôlé (cohorte réelle). " +
      "Tous les lots P0–P4 validés, KPIs shadow mesurés sur fenêtre de soak, " +
      "rollback < 15 min démontré lors d'un exercice documenté.",
    requiredLots: ["P0", "P1", "P2", "P3", "P4"],
    additionalCriteria: [
      "Shadow shadow_ready maintenu >= 14 jours sans régression",
      "tracking_continuity_over_30m >= 99.5% sur fenêtre shadow",
      "background_resume_success = 100% sur campagne contrôlée",
      "push_open_route_success >= 99.0% sur scenarios critiques",
      "mission_drift_persistent = 0 sur scenarios reconnect critiques",
      "Rollback exercice documenté : flag désactivé en < 15 min sans perte de données en vol",
      "Option A (store séparé ou deep link) disponible avant ce vote ou décision explicite de différer avec plan de mitigation",
      "Baseline ops collectée sur 14 jours (critère Track C)",
    ],
    owner: "Mobile Driver Lead",
    targetDate: "TBD",
    rollbackBudgetMinutes: 15,
    status: "pending",
    decisionDate: null,
    evidenceLink: "src/features/driver/runtimeGovernance/programGovernance.ts#DRIVER_PILOT_READY",
  },

  // ── Track B — Entreprise ───────────────────────────────────────────────────
  {
    id: "COMPANY_SHADOW_READY",
    track: "B",
    description:
      "La carte live entreprise unified est éligible au shadow parallèle. " +
      "La logique observability_only est implémentée dans useCompanyDriverLiveTracking " +
      "et validée par les tests d'anti-régression.",
    requiredLots: [],
    additionalCriteria: [
      "accepted_observability_only dans CompanyDriverLiveLocation (contracts.ts)",
      "shouldReplaceDriverLocation rejette les positions obs_only plus anciennes que la position live courante",
      "Tests d'anti-régression observability_only passent (5 cas)",
      "company_dispatch désactivé en prod (pas EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH) — activable en dev (registry + template)",
      "DriverUnifiedGateGuard ne couvre PAS les routes company (tracks indépendants)",
      "Silence detection 120 s et machine d'état companyRealtimeBridge validées en integration",
    ],
    owner: "Mobile Company Lead",
    targetDate: "TBD",
    rollbackBudgetMinutes: 10,
    status: "in_progress",
    decisionDate: null,
    evidenceLink: "src/features/driver/runtimeGovernance/programGovernance.ts#COMPANY_SHADOW_READY",
  },
  {
    id: "COMPANY_PILOT_READY",
    track: "B",
    description:
      "La carte live entreprise unified est éligible au pilot contrôlé. " +
      "Métriques shadow comparées à ops, aucune régression observability_only observée " +
      "sur la fenêtre de soak.",
    requiredLots: [],
    additionalCriteria: [
      "COMPANY_SHADOW_READY passée",
      "Carte live unified >= ops sur fenêtre 14 jours shadow : fraîcheur positions, taux de mise à jour, aucun écrasement obs_only",
      "Aucun incident driver_location_regression rapporté sur la fenêtre shadow",
      "company_dispatch reste off en build pilot (aucun EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH côté release)",
      "Rollback exercice documenté : désactivation shadow en < 10 min",
    ],
    owner: "Mobile Company Lead",
    targetDate: "TBD",
    rollbackBudgetMinutes: 10,
    status: "pending",
    decisionDate: null,
    evidenceLink: "src/features/driver/runtimeGovernance/programGovernance.ts#COMPANY_PILOT_READY",
  },

  // ── Track C — Remplacement ─────────────────────────────────────────────────
  {
    id: "OPS_REPLACEMENT_READY",
    track: "C",
    description:
      "Le remplacement de operations-app par unified-app est autorisé. " +
      "Toutes les gates Track A et Track B sont passées. " +
      "La baseline ops est collectée et les métriques unified sont supérieures ou égales.",
    requiredLots: ["P0", "P1", "P2", "P3", "P4"],
    additionalCriteria: [
      "DRIVER_PILOT_READY passée",
      "COMPANY_PILOT_READY passée",
      "Baseline ops collectée sur 14 jours prod avant début phase shadow (métriques backend LIRIE)",
      "Comparaison baseline ops vs unified documentée et validée par comité",
      "preDecommissionChecklist complète (5 items mandatory)",
      "Runbook décommission ops validé et partagé",
      "Support/ops/produit signoff daté",
      "Plan de communication chauffeurs et entreprises validé",
    ],
    owner: "Program Manager",
    targetDate: "TBD",
    rollbackBudgetMinutes: 60,
    status: "pending",
    decisionDate: null,
    evidenceLink: "src/features/driver/runtimeGovernance/programGovernance.ts#OPS_REPLACEMENT_READY",
  },
];

/**
 * Vérifie si une gate de migration est passable étant donné l'état courant
 * des lots et des critères additionnels non automatisables.
 *
 * Les critères additionnels (contrats, fenêtres de soak, UX signoff...) ne
 * peuvent pas être vérifiés automatiquement : ils sont laissés à la
 * responsabilité de l'owner et du comité.
 *
 * @returns `true` si les lots requis sont tous en `validé_lot` et que le
 * statut courant de la gate permet le vote (`in_progress` | `ready_for_vote`).
 */
export function canSubmitMigrationGateForVote(
  gateId: string,
  lotMaturities: Partial<Record<"P0" | "P1" | "P2" | "P3" | "P4", GovernanceLotMaturity>>
): { eligible: boolean; blockers: string[] } {
  const gate = migrationGates.find((g) => g.id === gateId);
  if (!gate) return { eligible: false, blockers: [`Gate inconnue : ${gateId}`] };

  const blockers: string[] = [];

  if (gate.status === "passed") {
    return { eligible: false, blockers: ["Gate déjà passée."] };
  }
  if (gate.status === "blocked" || gate.status === "rolled_back") {
    blockers.push(`Gate en statut "${gate.status}" — déblocage manuel requis avant vote.`);
  }

  for (const lot of gate.requiredLots) {
    const maturity = lotMaturities[lot];
    if (maturity !== "validé_lot") {
      blockers.push(
        `Lot ${lot} doit être "validé_lot" (statut courant : "${maturity ?? "inconnu"}").`
      );
    }
  }

  return { eligible: blockers.length === 0, blockers };
}

/**
 * Retourne un résumé de l'avancement du programme de migration
 * par track et par statut de gate.
 */
export function migrationProgramSummary(): {
  track: MigrationTrack;
  gates: { id: string; status: MigrationGateStatus; owner: string; targetDate: string }[];
}[] {
  const tracks: MigrationTrack[] = ["A", "B", "C"];
  return tracks.map((track) => ({
    track,
    gates: migrationGates
      .filter((g) => g.track === track)
      .map(({ id, status, owner, targetDate }) => ({ id, status, owner, targetDate })),
  }));
}
