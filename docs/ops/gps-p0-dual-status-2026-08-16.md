# Statuts figes — GPS P0 release (2026-08-16)

```text
GPS P0 / BACKEND / LEDGER / OPS (P0-A/B)
= CLOSED / PASS

D4-B
= CONFIRMED ✅ (hors cause cut mobile restant)
= HOT-PATCH LIVE = ROLLBACK
BACKEND PROD = GELÉ 286737a2 (READ-ONLY)

═══════════════════════════════════════
P0-D / D5 — BASELINE OFFICIELLE
═══════════════════════════════════════

T_FAIL = 2026-08-16 21:18:49.975 +02
LAST KNOWN GOOD = 21:18:44.491 Finished

D5-A IMMEDIATE CAUSE        = CLOSED ✅
NATIVE UNREGISTER ENTRY     = Location.stopLocationUpdatesAsync
                            = ATTRIBUTED ✅
JS STOP CALLER FAMILY       = NARROWED ✅
BUNDLE                      = EMBEDDED 77787a30 (PROD126) ✅

EXACT ROOT TRIGGER @ T_FAIL = OPEN ★
  Question : pourquoi le JS a jugé légitime d’arrêter
  une task Location encore saine 5,484 s plus tôt ?

───────────────────────────────────────
DISCRIMINANT MISSION 21:18:47→52 (artefacts)
───────────────────────────────────────

booking 38224
= IN_PROGRESS + driver_id=20135 avant ET après
= updated_at figé 10:27:28Z (≪ T_FAIL)
= booking_change_events @ fenêtre = 0
= change_events all-time = 0

HTTP (backend access log)
= 19:18:35 GET /bookings → 200 size=4908
= 19:18:45 GET /bookings/since → 200 size=3 (delta vide)
= 19:18:47 PUT /location → 202
= 19:18:48 GET watermark → 200
= 21:18:49.975 Unregister ★
= 19:18:50 GET /bookings → 200 size=4908 (identique)
= 0× force_tracking_restart dans access log
= burst FCM/token/health POST @ 19:18:50 = DOWNSTREAM

LOC PG
= 19:18:44.484 seq29 mission_id=38224 mission_live
= 19:18:50.515 seq30 mission_id=38224 mission_live
  (même session trk_sess_…003q07ju)

WS
= seuls driver.location.processed @ 45s / 51s
= aucun event mission/socket métier

VERDICT STRUCTUREL
= perte métier réelle de mission @ T_FAIL
  → FORTEMENT AFFAIBLIE / STRONGLY WEAKENED ✅
= pas de MISSION HOLE API/DB observable

AUDIT TIP 286737a2 (read-only)
= d5_session_normal/D5_DUAL_AUTHORITY_AUDIT.md

B2 manager STOP sans lifecycleGeneration = CONFIRMED ✅
  ensureManagerState → stopBackgroundLocationTask("ineligible_…")
  DIRECT, hors stopDriverTrackingBridge, sans gen guard

B1 trackingEligible=false hors mission bridge
  = IMPOSSIBLE tant que state.missionId actif
  = exige trou mission *bridge* (pas API)

DUAL START/STOP AUTHORITY
= CONFIRMED STRUCTURALLY ✅
B2 UNGUARDED MANAGER STOP   = CONFIRMED STRUCTURALLY ✅
B2 AS STORM AMPLIFIER       = LEADING ★★
B2 AS FIRST TRIGGER         = NO / insufficient alone ✅

WRITES state.missionId→null = EXACTEMENT 2 (STOP family only)
  → D5_MISSIONID_WRITES_AUDIT.md
hardStop @ T_FAIL           = EXCLUDED ✅
  (seul caller prod = leave driver context)
IGNITION PATH               = NORMAL stopDriverTrackingBridge ✅
IGNITION FAMILY             = STOP BRIDGE ✅
  clear AVANT native STOP ; B2 peut Unregister en 1er ★

PRE-CLEAR : aucun START effectif (gen bump)
  = CONFIRMED STRUCTURALLY ✅
A1e (start sans bump)       = EXCLUDED ✅ (bump = 1ʳᵉ ligne)

CLASSEMENT
B2 AS STORM AMPLIFIER       = LEADING ★★
B2 AS ROOT EVENT            = NO ✅
A1a pick/cache hole local   = LEADING FAMILY ★
FULL-POLL HOLE pré-T_FAIL   = STRONGLY WEAKENED ✅
  (0× GET /bookings entre 19:18:35 et T_FAIL ;
   seul /since size=3 ; 19:18:50 = post-fail)
EXTERNAL STOP + deps stables
  (pas de START auto)       = DOWNRANKED ✅
  (cascade prod=0 ; auth/terminal/remote/hardStop exclus)
HOOK cleanup / !missionId   = RE-LEADING ★★
  transition requise = cleanup=YES ∧ START=NO
  → T1 missionId→null LEADING
  → T2 status non-actif EXCLUDED
  → T3 unmount WEAKENED
  → T6 scheduling churn EXCLUDED as clear
  → D5_HOOK_TRANSITION_AND_LASTSENT_AUDIT.md

SELF-HEAL as clear missionId = EXCLUDED ✅
SELF-HEAL as FIRST Unregister = LEADING CONDITIONAL ★★
  (PUT BG ≠ lastSentAt ; last_fix_age NULL @ 19:18:35
   ⇒ lastFix null ; startedAge>60s si lastSentAt null)

WHY missionId→null local sans GET /bookings = BLOQUÉ ★
  (aucun writer local viable restant —
   D5_LOCAL_DATA_HOLE_AND_LASTSENT_FINAL.md)

remote kick                 = NOT OBSERVED ✅
FCM / health                = DOWNSTREAM
HOST REMOUNT                = NOT DEMONSTRATED
BRIDGE missionId HOLE       = REQUIRED for B1 ✅
EXACT T1 SOURCE             = UNATTRIBUTED / ARTEFACT-LIMITED ★
SELF-HEAL FIRST STOP        = LEADING CONDITIONAL ★
B2 BYPASS / DUAL AUTHORITY  = CONFIRMED DEFECT ✅
SELF-HEAL FALLBACK startedAge = UNSAFE DESIGN ✅

D5 RCA STRUCTURAL           = SUFFICIENT FOR PATCH DESIGN ✅
PATCH DESIGN                = GO ✅
  → d5_session_normal/D5_PATCH_DESIGN.md
IMPLEMENTATION              = DONE ✅
  (owner STOP + B2 + self-heal L1 + transient React)
CANARY INTERNE              = IN PROGRESS (C4 NEXT)
  → d5_session_normal/D5_CANARY_PROTOCOL.md
C1 PREFLIGHT                = PASS ✅ (versionCode 127 installé)
C1 RUN                      = PASS ✅
C2 RUN                      = PASS ✅
C3 RUN                      = PASS ✅
  binary 128 + inject hole 1s / pending sans confirmed / Unregister=0
  → d5_canary/C3_summary.txt
C4 RUN                      = FAIL ❌
  binary 129 + L1 destructive EXCLUDED (stop_requested=0)
  Unregister hors owner ★ CALLER OPEN
  → d5_canary/C4_summary.txt + C4_UNREG_ATTRIBUTION.md
INSTRUMENTATION             = DONE ✅ (130)
C4_130 ATTRIBUTION          = DONE ✅
  W1 = ensure_manager_state:owner_version_mismatch ★ CLOSED
  → d5_canary/C4_130_ATTRIBUTION.txt
FIX W1                      = DONE ✅
BUILD 131                   = INSTALLED ✅
C4 RE-RUN (131)             = PASS ✅
  Unregister=0 / NATIVE_STOP_ENTRY=0
  → d5_canary/C4_summary_131.txt
CANARY VALIDATED (full)     = YES ✅ (interne C1–C4)
RELEASE REVIEW 131          = FAIL ❌ (historique) → B1/B2/B3 traités sur 132
  B1 SHA FINAL S            = FIXED ✅ a851cf15… + tag d5-rc-final (pushed)
  B2 AAB production QA OFF  = FIXED ✅ ab91958e… versionCode 132 (SHA=S)
  B3 tsc télémétrie D5      = FIXED ✅
  → d5_canary/B2_PRODUCTION_AAB_132.txt
SMOKE RC132                 = PASS ✅ (FG 90s + HOME 180s ; Unreg=0 ; PUT OK ; FGS alive)
  → d5_canary/RC132_SMOKE_summary.txt
RELEASE REVIEW 132          = PASS ✅
RC132 VALIDATED             = YES ✅
RC132                       = FROZEN ✅ (ne plus modifier D5 / canary / SHA S)
  → d5_canary/RELEASE_REVIEW_132.md
  PLAY SUBMISSION / ROLLOUT = HOLD ⛔
DISTRIBUTION                = NO-GO ⛔

═══════════════════════════════════════
P0-E — GPS CONFIRMATION / MAP FRESHNESS
═══════════════════════════════════════

P0-E Q2                     = RCA CLOSED ✅
  PHASE 1 A–E               = PASS ✅ (sha-d5694d8e7cec, PG_FIRST=false)
  NOTE                      = OUTBOX=true restauré mid-lot (défaut compose false)
  PG LOC + capture_id       = OK post-restore
  canonical via P5-B        = OFF (LocationService sync peut encore écrire Redis)
  PHASE 2 flag ON canary    = HOLD ⛔
  → gps-p0e-phase1-ae-result-2026-08-17.md
  INTERDIT                  = RC132 / frontend / flag ON sans GO

BACKEND PROD                = READ-ONLY (obs + MDS)
GPS UI freshness            = P0-E (ouvert)

FERMÉ / NE PLUS ROUVRIR SANS PREUVE
= OTA / AppConfigurationError / not-defined /
  startLocationUpdatesAsync Unregister / unregister* app /
  health monitor caller / perte métier mission API-DB /
  chasse aveugle root T1 (artefacts saturés) /
  nouveaux discriminants D5 (dossier design-complete)

→ d5_session_normal/D5_RCA_SORTIE_OBLIGATOIRE.md
→ d5_session_normal/D5_PATCH_DESIGN.md
→ d5_session_normal/D5_CANARY_PROTOCOL.md
→ d5_session_normal/D5_LOCAL_DATA_HOLE_AND_LASTSENT_FINAL.md
→ d5_session_normal/access_bookings_191820_191910.txt
```
