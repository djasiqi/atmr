# Plan d’implémentation – Application Mobile Enterprise Dispatch

## Backlog détaillé (S1 → S11+)

| ID    | Semaine | User Story (INVEST)                                                                                    | Description & critères Gherkin                                                                                                                                                           | Livrables                                         |
| ----- | ------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| US-01 | 1       | En tant que dispatcher, je veux partager mes usages actuels pour que l’app mobile reflète mes besoins. | Given un atelier “dispatch terrain” <br>When je décris mes actions <br>Then les parcours clés (Accueil, Liste, Fiche, Actions rapides, Mode) sont documentés.                            | Compte-rendu ateliers, personae, parcours annotés |
| US-02 | 1       | En tant que superviseur, je veux clarifier mes attentes de monitoring.                                 | Gherkin similaire US-01 mais focus metrics, alertes, pilotage.                                                                                                                           | Tableau besoins superviseur                       |
| US-03 | 1       | En tant qu’Admin Sécurité, je veux lister les exigences SSO/MFA/MDM.                                   | Gherkin “Given workshop sécu, When collecte exigences, Then exigences listées (SSO, MFA, sessions, MDM, RGPD).”                                                                          | Doc exigences sécu                                |
| US-04 | 2       | En tant que PO, je veux inventorier les briques réutilisables de la Driver App.                        | Gherkin “Given audit repo mobile/driver-app, When j’identifie modules (auth, nav, i18n), Then je rédige une matrice de réutilisation.”                                                   | Mapping composants / debt                         |
| US-05 | 2       | En tant que designer, je veux produire des maquettes haute fidélité pour tous les écrans.              | Gherkin “Given Figma project, When je livre Accueil, Liste, Fiche, Assignation, Réassignation, Annulation, Modes, Incidents, Monitoring, Then les flows sont validés par 2 dispatchers.” | Maquettes + prototype                             |
| US-06 | 2       | En tant que PO, je veux des user stories détaillées avec critères Gherkin.                             | Gherkin “Given backlog initial, When j’écris stories, Then chacune a critères mesurables.”                                                                                               | Backlog complet (ce tableau)                      |
| US-07 | 3       | En tant que dev backend, je veux définir les endpoints /company_mobile/dispatch.                       | Scenario: documentation OpenAPI versionnée, statuts, paramètres, erreurs.                                                                                                                | OpenAPI brouillon (v1)                            |
| US-08 | 3       | En tant qu’architecte sécu, je veux spécifier SSO/MFA (OIDC/SAML, TOTP/Push).                          | Gherkin “Given systèmes SSO existants, When je spécifie OIDC/SAML, Then les flows tokens + scopes sont détaillés.”                                                                       | Doc flux auth                                     |
| US-09 | 3       | En tant que dev observabilité, je veux définir les événements audit/OTel.                              | Gherkin “Given besoin traçabilité, When je modélise table audit + spans, Then je documente schéma JSON + attributs.”                                                                     | Schéma audit + plan instrumentation               |
| US-10 | 4       | En tant que dev backend, je veux développer squelette Flask des endpoints (lecture seule).             | Gherkin “Given contract OpenAPI, When j’implémente /status, /rides, Then tests de contrat green.”                                                                                        | MR backend lecture seule                          |
| US-11 | 4       | En tant que responsable sécu, je veux intégrer SSO/MFA dans l’API.                                     | Gherkin “Given OIDC provider, When user se connecte, Then tokens avec scopes mobile et MFA enforced.”                                                                                    | MR auth + tests                                   |
| US-12 | 5       | En tant que dev mobile, je veux bootstraper l’app (workspace RN/Expo).                                 | Gherkin “Given repo mono, When je crée package mobile-enterprise-dispatch, Then navigation, theming, i18n configurés.”                                                                   | Repo initial + CI                                 |
| US-13 | 5       | En tant que dev mobile, je veux afficher dashboard read-only.                                          | Gherkin “Given API /status, When user ouvre app, Then KPIs et état OSRM/Agent s’affichent même offline (cache).”                                                                         | Écran accueil                                     |
| US-14 | 5       | En tant que dev mobile, je veux afficher liste + fiches courses.                                       | Gherkin “Given API /rides, When user consulte, Then liste triée avec filtres, états loading/offline.”                                                                                    | Écrans liste/fiches                               |
| US-15 | 6       | En tant que dev mobile, je veux persister un cache minimal (MMKV/SQLite).                              | Gherkin “Given absence réseau, When j’ouvre app, Then dernières données sont visibles + tag offline.”                                                                                    | Module storage                                    |
| US-16 | 6       | En tant que QA, je veux tests unitaires sur services mobile.                                           | Gherkin “Given services API, When je lance tests, Then couverture ≥80%.”                                                                                                                 | Tests Jest                                        |
| US-17 | 7       | En tant que dispatcher, je veux assigner/réassigner/annuler depuis mobile.                             | Gherkin multi-scenario (succès, échec fairness, 409).                                                                                                                                    | UI actions + appels API                           |
| US-18 | 7       | En tant que dev backend, je veux gérer validations fairness/préférence côté API mobile.                | Gherkin “Given requête assign, When fairness violée, Then 422 avec détails logs.”                                                                                                        | MR backend validations                            |
| US-19 | 8       | En tant que dispatcher, je veux basculer de mode Manuel/Semi/Full.                                     | Gherkin “Given current mode, When je demande FULLY, Then confirmation multi-étapes + audit.”                                                                                             | UI modes + endpoint                               |
| US-20 | 8       | En tant que superviseur, je veux déclarer incident + escalade chauffeur urgent.                        | Gherkin “Given ride, When incident signalé, Then log audit + règles chauffeur urgent respectées.”                                                                                        | Formulaire incidents                              |
| US-21 | 9       | En tant que dispatcher, je veux recevoir notifications push.                                           | Gherkin “Given ride imminent non assigné, When seuil atteint, Then notif push + deep-link assignation.”                                                                                  | Push (Expo) + backend                             |
| US-22 | 9       | En tant que dev mobile, je veux actions rapides via deep-links sécurisés.                              | Gherkin “Given notif assign, When user clique, Then app ouvre fiche avec contexte.”                                                                                                      | Linking + guard                                   |
| US-23 | 10      | En tant que superviseur, je veux un monitoring Fully-Auto temps réel.                                  | Gherkin “Given agent tick, When WS event, Then écran monitoring se met à jour (fallback polling).”                                                                                       | Vue monitoring + WS                               |
| US-24 | 10      | En tant que dev backend, je veux exposer feed WebSocket/polling pour monitoring.                       | Gherkin “Given agent log, When push event, Then message JSON standard.”                                                                                                                  | Endpoint WS/poll                                  |
| US-25 | 11      | En tant que chef de projet, je veux offline minimal avec reprise d’actions.                            | Gherkin “Given perte réseau pendant assign, When réseau revient, Then action rejouée transactionnellement.”                                                                              | File actions offline                              |
| US-26 | 11      | En tant qu’admin IT, je veux builds signés prêts MDM/Stores.                                           | Gherkin “Given build Expo, When je génère IPA/APK, Then profils MDM appliqués + doc déploiement.”                                                                                        | Builds & doc MDM                                  |
| US-27 | 11      | En tant que DPO, je veux checklist RGPD complète.                                                      | Gherkin “Given audit data, When je vérifie traitement, Then checklist validée (consent, retention).”                                                                                     | Checklist RGPD                                    |
| US-28 | 11      | En tant que QA lead, je veux plan de tests complet (unit/inté/e2e).                                    | Gherkin “Given suites tests, When j’exécute CI, Then pipelines passent et rapports disponibles.”                                                                                         | Plan tests + scripts CI                           |
| US-29 | 11+     | En tant que PO, je veux un plan pilote client.                                                         | Gherkin “Given features prêtes, When pilote lancé, Then métriques success (CSAT, time-to-assign) suivies.”                                                                               | Plan pilote + template feedback                   |
| US-30 | 11+     | En tant que support, je veux runbook incidents mobile.                                                 | Gherkin “Given incident app, When je consulte runbook, Then procédure rollback / escalade.”                                                                                              | Runbook                                           |
| US-31 | 11+     | En tant que data analyst, je veux dashboard métriques.                                                 | Gherkin “Given OTel metrics, When j’ouvre dashboard, Then visualisations (taux assignation, retards évités).”                                                                            | Dashboard (Grafana)                               |

## OpenAPI initial (extrait structurant)

```yaml
openapi: 3.1.0
info:
  title: ATMR Enterprise Dispatch Mobile API
  version: 1.0.0
servers:
  - url: https://api.atmr.local/company_mobile/dispatch/v1
    description: Environnement interne
security:
  - bearerAuth: []
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  headers:
    X-Request-ID:
      schema: { type: string }
    X-Company-ID:
      schema: { type: string }
    X-Session-ID:
      schema: { type: string }
    X-Device-ID:
      schema: { type: string }
  parameters:
    DateParam:
      in: query
      name: date
      schema: { type: string, format: date }
    StatusParam:
      in: query
      name: status
      schema:
        type: string
        enum: [assigned, unassigned, urgent, cancelled]
    SearchParam:
      in: query
      name: q
      schema: { type: string, maxLength: 80 }
    PageParam:
      in: query
      name: page
      schema: { type: integer, minimum: 1 }
    PageSizeParam:
      in: query
      name: page_size
      schema: { type: integer, minimum: 1, maximum: 100 }
  schemas:
    StatusSummary:
      type: object
      properties:
        osrm:
          type: object
          properties:
            status: { type: string, enum: [OK, WARNING, DOWN] }
            latency_ms: { type: integer }
            last_check: { type: string, format: date-time }
        agent:
          type: object
          properties:
            mode: { type: string, enum: [MANUAL, SEMI_AUTO, FULLY_AUTO] }
            active: { type: boolean }
            last_tick: { type: string, format: date-time, nullable: true }
        optimizer:
          type: object
          properties:
            active: { type: boolean }
            next_window_start:
              { type: string, format: date-time, nullable: true }
        kpis:
          type: object
          properties:
            date: { type: string, format: date }
            total_bookings: { type: integer }
            assigned_bookings: { type: integer }
            assignment_rate: { type: number, format: float }
            at_risk: { type: integer }
    RideSummary:
      type: object
      properties:
        id: { type: string }
        time:
          type: object
          properties:
            pickup_at: { type: string, format: date-time, nullable: true }
            drop_eta: { type: string, format: date-time, nullable: true }
            window_start: { type: string, format: date-time, nullable: true }
            window_end: { type: string, format: date-time, nullable: true }
        client:
          type: object
          properties:
            id: { type: string }
            name: { type: string }
            priority: { type: string, enum: [LOW, NORMAL, HIGH] }
        route:
          type: object
          properties:
            pickup_address: { type: string }
            dropoff_address: { type: string }
            distance_km: { type: number, format: float, nullable: true }
        status:
          { type: string, enum: [assigned, unassigned, completed, cancelled] }
        driver:
          type: object
          properties:
            id: { type: string, nullable: true }
            name: { type: string, nullable: true }
            is_emergency: { type: boolean }
        flags:
          type: object
          properties:
            risk_delay: { type: boolean }
            prefs_respected: { type: boolean }
            fairness_score: { type: number, format: float, nullable: true }
            override_pending: { type: boolean }
    AssignRequest:
      type: object
      required: [driver_id]
      properties:
        driver_id: { type: string }
        reason: { type: string, nullable: true, maxLength: 280 }
        respect_preferences: { type: boolean, default: true }
        allow_emergency: { type: boolean, default: false }
        idempotency_key: { type: string, format: uuid }
    AssignResponse:
      type: object
      properties:
        ride_id: { type: string }
        driver_id: { type: string }
        scheduled_time: { type: string, format: date-time }
        fairness_delta: { type: number, format: float }
        audit_event_id: { type: string }
        message: { type: string }
```

### Exemples de payloads

1. **GET `/status`**

   ```json
   {
     "osrm": {
       "status": "OK",
       "latency_ms": 42,
       "last_check": "2025-11-07T05:10:00+01:00"
     },
     "agent": {
       "mode": "FULLY_AUTO",
       "active": true,
       "last_tick": "2025-11-07T05:09:30+01:00"
     },
     "optimizer": {
       "active": true,
       "next_window_start": "2025-11-07T06:00:00+01:00"
     },
     "kpis": {
       "date": "2025-11-07",
       "total_bookings": 24,
       "assigned_bookings": 20,
       "assignment_rate": 0.83,
       "at_risk": 2
     }
   }
   ```

2. **GET `/rides?date=2025-11-07&status=unassigned`**

   ```json
   {
     "page": 1,
     "page_size": 50,
     "total": 4,
     "items": [
       {
         "id": "RID-2025-11-07-0019",
         "time": {
           "pickup_at": null,
           "window_start": "2025-11-07T15:30:00+01:00",
           "window_end": "2025-11-07T16:00:00+01:00"
         },
         "client": {
           "id": "C-051",
           "name": "Charlotte Walter",
           "priority": "NORMAL"
         },
         "route": {
           "pickup_address": "Chem. Thury 7B, 1206 Genève",
           "dropoff_address": "Rue de Vermont 6bis, 1202 Genève",
           "distance_km": 8.6
         },
         "status": "unassigned",
         "driver": null,
         "flags": {
           "risk_delay": false,
           "prefs_respected": true,
           "fairness_score": null,
           "override_pending": false
         }
       }
     ]
   }
   ```

3. **GET `/rides/{ride_id}`**

   ```json
   {
     "summary": {
       "id": "RID-2025-11-07-0008",
       "time": {
         "pickup_at": "2025-11-07T13:15:00+01:00",
         "drop_eta": "2025-11-07T13:45:00+01:00"
       },
       "client": {
         "id": "C-032",
         "name": "Akbar Kherad",
         "priority": "NORMAL"
       },
       "route": {
         "pickup_address": "Clinique les Hauts d'Anières",
         "dropoff_address": "Av. de Champel 42",
         "distance_km": 6.2
       },
       "status": "assigned",
       "driver": {
         "id": "DRV-012",
         "name": "Giuseppe Bekasy",
         "is_emergency": false
       },
       "flags": {
         "risk_delay": false,
         "prefs_respected": true,
         "fairness_score": 0.72,
         "override_pending": false
       }
     },
     "history": [
       {
         "ts": "2025-11-07T05:00:00+01:00",
         "event": "CREATED",
         "actor": "system",
         "details": {}
       },
       {
         "ts": "2025-11-07T05:05:12+01:00",
         "event": "ASSIGN",
         "actor": "agent:fully_auto",
         "details": {
           "driver_id": "DRV-012",
           "fairness_delta": -0.2,
           "reason": "Simple assign (pas d’impact).",
           "dispatch_run_id": "RUN-2025-11-07-0004"
         }
       }
     ],
     "conflicts": [],
     "notes": []
   }
   ```

4. **POST `/rides/{ride_id}/assign`**

   ```json
   {
     "driver_id": "DRV-045",
     "reason": "Assignation manuelle suite appel clinique.",
     "respect_preferences": true,
     "allow_emergency": false,
     "idempotency_key": "781e4d7c-381f-4f4d-9dce-8f7d2f3bb111"
   }
   ```

5. **POST `/modes/switch`**

   ```json
   {
     "mode_before": "SEMI_AUTO",
     "mode_after": "FULLY_AUTO",
     "effective_at": "2025-11-07T06:00:00+01:00",
     "requires_approval": false,
     "audit_event_id": "AUD-2025-11-07-00012"
   }
   ```

## Écrans clés

- **Accueil KPI** : header mode actif avec pastille couleur, tuiles KPIs (courses totales, assignées, taux, retards), pastilles OSRM/Agent/Optimiseur, liste alertes récentes, bouton flottant “Lancer dispatch”, bandeau offline.
- **Liste des courses** : onglets (Non assignées, Assignées, Urgentes), barre recherche, cartes détaillées (heure, client, trajet, chauffeur, badges), gestes swipe pour actions express/détails, indicateur offline.
- **Fiche course** : horaires et fenêtres, adresses, section patient (données sensibles masquées), bloc chauffeur (actuel + suggestions triées), historique timeline, actions (Assigner, Réassigner, Annuler, Incident), indicateur fairness.
- **Assignation / Réassignation** : modale suggestions avec scores, champ raison, toggle chauffeur d’urgence, résumé impact fairness & retard.
- **Bascules de mode** : écran slider modes avec descriptions, checklist pré-requis, confirmation multi-étapes, MFA, récap audit.
- **Incidents / Urgent** : formulaire type/sévérité/note/photo, suggestions actions, bouton bascule chauffeur urgence avec justification obligatoire.
- **Monitoring Fully-Auto** : timeline des ticks (actions acceptées/rejetées), graphique fairness, liste alertes, bouton re-run ciblé.

## Stratégie de notifications push

| Cas                    | Priorité | Message                                          | Deep-link               | Conditions                                  | Rate limit         |
| ---------------------- | -------- | ------------------------------------------------ | ----------------------- | ------------------------------------------- | ------------------ |
| Non assigné à T-30 min | Haute    | « ⚠️ Course #123 dans 30 min sans chauffeur »    | `app://rides/RID-...`   | Modes MANUEL/SEMI ou FULL si agent en pause | 1 notif/ride/heure |
| Annulation patient     | Haute    | « ❌ Patient X a annulé la course de 14h00 »     | `app://rides/RID-...`   | Toujours                                    | 1                  |
| Retard probable        | Moyenne  | « ⏳ Risque retard pour Chauffeur Y (+12 min) »  | `app://drivers/DRV-...` | Delta ETA > seuil                           | 1/2h/chauffeur     |
| Échec OSRM             | Haute    | « 🚨 OSRM indisponible, vérifiez routes »        | `app://status`          | OSRM DOWN > 2 min                           | 1/événement        |
| Changement mode        | Moyenne  | « 🔄 Mode dispatch passé en FULLY_AUTO (par Z) » | `app://modes`           | Toujours                                    | 3/jour             |
| Incident signalé       | Haute    | « 🚑 Incident : panne déclarée par chauffeur Y » | `app://incidents`       | Toujours                                    | 1/incident         |

Notifications chiffrées, payload minimal, actions rapides (assigner, rappeler).

## Stratégie temps réel

- **WebSocket** `wss://api…/company_mobile/dispatch/v1/ws`
  - Auth JWT + en-têtes `X-Company-ID`, `X-Session-ID`.
  - Messages typés (`tick`, `ride_updated`, `mode_changed`, `alert`) avec structure JSON documentée.
  - Heartbeat 30 s, reconnexion exponentielle (1/5/10 s).
- **Fallback polling**
  - `GET /status/stream?since=<timestamp>` avec ETag/Last-Modified.
  - Backoff exponentiel 15→60 s.

## Checklist RGPD & MDM

**RGPD**

- [ ] Registre traitement « Dispatch mobile » créé.
- [ ] Base légale définie (contrat/consentement).
- [ ] Minimisation des données (patient partiellement affiché).
- [ ] Droits utilisateurs (politique, contact DPO).
- [ ] Stockage local chiffré (SecureStore/MMKV).
- [ ] Rétention cache ≤ 30 jours.
- [ ] Journalisation pseudonymisée.
- [ ] Procédure d’effacement disponible.

**MDM**

- [ ] Builds signés (Apple Enterprise / Android Private).
- [ ] PIN/biométrie obligatoires.
- [ ] Blocage copier/coller sensible.
- [ ] Blocage capture écran (si supporté).
- [ ] Wipe à distance configuré.
- [ ] Cert pinning/proxy.
- [ ] Mise à jour forcée (version minimale).

## Batterie de tests E2E (Detox)

1. **Connexion SSO + MFA** : login SSO → MFA TOTP → arrivée dashboard (assertion token & mode affiché).
2. **Consultation offline** : cache préparé → mode avion → données affichées avec badge offline.
3. **Assignation simple** : liste non assignées → assignation chauffeur recommandé → vérifier fiche & fairness delta.
4. **Réassignation conflit 409** : simuler ETag mismatch → attente erreur → UI propose rafraîchissement/diff.
5. **Incident & urgence** : fiche assignée → incident panne → bascule chauffeur urgence → audit vérifié.
6. **Bascule mode FULL → SEMI** : écran modes → confirmation → audit ID visible.
7. **Notification deep-link** : simuler push “non assigné imminent” → app reprend sur fiche course et action disponible.

Tests exécutés en CI (Expo + Detox) avec backend mocké (wiremock).
