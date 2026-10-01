# Temps de travail chauffeur

## Règle

✅ **Implémenté** : chaque course `COMPLETED` ou `RETURN_COMPLETED` est un transport. Le forfait lit uniquement `transport_flat_minutes` (recopie de `one_way_minutes` dans la migration `001d2f093160`, sans repli à la lecture). Le sélecteur Temps réel | Forfait entreprise est un mode d’affichage. La clôture fige la politique du jour de chaque ligne (`line_key` `bk:<id>` ou `manual:<id>`, `legacy:<ledger_id>` pour l’existant).

- `A → B` puis `B → A` = 2 transports = 2 × `transport_flat_minutes`. `A → B → C → A` = 3 transports. `journey_key` n’explique que l’appartenance à une mission.
- `PENDING`, `ACCEPTED`, `ASSIGNED`, `EN_ROUTE`, `IN_PROGRESS` = 0 transport, et elles vont dans À vérifier.
- `CANCELED` sans arrivée du chauffeur = 0 transport, absente de l’historique.
- `CANCELED` avec chauffeur arrivé sur place (`arrived_at`, ou assignation `ARRIVED_PICKUP` ou un état ultérieur fiable) = 1 transport, forfait appliqué, visible dans l’historique. Le temps réel n’invente pas de durée s’il n’y a pas d’intervalle fiable. La décision est `counts_as_transport`.
- Date de politique, jour civil Europe/Zurich : fin effective, sinon fin rectifiée, sinon `scheduled_time` seulement s’il n’y a aucune fin. Une course prévue le 30.09 et terminée le 01.10 prend la politique du 1er octobre.
- Temps réel : durée retenue. Les minutes seulement proposées restent hors du total jusqu’à validation.
- Temps ajouté : en temps réel, la durée saisie. En forfait, la règle du type ; si elle est absente, anomalie `manual_compensation_rule_missing`, hors du total forfaitaire.
- Plusieurs versions peuvent coexister sur la période. Près de Clôturer : une phrase unique, ou « plusieurs versions » avec le détail. Chaque ligne de snapshot garde son `policy_id`.
- Écran : liste des chauffeurs, puis résumé pleine page (phrase, tableau des journées, détail dépliable). Le détail ne s’affiche pas sous le tableau.
- Un trajet réparti entre plusieurs chauffeurs reste `requires_review` et 0 minute. Une règle de transport absente aussi.

`one_way_minutes`, `round_trip_minutes` et `intermediate_stop_minutes` restent en base pour l’historique. Ils ne pilotent plus cet écran ni l’onglet de réglage.

## Qualité de la durée

| Situation | Qualité | Minutes |
| --- | --- | --- |
| `arrived_at` et `completed_at` | `verified` | écart réel |
| Avant la bascule, sans `arrived_at`, avec prise en charge et fin | `estimated_historical` | `boarded_at → completed_at`, affiché « ≈ Estimé » |
| Après la bascule, sans `arrived_at`, itinéraire voiture | `pending_validation` | Le temps de trajet est la durée estimée de la réservation : historique des courses semblables, sinon durée OSRM voiture × 1,55 (même facteur que `GET /osrm/route`). Si la course n’a pas de GPS, les adresses sont géocodées puis OSRM est appelé. Les minutes offertes au chauffeur (5 par défaut, réglables) s’ajoutent ensuite, une fois, et ne corrigent pas le trajet. Pas de vol d’oiseau. Le temps OSRM à vide n’est pas affiché. Si le géocodage ou le routeur échoue, `routing_status = unavailable` et aucune minute n’est proposée (`Temps à déterminer`). Le temps proposé n’entre dans le temps travaillé validé qu’après validation ou rectification. |
| Statut terminé sans `completed_at` | `incomplete` | temps réel à valider, hors du total réel ; le forfait s’applique si `transport_flat_minutes` de la date métier est univoque. Anomalie `completion_time_missing`. |
| Correction administrative | `adjusted` | snapshot complet arrivée + fin |

La bascule `WORK_TIME_ARRIVED_AT_CUTOVER_AT` est un instant UTC absolu (un horodatage naïf est lu comme UTC). Le défaut est le **01.10.2026 00:00 Europe/Zurich**, soit `2026-09-30T22:00:00Z` : le 1er octobre 2026, Genève est encore en UTC+2. `2026-10-01T00:00:00Z` correspondrait à 02:00 à Genève, et classerait du mauvais côté les courses entre 00:00 et 01:59 heure suisse. La comparaison est `completed_at < cutover` : minuit pile est déjà du nouveau côté. `scheduled_time` range une course sans heure de fin et choisit alors sa politique. Il ne fabrique ni une fin ni une durée.

Une course `23:50 → 00:20` compte 10 min le premier jour civil Europe/Zurich et 20 min le suivant. Le forfait, lui, est rattaché au jour de fin. Les changements d'heure (29.03 et 25.10) utilisent la durée absolue.

## Écriture de l'arrivée

`record_booking_arrival` est le seul point qui pose `booking.arrived_at`. Il ne l'écrase jamais. Il est appelé par le jalon chauffeur et par le passage d'assignation à `ARRIVED_PICKUP` (PATCH dispatcher).

## Rémunération historisée

Les politiques `driver_compensation_policy` ont une fenêtre `[effective_from, effective_until)`. La règle utilisée est celle en vigueur à la date de fin du tronçon porteur, pas la configuration du jour.

Tant que le mois n'est pas clôturé, le calcul est dynamique. Seul un mois civil complet déjà terminé en Europe/Zurich peut être figé : le dernier jour compte encore, même un samedi, un dimanche ou un jour férié. Le bouton affiché est « Clôturer le mois », puis « Réouvrir » suivi du nom du mois. Jour, semaine et période personnalisée ne se clôturent pas (`409`, `PERIOD_NOT_CLOSABLE`). Octobre 2026 n'est pas clôturable le 30 ni le 31 octobre ; il le devient le 1er novembre 2026 à 00:00, heure de Zurich. `POST /companies/me/work-time/periods/finalize` copie une ligne par `line_key` dans `driver_compensation_ledger` (`bk:<booking_id>`, `manual:<entry_id>`, ou `legacy:<ledger_id>` pour une clôture déjà figée). Un trajet partagé entre plusieurs chauffeurs ne retire pas le forfait et ne crée pas d'anomalie : chaque course comptée garde ses `transport_flat_minutes`. Avant l'écriture, les totaux du rapport ouvert (transports, minutes réelles, forfait, temps ajouté, éléments à vérifier) sont relus sur le snapshot. Un écart répond `409` et aucune clôture n'est enregistrée. L’unicité est `(closure_id, line_key)`. Une seconde finalisation du même intervalle répond `409`. Un intervalle qui chevauche une clôture active répond `409`. Deux finalisations simultanées de la même entreprise sont sérialisées par `pg_advisory_xact_lock`. `POST .../periods/reopen` exige un motif et ne s'applique qu'à une clôture encore active (`404` sinon). Le ledger précédent n'est pas effacé. Une période close n’est ni éclatée ni fusionnée.

Après clôture, une correction, une saisie manuelle, son annulation, ou une politique dont la fenêtre touche la période répondent `409`. Une course modifiée ensuite ne change pas les minutes rémunérées : elles restent celles du snapshot, sur la date comptable figée.

## API

Préfixe `/companies/me/work-time` (rôle entreprise, isolation `company_id`).

- `GET /summary?from&to` — transports, temps réel, forfait, éléments à vérifier, et `contractual_rules`.
- `GET /drivers/<id>?from&to&filter` — détail par jour. Filtres : `all`, `transports`, `manual`, `adjusted`, `anomalies`.
- `GET /bookings/<id>/explain` — jalons, règle, corrections.
- `POST /adjustments` — snapshot complet `corrected_arrived_at` et `corrected_completed_at`.
- `POST /manual-entries` — durée calculée `ended_at − started_at`. `POST /manual-entries/<id>/cancel` n'efface pas la ligne.
- `GET|POST /compensation-policies` — `mode` et `transport_flat_minutes`. Les colonnes aller / aller-retour / étape ne sont plus lues par ce module.
- `POST /periods/finalize` et `POST /periods/reopen`

Les droits `company.work_time.view|manage|configure` sont préparés. Tant que le control plane est en mode shadow, le rôle `company` suffit.

## Export

Pas d'export fichier dans cette version. Le détail par jour est déjà une liste de lignes plates (segment, jour, minutes, forfait, anomalies) réutilisable plus tard en CSV.

## Données historiques

La plupart des courses terminées n'ont ni `arrived_at` ni souvent `completed_at`. Elles restent visibles comme incomplètes ou, avant la bascule, comme estimées si la prise en charge et la fin existent. Aucune heure passée n'est réécrite.

## Rapport de phase 1

Migration générée puis appliquée : `backend/migrations/versions/886c4db45b84_temps_de_travail_chauffeur.py` (`b2ca98b6cb2d` → `886c4db45b84`), puis `9ba102797f17` et `001d2f093160`. Cette branche et la tête facturation `1aa803e70656` sont réunies par `aaaf36d71923` (fusion sans changement de schéma), de sorte qu'il n'y a plus qu'une tête Alembic. Elle ajoute `booking.arrived_at`, les index `(driver_id, completed_at)` et l’index partiel `(driver_id, scheduled_time) WHERE completed_at IS NULL`, ainsi que les tables de politique, correction, saisie, clôture et ledger. L’autogénération proposait aussi des écarts de schéma sans lien avec ce module ; ils n’ont pas été appliqués.

Fichiers de la phase 1 : `backend/application/bookings/record_booking_arrival.py`, `backend/domain/work_time/`, `backend/services/work_time/report_service.py`, `backend/services/work_time/permissions.py`, `backend/routes/company_work_time.py`, `frontend/src/pages/company/Driver/workTime/`.

Endpoints : `GET /companies/me/work-time/summary`, `GET /companies/me/work-time/drivers/<id>`, `GET /companies/me/work-time/bookings/<id>/explain`.

Exemples (forfait 30 min par transport) :

```text
A→B terminé
  line_key=bk:10 rule_type=transport_flat compensated_minutes=30

A→B terminé, B→A encore assigné
  line_key=bk:1 compensated_minutes=30
  B→A : kind=open, anomalie transport_not_completed, 0 transport

A→B→C→A complet
  line_key=bk:1, bk:2, bk:3, même journey_key, 3 × 30 = 90

Clôture antérieure
  line_key=legacy:<ledger_id>, snapshot inchangé
```

Durées et politique :

```text
COMPLETED ou RETURN_COMPLETED sans completed_at
  quality=incomplete worked_minutes=null anomalie=completion_time_missing

Avant la bascule, sans arrived_at, avec prise en charge et fin
  quality=estimated_historical source=historical_pickup_to_completed

Après la bascule, sans arrived_at
  quality=incomplete worked_minutes=null anomalie=arrival_not_recorded

30.09 23:50 → 01.10 00:20 (Europe/Zurich)
  10 min le 30.09, 20 min le 01.10, total 30
  forfait rattaché au 01.10
```

Une durée inconnue reste `null` sur la ligne et dans les totaux de journée (`worked_minutes`, `real_minutes`, `flat_minutes`). Elle n'est pas sérialisée en 0. `kpis.total_worked_minutes` n'additionne que les durées entières déjà connues ; le reste est porté par `review_count_real`. La somme des `compensated_minutes` du détail égale `kpis.compensated_minutes`.

Tests de cette phase : `backend/tests/domain/work_time/test_work_time.py`, `backend/tests/routes/test_company_work_time_access.py`, `backend/tests/routes/test_company_work_time_hardening.py`. Jest : `frontend/src/pages/company/Driver/workTime/`.

## Livraison

Migration `886c4db45b84_temps_de_travail_chauffeur` (`booking.arrived_at`, politiques, corrections, saisies, clôture, ledger).

Fichiers principaux : `backend/application/bookings/record_booking_arrival.py`, `backend/domain/work_time/`, `backend/services/work_time/`, `backend/routes/company_work_time.py`, `frontend/src/pages/company/Driver/workTime/`, `frontend/src/pages/company/Settings/tabs/CompensationTab.jsx`.

Tests exécutés : `tests/domain/work_time/test_work_time.py` (domaine, dont A→B, A/R incomplet, A→B→C→A, fin manquante, estimé, bascule, minuit, DST, multi-chauffeurs, règle historisée, somme détail = résumé) et Jest `workTimeFormat` / `periodRange`.

## Risques résiduels

- Les courses historiques sans `arrived_at` ni `completed_at` n'ont pas de durée tant qu'elles ne sont pas rectifiées.
- Un trajet partagé entre plusieurs chauffeurs reste payé course par course : deux bookings du même `journey_key` font deux `line_key` et deux forfaits, sans anomalie automatique.
- Le retour d'un parcours institutionnel sans drapeau `is_return` est déduit par égalité d'adresse pour le contexte de mission. Cette déduction ne fusionne pas les transports et ne retire pas le forfait.
- Les droits `company.work_time.*` ne sont pas encore appliqués : le control plane entreprise est en mode shadow, seul le rôle entreprise et l'isolation `company_id` comptent.
- L'export CSV, Excel ou PDF n'est pas branché. Le détail par jour est déjà une liste de lignes plates.
