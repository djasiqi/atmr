# P1-C1 — La primitive qui bloque le worker (identifiée, sans déploiement)

**Date :** 2026-08-27 · **Prod :** atmr-backend-1 (gunicorn workers=1 gevent) · **Méthode :** faits de config + échantillonnage live `pg_stat_activity` pendant un switch ADB réel + fenêtres Traefik. **Aucun code ni config modifiés.**

## Verdict C1

```text
BLOQUEUR PRINCIPAL (★)
= orage N+1 des endpoints hub messages
  (GET /messages/<cid>/hub/threads, /hub/unread-count, /conversations/inbox)
  exécuté sur le worker gevent unique avec psycopg2 NON coopératif.

Classification demandée :
A. SQL lent                 = NON  (0 requête > 0,2 s sur tous les échantillons)
B. attente pool DB          = NON  (pool 50+20, requêtes s'écoulent en continu)
C. verrou transactionnel    = NON  (0 wait_event Lock sur tous les échantillons)
D. DNS / HTTP externe       = NON  (aucun appel externe dans services/messaging ;
                                    handlers occupés en micro-SELECT en continu)
E. I/O sync non cooperative = OUI ★ (dialecte postgresql+psycopg2, psycogreen ABSENT,
                                    monkey.patch_all ne patche pas psycopg2 →
                                    chaque micro-query fige la boucle)
F. travail CPU              = OUI   (hydratation ORM de centaines de rows/queries
                                    par appel, jamais de yield entre les queries)
G. greenlet monopolisant    = OUI   (conséquence : chaque handler N+1 monopolise
                                    le worker 4-15 s ; 3-6 handlers concurrents
                                    au switch s'entre-affament)
```

## Le N+1, nommé dans le code

`ConversationService.build_company_inbox` (et variante driver) :

1. Charge **toutes** les conversations de la company (`.all()`).
2. `_load_read_message_ids(user_id)` : charge **tous** les MessageRead de
   l'utilisateur (liste d'IDs potentiellement énorme).
3. **Par conversation** — `_thread_row(conv, ...)` :
   - `Message.query.filter_by(conversation_id).order_by(timestamp desc).first()`
     → 1 SELECT message / conv (le « SELECT message... » répété du sampler) ;
   - `unread_count_for_user(...)` → 1 `SELECT count(*)` / conv avec
     `~Message.id.in_(read_ids)` (liste IN géante ; le « SELECT count(*) FROM
     (SELECT message... » du sampler) ;
   - `Booking.query.get(context_id)` par conversation mission
     (les « SELECT booking... » du sampler).

Pour N conversations : **~2N+M micro-requêtes par appel**. Le mobile appelle
`threads` + `unread-count` + `inbox` en parallèle à l'arrivée company, puis
re-tente après chaque timeout 15 s (499) — le serveur **continue de traiter les
requêtes abandonnées** (Flask ne s'arrête pas sur un client parti).

## Preuves dynamiques (épisode contrôlé 22:37-22:38 UTC, switch ADB réel)

- Sampler `pg_stat_activity` (~330 ms) pendant 100 s : max **0,2 s** par requête,
  1-4 actives, **0 verrou** — pendant que Traefik enregistre des handlers de
  **15 s → 499**. Le mur de 15 s n'est PAS une requête SQL : c'est le cumul
  micro-SQL + CPU ORM du handler, dilaté par la concurrence sur le worker unique.
- Fenêtre Traefik de l'épisode (~2 min, cumuls) :
  - `messages/1/hub/unread-count` : 3 req = **44,9 s** (3×15 s→499)
  - `messages/1/hub/threads` : 2 req = **30 s** (2×15 s→499)
  - `company_mobile/dispatch/v1/rides` : 5 req = 27,8 s (8,6/7,3/4,9/4,3/2,7 s)
  - **`auth/refresh-token` : 71 req** (storm token révoqué — C2), dont un 499 à 15 s
  - **`driver/me/bookings/eta` : 40 req** en contexte company (fuite pollers — C3),
    `driver/me/telemetry/push` : 16, `driver/me/location` : 3
  - `auth/switch-context` : **124 ms** (nominal quand le worker est libre)
- Rappel C0 : les spikes switch-context 6,8/6,5/12 s se produisent quand le POST
  arrive PENDANT cet orage.

## Faits de configuration (image prod sha-5c86097828af)

```text
DATABASE_URL scheme  = postgresql+psycopg2     (driver C)
psycogreen           = ABSENT de l'image
wsgi.py              = gevent.monkey.patch_all() applique (ne couvre pas psycopg2)
gunicorn             = --workers 1 --worker-class gevent (API + Socket.IO)
PgBouncer            = transaction, DEFAULT_POOL_SIZE=200 (pas un facteur)
SQLAlchemy prod      = pool_size 50, max_overflow 20 (pas un facteur)
pg_stat_statements   = non installe (au passage)
```

## Gate P1-C1 (défini — à exécuter après corrections C1/C2/C3)

```text
10 switches alternés driver <-> company (vrai UI ADB)
Pour chaque switch :
  - switch-context HTTP < 1 s (Traefik)
  - aucune requête post-switch > 2 s
  - aucun 499
  - aucun refresh révoqué rejoué
  - aucun poller du contexte précédent (0 requête /driver/me/* en company)
  - pas d'accumulation Socket.IO anormale (handle_connect stables)
Puis seulement : P1-C2 = gate mobile 10/10 UX (seuils habituels).
```

## C1a — Dé-N+1 inbox : IMPLÉMENTÉ EN LOCAL (2026-08-27, non déployé)

✅ **Implémenté** : `_batch_thread_row_data()` dans
`backend/services/messaging/conversation_service.py` — 3 requêtes par lot
(dernier message via ROW_NUMBER, non-lus via anti-jointure MessageRead,
statuts booking en un IN) au lieu de 2N+M requêtes unitaires. `_thread_row`
accepte `preloaded=` (chemin unitaire conservé pour compat) ;
`build_company_inbox` / `build_driver_inbox` batchent ; `_load_read_message_ids`
(liste IN géante) retiré du chemin inbox. Logs `inbox_perf`
(duration_ms/conversations/threads) ajoutés. Aucun changement de contrat.

Mesures locales (Docker dev, dataset seedé 55 conversations / 2750 messages /
840 lectures — `backend/scripts/perf/measure_inbox_c1a.py`) :

| Scope | Requêtes SQL | SQL cumulé | Handler |
|---|---|---|---|
| company_inbox | 242 → **48** (−80 %) | 1358 → 164 ms | 1985 → **585 ms** |
| company_hub_threads | 237 → **48** | 962 → 87 ms | 1449 → **165 ms** |
| driver_inbox | 188 → **54** (−71 %) | 385 → 109 ms | 645 → **276 ms** |

- Nombre de requêtes désormais **indépendant du nombre de conversations**
  (reliquat = ensure/backfill/dedupe, hors scope C1a).
- **Équivalence fonctionnelle prouvée** : dumps JSON complets avant/après
  byte-à-byte identiques (3 scopes) ; unread_total identiques (875).
- Tests existants : 8/8 PASS (`test_conversation_unread_perf`,
  `test_conversations_inbox`, `test_dispatch_conversation_resolve`).
  Ruff : 0 warning sur les fichiers touchés.

### Contrôles pré-commit (revue 2026-08-27)

1. **ROW_NUMBER tie ordering : PASS** — l'ancien `ORDER BY timestamp DESC`
   n'avait pas de départage (non déterministe sur collision) ; tie-breaker
   `id DESC` ajouté dans la fenêtre **et** dans le chemin unitaire pour que
   les deux chemins choisissent la même ligne.
2. **MessageRead.message_id nullability : PASS** — `nullable=False`
   (FK CASCADE + index unique `(user_id, message_id)`) ; anti-jointure
   proprement équivalente au `NOT IN read_ids` historique.
3. Renforcement : les objets `Booking` préchargés sont **explicitement
   référencés** dans le résultat du batch (clé `booking`) et utilisés par les
   branches mission des builders — plus aucune dépendance à l'identity map
   (références faibles). Effet mesuré : 48 → **23** requêtes (company),
   54 → **29** (driver), parité byte-à-byte reconfirmée.
4. Logs `inbox_perf` : durée + volumes uniquement, aucun contenu de message
   ni identifiant utilisateur.

Commité localement (pas de push sans accord explicite).

Reste à faire (hors C1a) : C1b (psycogreen/psycopg3) uniquement si latence
résiduelle après validation prod de C1a ; gate P1-C1 après C2/C3.

## Candidats de correction (décision à prendre — RIEN n'est appliqué)

1. **C1** : dé-N+1 l'inbox (agrégats SQL : last message par conv via fenêtre,
   unread par jointure MessageRead, statuts booking en un IN) — supprime la
   primitive quelle que soit l'architecture worker.
2. **C1'** : rendre le driver coopératif (psycogreen.patch_psycopg() au boot
   gevent, ou migration dialecte psycopg3 déjà présent dans l'image) — réduit
   la famine mais ne supprime pas le coût CPU N+1.
3. **C2** : stopper le rejeu d'un refresh token révoqué (état terminal local).
4. **C3** : arrêter les pollers driver à la sortie du contexte (gate mobile).
5. **C4** (après C1-C3) : re-tester l'architecture workers / séparation
   API vs Socket.IO.

## Notes harness (annexe)

- La preuve visuelle « company » par empreinte est fragile quand la carte change
  (faux négatif observé, Genève vs ref Annecy). La sonde chrome header
  (`Test-P1ContextChrome`) s'est montrée fiable — à promouvoir en détecteur
  principal pour les prochains gates mobiles.
- Sonde no-DB in-container : `docker exec` urllib passait par un proxy env
  (~5 s constant, artefact) — à refaire via `http.client` direct si besoin.

## Fichiers preuve

- `docs/ops/_mob_ent_p1/c1_sampler.out` (pg_stat_activity live)
- `docs/ops/_mob_ent_p1/c1_traefik_episode.log` + `c1_analyze_episode.ps1`
- `docs/ops/_mob_ent_p1/c1_server_facts.sh` (dialecte, psycogreen, WSGI boot)
- C0 : `docs/ops/p1-c0-attribution-switch-context-2026-08-26.md`
