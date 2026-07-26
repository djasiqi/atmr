# F-03 — Invalidation des anciens liens d'activation

**Statut** : implémenté sur `main`.  
**GO production** : bloqué tant que GO capacité F-02 §18 n’est pas obtenu (ou hotfix depuis SHA prod avec migration sur head Alembic réel).

## Garantie

Après création commitée d’une livraison B, tout jeton de A est définitivement inutilisable — même si le miroir de session porte l’expiration de B, si Celery/Brevo A arrivent tardivement, ou en courses vérif/renvoi.

**Autorité** = `ActivationEmailDelivery` (`token_expires_at`, `superseded_at`, pointeur courant).  
**Miroir** `ActivationSession.email_*` = compat / affichage uniquement.

## Colonnes

- `superseded_at`, `superseded_by_delivery_id` (audit sans FK)
- Index `(activation_session_pk, superseded_at)`
- Migration : `b79c3a9a4958` (`down_revision = d07b29c401ea`)

## Helpers clés

| Fonction | Rôle |
|----------|------|
| `get_activation_session_for_update` | `populate_existing` + `FOR UPDATE` |
| `can_start_new_delivery_snapshot` | lecture seule (route) |
| `expire_stale_sending_under_lock` | lease stale → failed sous verrou |
| `set_current_delivery` | **uniquement** création atomique de B |
| `sync_current_delivery_mirror` | jamais change `email_delivery_id` |
| `mark_delivery_failed` | CAS `queued/sending` + `provider_accepted_at IS NULL` |

## Legacy

```dotenv
ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_FROM_UTC=
ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_UNTIL_UTC=
```

Deux vides = désactivé. Fenêtre ≤ 35 min, UTC aware obligatoire. Module : `services/security/activation_legacy.py`.

## Inventaire pré-deploy (SQL)

```sql
-- Pointeur vers livraison inexistante
SELECT s.id, s.email_delivery_id
FROM activation_session s
WHERE s.email_delivery_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM activation_email_deliveries d
    WHERE d.email_delivery_id = s.email_delivery_id
  );

-- Pointeur inter-session
SELECT s.id, s.email_delivery_id, d.activation_session_pk
FROM activation_session s
JOIN activation_email_deliveries d ON d.email_delivery_id = s.email_delivery_id
WHERE d.activation_session_pk <> s.id;

-- Sessions legacy (pas de livraison HMAC)
SELECT COUNT(*) FROM activation_session s
WHERE s.email_verified_at IS NULL
  AND s.email_delivery_id IS NULL
  AND s.email_token_hash IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM activation_email_deliveries d
    WHERE d.activation_session_pk = s.id
  );
```

Si le dernier COUNT = 0 → laisser FROM/UNTIL vides.

## Déploiement (Alembic-first)

1. Arrêter tous les writers (backend + workers).
2. Infra seule : postgres, pgbouncer, redis.
3. Alembic one-shot image F-03 (`flask db upgrade heads`).
4. Réconciliation : `reconcile_superseded_deliveries()` si besoin.
5. Workers F-03 puis backend F-03.
6. Vérifier zéro conteneur pré-F-03.
7. Smoke : lien A → renvoi B → rejet A → succès B.

**Hotfix** : migration avec `down_revision` = head Alembic **prod** ; au report sur `main`, recréer/merge — ne pas cherry-picker aveuglément.

**Rollback** = roll-forward uniquement (jamais image pré-F-03).

## Logs

```text
activation_email_verify_rejected reason=superseded|expired|invalid|duplicate_hash|legacy_disabled
activation_email_delivery_ignored reason=not_current
failure_ignored reason=already_accepted_or_terminal
```

## Tests

```bash
docker exec atmr-atmr_api python -m pytest \
  tests/security/test_activation_link_supersession_f03.py \
  tests/unit/test_activation_email_delivery.py \
  tests/e2e/test_auth_activation_e2e.py \
  tests/security/test_brevo_webhook_p1.py -q
```

## Critères GO

- [ ] Ancien lien rejeté après renvoi
- [ ] Miroir hors validation HMAC
- [ ] Celery superseded avant claim → pas de Brevo
- [ ] Finalize/webhook tardif → historique OK, pointeur intact
- [ ] Legacy off ou fenêtre absolue ≤ 35 min
- [ ] Suites tests vertes
- [ ] F-02 §18 PASS (ou hotfix séparé)
