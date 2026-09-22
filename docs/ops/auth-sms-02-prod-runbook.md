# AUTH-SMS-02 — PROD RUNBOOK

Périmètre fonctionnel **figé**. Aucune autre logique avant le smoke production.

Twilio reste **OFF**. Ne pas lancer `deploy.yml` **avec déploiement**
(`skip_deploy=false`) : le job serveur exécuterait `flask db upgrade`.
Le build d’image seul est autorisé : `workflow_dispatch` + `skip_deploy=true`.

**Garde-fou SHA** : l’image `sha-<12>` et `/srv/atmr/scripts/deploy-production.sh`
doivent provenir du **même commit**. Pas de nouvelle image avec un ancien
script, ni l’inverse.

L’entrypoint prod ne migre pas (`RUN_ENTRYPOINT_MIGRATIONS=0`). Le risque
d’écriture automatique est uniquement dans `scripts/deploy-production.sh`.

```text
AUTH-SMS-02
===========

EMAIL → ACTIVATION ................ PASS
LOGIN SANS SMS .................... PASS code/tests
SMS RETIRÉ DU SIGNUP .............. PASS
PHONE_VERIFIED_AT SOURCE VÉRITÉ ... PASS
GATE ROUTE ........................ PASS
GATE USE CASE ..................... PASS
CHANGEMENT TÉLÉPHONE .............. PASS
NORMALISATION E.164 ............... PASS
PREVIEW MIGRATION READ-ONLY ....... PASS
INSTITUTIONS HORS SCOPE ........... PASS
TRANSPORT HORS SCOPE .............. PASS

PREVIEW PROD ...................... PASS 2026-09-22 (SQL read-only, 1 = Philippe 192847)
POPULATION ........................ PASS
COMPTES INATTENDUS ................ 0
CODE PROD ......................... OLD (sha-fca85737fb7b)
MIGRATION ......................... NOT RUN
TWILIO ............................ OFF volontairement

VERDICT :
GO POUR PUBLICATION AUTH-SMS-02
NO-GO MIGRATION TANT QUE LE NOUVEAU CODE N'EST PAS EN EXÉCUTION
```

## Séquence

```text
1. COMMIT / PUSH AUTH-SMS-02
   + migration 395bd3663e8d
   + preview
   + deploy-production.sh avec SKIP_DB_UPGRADE

2. CI VERTE

3. BUILD / PUBLICATION IMAGE
   deploy.yml skip_deploy=true
   image sha-<12> du même commit

4. SCRIPT SERVEUR = même SHA
   /srv/atmr/scripts/deploy-production.sh
   vérifier SKIP_DB_UPGRADE reconnu

5. DEPLOY CODE SEUL
   SKIP_DB_UPGRADE=1
   DOCKER_TAG=sha-<12> du commit

6. CONTRÔLES
   running image = nouveau SHA
   alembic current = encore 7575a80bda48
   SMS_NOTIFICATIONS_ENABLED=false
   aucune migration

7. PREVIEW DEPUIS LE NOUVEAU CODE
   users_a_promouvoir = 1
   uniquement Philippe 192847

8. GO MIGRATION
   flask db upgrade heads

9. CONTRÔLES APRÈS
   alembic = 395bd3663e8d
   users_a_promouvoir = 0
   Philippe active + phone_verified_at NULL

10. FINALISER
    deploy-production.sh sans SKIP
    upgrade no-op + Celery + smokes

11. SMOKE AUTH-SMS-02

12. CLOSED PROD

13. ENSUITE SEULEMENT Twilio
```

## Commandes serveur

Conteneur prod = `backend` (pas `atmr_api`).

Entre l’étape 1 et l’étape 3, le nouveau code est déjà levé, la colonne
`phone_verified_at` n’existe pas encore. Les SELECT ORM `User` peuvent
échouer. Celery n’est **pas** démarré. Relire le COUNT puis upgrader
tout de suite.

### 1. Déployer le code sans migration

```bash
SKIP_DB_UPGRADE=1 /srv/atmr/scripts/deploy-production.sh
```

Le script lance le preview read-only dans les logs, **sans**
`flask db upgrade`, **sans** Celery ni smoke. Relire `users_a_promouvoir`
et l’échantillon (Philippe 192847). Le preview reste lisible même si la
colonne n’existe pas encore (`colonne phone_verified_at : absente`).

Ne pas lancer `.github/workflows/deploy.yml` : il n’envoie pas
`SKIP_DB_UPGRADE` et écrirait tout de suite.

### 2. Preview manuel (si besoin de relancer)

```bash
docker compose -f /srv/atmr/docker-compose.production.yml exec -T \
  -e DISABLE_EVENTLET=1 backend \
  python -m scripts.preview_auth_sms_02_promotion
```

Mêmes prédicats que la migration : `pending_activation` + `CLIENT` +
`PORTAL` + e-mail + session e-mail confirmée + `institution_id IS NULL` +
`disabled_at IS NULL`. Jamais d’écriture de `phone_verified_at`.

### 3. Migration (après revue du COUNT)

```bash
docker compose -f /srv/atmr/docker-compose.production.yml exec -T \
  -e DISABLE_EVENTLET=1 backend \
  flask db upgrade heads
```

### 4. Preview après

Même commande qu’en 2. Attendu : `users_a_promouvoir = 0` et
`colonne phone_verified_at : presente`.

### 5. Terminer le déploiement

```bash
/srv/atmr/scripts/deploy-production.sh
```

Sans `SKIP_DB_UPGRADE` : upgrade no-op (déjà au head), puis Celery,
healthcheck et smoke.

Philippe : `account_status=active`, `phone_verified_at=NULL`, login OK,
aucun SMS.

## Après CLOSED PROD

`AUTH-SMS-01 PROD ENABLEMENT` : credentials Twilio, sender, numéro
d’équipe, SMS réel, OTP, création transport. Puis seulement le parcours
Philippe. `AUTH-SMS-03` / Verify plus tard.
