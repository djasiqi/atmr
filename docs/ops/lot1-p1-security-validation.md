# Lot 1 P1 Sécurité — RC → production validée

Baseline Lot 0 : `f920d42d` — voir [lot0-p0-security-validation.md](lot0-p0-security-validation.md).

## Verdict courant

| Niveau | Statut | Détail |
|--------|--------|--------|
| Code + tests Docker (RC) | **Vert** | **64 passed** sur SHA figé (rejoué 2026-07-26) |
| Prérequis frontend `www` | **Vert** | `https://www.lirie.ch/activate-account` sert le SPA + `SignupActivation` |
| SHA figé Lot 1 | **Vert** | `4ccd61fa2879318aa946cd18499338f3c38f3419` — **unique candidat déploiement** |
| Production validée | **Bloqué** | Checklist ops secrets → Brevo → backup → migration → smoke → `GO`/`ROLLBACK` |

**Prochain objectif concret :** secrets / Brevo → backup PostgreSQL → migration `16f950e9a85f` → déploiement de `4ccd61fa` → smoke → surveillance → clôture.

---

## 1. Figer la version à déployer

### État figé (2026-07-26)

```text
SHA Lot 1 figé : 4ccd61fa2879318aa946cd18499338f3c38f3419
Message       : feat(security): implement Lot 1 activation and auth hardening
Date gel      : 2026-07-26
Migration     : 16f950e9a85f (incluse et suivie dans le commit)
Tests RC      : 64 passed (rejoués sur ce contenu)
```

- [x] Commit dédié Lot 1 uniquement (31 fichiers)
- [x] Migration `16f950e9a85f` suivie dans le commit
- [x] 64 tests rejoués verts
- [x] Pas de secrets dans le commit
- [x] Ne plus modifier ce SHA (candidat déploiement unique)

### Commande validation Docker (RC)

```bash
docker compose exec -e ENVIRONMENT=production \
  -e REFRESH_FAIL_CLOSED=true -e CSRF_STRICT=true -e CSRF_ENABLED=true \
  -e BREVO_WEBHOOK_SECRET=test-secret-lot1 \
  -e ACTIVATION_TOKEN_KEY_V1=test-activation-key-v1 \
  -e LOGIN_ALLOWED_ORIGINS=https://app.lirie.ch \
  -e JWT_SECRET_KEY=test-jwt-secret-lot1-validation \
  atmr_api python -m pytest \
  tests/unit/test_activation_email_delivery.py \
  tests/security/test_brevo_webhook_p1.py \
  tests/security/test_refresh_fail_closed.py \
  tests/security/test_csrf_strict.py \
  tests/security/test_web_mobile_auth_split.py \
  tests/test_auth_cookies.py -v
```

| Date | Résultat |
|------|----------|
| 2026-07-26 (1er run) | 64 passed |
| 2026-07-26 (rejeu RC→prod) | 64 passed |
| 2026-07-26 (post-commit `4ccd61fa`) | 64 passed |

---

## 2. Prérequis frontend bloquant — vérifié

Route React (code) : `frontend/src/App.js` → `/activate-account` → `<SignupActivation />`.

Vérification live (2026-07-26) :

| Scénario | Résultat |
|----------|----------|
| Accès direct `https://www.lirie.ch/activate-account` | SPA Lirie, titre « Activation du compte », UI email/SMS |
| Accès `...?token=probe-lot1` | Même SPA ; traitement token déclenché |
| Rechargement / nouvel accès direct | SPA toujours servi (pas de 404 serveur) |

- [x] SPA React servi sur `/activate-account`
- [x] Composant `SignupActivation` visible
- [x] Accès direct / refresh sans 404 HTML serveur

---

## 3. Secrets de production

- [ ] Générer `ACTIVATION_TOKEN_KEY_V1` (fort, aléatoire, **≠ JWT/CSRF**)
- [ ] Générer `BREVO_WEBHOOK_SECRET` (fort, aléatoire, **≠ JWT/CSRF**)
- [ ] Sauvegarder hors dépôt (coffre / `.env.production.local` serveur uniquement)
- [ ] Documenter : perdre `ACTIVATION_TOKEN_KEY_V1` invalide les activations en cours

Ne jamais committer ces valeurs.

---

## 4. Variables et flags

Défauts non secrets ajoutés dans `scripts/env.production.defaults.fragment` :

```env
CSRF_ENABLED=true
CSRF_STRICT=true
REFRESH_FAIL_CLOSED=true
LOGIN_ALLOWED_ORIGINS=https://www.lirie.ch,https://lirie.ch
```

Cookies déjà prévus dans le fragment : `COOKIE_SECURE=true`, `COOKIE_HTTP_ONLY=true`, `COOKIE_SAME_SITE=Strict`.

Contrôles manuels prod :

- [ ] `ACTIVATION_TOKEN_KEY_V1` / `BREVO_WEBHOOK_SECRET` présents dans `.env.production.local`
- [ ] Origines exactes incluent `https://www.lirie.ch`
- [ ] `FRONTEND_URL=https://www.lirie.ch/` (URL canonique d’activation)
- [ ] Identité d’envoi LIRIE (Brevo sender / reply-to)
- [ ] URL publique webhook : `https://api.lirie.ch/api/v1/webhooks/brevo`

---

## 5. Configurer Brevo

```text
POST https://api.lirie.ch/api/v1/webhooks/brevo
Authorization: Bearer <BREVO_WEBHOOK_SECRET>
```

Événements : delivered, soft_bounce, hard_bounce, spam/complaint, blocked, invalid_email.

- [ ] Webhook créé / mis à jour
- [ ] Bearer aligné avec `BREVO_WEBHOOK_SECRET`
- [ ] Événements sélectionnés
- [ ] Note ID webhook Brevo (non sensible) : ______________

---

## 6. Sauvegarder PostgreSQL

Avant migration :

- [ ] Sauvegarde datée créée
- [ ] Fichier non vide (taille vérifiée)
- [ ] Commande de restauration documentée
- [ ] (Idéal) test de lecture / restore dry-run

```text
Fichier backup : ______________________________
Checksum / taille : ______________________________
Restore cmd : ______________________________
```

---

## 7. Déployer + migration `16f950e9a85f`

- [ ] Déployer le **SHA figé** uniquement
- [ ] Appliquer `16f950e9a85f` **une seule fois**
- [ ] `flask db current` = `16f950e9a85f` (ou head l’incluant)
- [ ] Tables `activation_email_deliveries`, `brevo_webhook_events`
- [ ] `UNIQUE` sur `brevo_webhook_events.idempotency_key`
- [ ] Index livraison / token_hash présents
- [ ] Pas de sessions/livraisons historiques incohérentes bloquantes

---

## 8. Smoke tests réels

- [ ] Inscription web
- [ ] Réception / ouverture email
- [ ] Activation depuis `www.lirie.ch`
- [ ] Renvoi email + cooldown
- [ ] Webhook Brevo rattaché (`X-Mailin-custom` / messageId)
- [ ] Login web : pas de tokens JSON / pas de `localStorage` Bearer
- [ ] Refresh web avec CSRF
- [ ] Refresh mobile avec `refresh_token` JSON (cookies piégés ignorés)
- [ ] `401` / `403` / `503` conformes (dont Redis down → 503, jamais JWT-only)

---

## 9. Surveillance post-déploiement (premières heures)

- [ ] Erreurs inscription / activation
- [ ] Pics `401` / `403` / `503`
- [ ] Rejets Origin/Referer ou CSRF
- [ ] Webhooks Brevo inconnus / rejetés
- [ ] Livraisons bloquées en `sending`
- [ ] Hard bounces, spam, doubles envois post-201
- [ ] Erreurs Celery / Redis / PostgreSQL

---

## 10. Clôture officielle Lot 1

| Champ | Valeur |
|-------|--------|
| SHA déployé | |
| Heure déploiement (UTC) | |
| Sauvegarde réalisée | oui / non — chemin |
| Migration appliquée | `16f950e9a85f` oui / non |
| Smoke tests | OK / KO (détail) |
| Webhook Brevo (id non sensible) | |
| Décision finale | `GO` / `ROLLBACK` |
| Opérateur | |
| Notes | |

Après `GO` uniquement : Lot 1 = **production validée** → Lot 2 autorisé.

---

## Périmètre technique (implémenté)

| ID | Objet |
|----|--------|
| L1-A | HMAC `ACTIVATION_TOKEN_KEY_V1` ; finalisation `provider_accepted_at` |
| L1-B | Webhook Brevo Bearer + `ON CONFLICT` + txn atomique |
| L1-C | Refresh fail-closed 503/401 |
| L1-E | Web cookies-only ; mobile Bearer + JSON refresh |
| L1-D | `CSRF_STRICT` + Origin/Referer login |

## Fichiers clés

- `backend/models/activation_email_delivery.py`
- `backend/services/notifications/activation_token.py`
- `backend/services/notifications/activation_email_delivery.py`
- `backend/services/notifications/brevo_webhook.py`
- `backend/routes/webhooks_brevo.py`
- `backend/services/security/csrf.py`
- `backend/services/security/login_origin.py`
- `backend/security/refresh_token_service.py`
- `backend/migrations/versions/16f950e9a85f_lot1_activation_email_deliveries_brevo_.py`
- `scripts/env.production.defaults.fragment`
