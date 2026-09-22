"""SQL AUTH-SMS-02 : promotion des CLIENT/PORTAL déjà e-mail vérifiés.

Aucun import Flask. La migration Alembic et le script de prévisualisation
doivent utiliser **exactement** ces prédicats.
"""

from __future__ import annotations

# Compte PORTAL, rôle CLIENT, e-mail présent, pas institution, pas désactivé,
# session d'activation avec e-mail confirmé. Ne touche jamais phone_verified_at.
PORTAL_PROMOTION_USER_PREDICATE = """
    u.account_status = 'pending_activation'
    AND u.role = 'CLIENT'
    AND u.institution_id IS NULL
    AND u.disabled_at IS NULL
    AND NULLIF(BTRIM(COALESCE(u.email, '')), '') IS NOT NULL
    AND EXISTS (
        SELECT 1
        FROM activation_session AS s
        WHERE s.user_id = u.id
          AND s.email_verified_at IS NOT NULL
    )
    AND EXISTS (
        SELECT 1
        FROM client AS c
        WHERE c.user_id = u.id
          AND c.client_type = 'PORTAL'
    )
"""

COUNT_PORTAL_PROMOTION_USERS_SQL = f"""
SELECT COUNT(*)
FROM "user" AS u
WHERE {PORTAL_PROMOTION_USER_PREDICATE}
"""

# Pas de phone_verified_at ici : le preview tourne avant la migration qui
# ajoute la colonne. Le COUNT / UPDATE restent les mêmes prédicats.
LIST_PORTAL_PROMOTION_USERS_SQL = f"""
SELECT u.id, u.email, u.account_status
FROM "user" AS u
WHERE {PORTAL_PROMOTION_USER_PREDICATE}
ORDER BY u.id
LIMIT :limit
"""

PROMOTE_PORTAL_USERS_SQL = f"""
UPDATE "user" AS u
SET account_status = 'active'
WHERE {PORTAL_PROMOTION_USER_PREDICATE}
"""

COUNT_PORTAL_PROMOTION_CLIENTS_SQL = """
SELECT COUNT(*)
FROM client AS c
JOIN "user" AS u ON u.id = c.user_id
WHERE c.client_type = 'PORTAL'
  AND c.is_active IS NOT TRUE
  AND u.account_status = 'pending_activation'
  AND u.role = 'CLIENT'
  AND u.institution_id IS NULL
  AND u.disabled_at IS NULL
  AND NULLIF(BTRIM(COALESCE(u.email, '')), '') IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM activation_session AS s
      WHERE s.user_id = u.id
        AND s.email_verified_at IS NOT NULL
  )
"""

PROMOTE_PORTAL_CLIENTS_SQL = """
UPDATE client AS c
SET is_active = true
FROM "user" AS u
WHERE c.user_id = u.id
  AND c.client_type = 'PORTAL'
  AND c.is_active IS NOT TRUE
  AND u.account_status = 'active'
  AND u.role = 'CLIENT'
  AND u.institution_id IS NULL
  AND u.disabled_at IS NULL
  AND NULLIF(BTRIM(COALESCE(u.email, '')), '') IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM activation_session AS s
      WHERE s.user_id = u.id
        AND s.email_verified_at IS NOT NULL
  )
"""

COUNT_AFTER_PROMOTION_PENDING_SQL = COUNT_PORTAL_PROMOTION_USERS_SQL
