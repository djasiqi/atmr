# Verrouillage rôle ADMIN plateforme

## Problème

Un compte `ADMIN` (ex. `jasiqi.drin@gmail.com`) pouvait aboutir dans l’UI / le JWT d’un espace **entreprise transport**, alors qu’il doit rester strictement sur le périmètre admin LIRIE.

Causes principales :

1. **Frontend** : `writeAuthSession` rangeait les admins sous `company_user` et réécrivait parfois `role` en `company`.
2. **Frontend** : `ProtectedRoute` acceptait une session « company » polluée pour le même `public_id` qu’un admin.
3. **Backend** : `_resolve_company_id` pouvait poser un `company_id` JWT si `Company.user_id` pointait vers l’admin ; le login mobile entreprise acceptait parfois `ADMIN`.

## Règles verrouillées

| Couche | Comportement |
|--------|----------------|
| JWT | `company_id = null` si `role == ADMIN` |
| Contextes mobile | un seul contexte `admin` ; pas de switch vers company |
| `GET` entreprise courante | `403` pour ADMIN |
| Login entreprise (mobile) | filtre `COMPANY` uniquement |
| Promotion → ADMIN | détache `Company.user_id` |
| Session web | scope `admin_user` dédié ; jamais `company_user` |
| Routes `/dashboard/company/*` | refus si `admin_user` présent ou pollution même `public_id` |

## Après déploiement

1. Se déconnecter complètement.
2. Vider le localStorage des clés `admin_*`, `company_*`, `app_*`, `authToken`, `user`.
3. Se reconnecter → redirection `/dashboard/admin/<public_id>`.

## Fichiers clés

- `frontend/src/utils/webAuthSession.js`
- `frontend/src/utils/ProtectedRoute.jsx`
- `backend/routes/auth.py` (`_resolve_company_id`, contexts, switch-context)
- `backend/routes/company_mobile_auth.py`
- `backend/application/users/get_current_company.py`
- `backend/services/admin_account_role_transition.py`
