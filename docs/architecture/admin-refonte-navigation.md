# Refonte Administration LIRIE — Navigation (PR1)

## Règle de compatibilité PR1

Le nouveau shell réutilise uniquement les capacités Platform Ops déjà existantes (`usePlatformCapabilities` / `PlatformSegmentGuard`). Il n’introduit aucune permission `admin.*`. La généralisation des capacités administratives reste exclusivement réservée à PR2bis.

## Architecture

```text
AdminShell
├── AdminSidebar          ← 6 workspaces
├── AdminTopbar           ← titre du workspace
├── AdminWorkspaceNav     ← children du registre (+ platformCapability)
└── Outlet
```

Fichiers :

- [`frontend/src/pages/admin/navigation/adminNavRegistry.js`](../../frontend/src/pages/admin/navigation/adminNavRegistry.js)
- [`frontend/src/pages/admin/routing/adminRoutePaths.js`](../../frontend/src/pages/admin/routing/adminRoutePaths.js)
- [`frontend/src/pages/admin/routing/adminLegacyRedirects.js`](../../frontend/src/pages/admin/routing/adminLegacyRedirects.js)

## Workspaces cibles

| Workspace | Path | Sous-pages |
|-----------|------|------------|
| Vue d’ensemble | `/` | — |
| Opérations | `operations` | `bookings`, `bookings/:bookingId` |
| Partenaires | `partners` | `users`, `demo-requests` |
| Finance | `finance` | index, `releves`, `config` |
| Configuration | `configuration` | (AdminSettings temporaire) |
| Outils avancés | `advanced` | `platform/*` (7 segments), `labs/shadow-mode`, `labs/optuna` |

## Redirections legacy

| Legacy | Cible |
|--------|-------|
| `reservations` | `operations/bookings` |
| `reservations/:bookingId` | `operations/bookings/:bookingId` |
| `users` | `partners/users` |
| `demo-requests` | `partners/demo-requests` |
| `billing`, `billing/releves`, `billing/config` | `finance…` |
| `settings` | `configuration` |
| `shadow-mode` / `optuna` | `advanced/labs/…` |
| `platform-ops/:segment` | `advanced/platform/:segment` |

Search, hash et state React Router sont conservés.

## Tokens

Variables `--admin-*` dans [`adminTokens.css`](../../frontend/src/pages/admin/shell/adminTokens.css). Largeur sidebar : `--admin-sidebar-width` (CSS only).

## Critères PR1

- ✅ **Implémenté** : 6 workspaces, shell grille, AdminWorkspaceNav unique, redirects dynamiques, PlatformSegmentGuard conservé, menu user sans liens client, sélecteur date dashboard retiré, helpers `adminPaths`.

## Critères PR2

- ✅ **Implémenté** : `return` après ouverture sélecteur chauffeur ; `fetchUsers` propage les erreurs ; `AdminTempPasswordDialog` (secret hors alert/toast/console) ; `AdminActionDialog` (contrat strict) pour users + actions billing critiques (lock, delete support, reopen, void agreement).

## Critères PR2bis

- ✅ **Implémenté** : `services/admin_authz.py` + flag `ADMIN_CAPABILITIES_ENFORCED` (défaut `false`) ; grants `admin.*` via `platform_admin_permission_grant` ; logs `admin_capability_would_deny` ; endpoint `GET /admin/capabilities` ; garde Optuna + billing lock/issue ; hook `useAdminCapabilities` ; filtre labs dans `AdminWorkspaceNav` ; bouton Verrouiller conditionné.

## Durcissement post-merge (`fix/admin-post-merge-hardening`)

- ✅ **Implémenté** : alignement FE/BE sur `enforced` (`hasAdminCapability` ignore la liste en compat) ; mode enforced = grants uniquement (sans grants ⇒ aucune capacité) ; `AdminCapabilityGuard` sur routes Labs ; `canBillingIssue` sur « Générer PDF/QR » ; cycle de vie `AdminActionDialog` (reset uniquement à l’ouverture) ; reset MDP valide avant fermeture ; overlay secret non cliquable ; tests ciblés.

## Backlog

- **PR3** : opérations / partenaires (fiches org, pipeline démos)
- **PR4** : finance pages dédiées
- **PR5** : split configuration, Platform Ops 4 écrans, labs
- **PR6** : React Query, split adminService, E2E, budgets perf
