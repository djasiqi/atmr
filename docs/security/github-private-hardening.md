# GitHub privé — durcissement LIRIE

Le monorepo `djasiqi/atmr` est le dépôt **principal**. Il doit rester
**privé**. Ne pas publier backend, mobile, infrastructure ni l’historique
des lots sécurité.

Un dépôt privé n’est pas une preuve de sécurité du code. Il réduit
seulement la surface d’exposition. Le code doit rester défendable même
s’il est lu.

## État appliqué (2026-09-13)

| Contrôle | État |
|----------|------|
| Visibilité | **private** |
| Collaborateurs | 1 : `djasiqi` (admin) |
| Forks publics | 0 au moment du passage en privé |
| Teams | aucune |
| Deploy keys | 0 |
| Webhooks | 0 |
| Dependabot | activé, 0 open |
| Secret scanning | activé, 0 open |
| Secret scanning push protection | activé |
| CodeQL | Python + JS/TS + Actions (baseline `4fe9d96d`) |
| Actions `default_workflow_permissions` | **read** |
| Actions `can_approve_pull_request_reviews` | false |
| Actions `allowed_actions` | `all` (marketplace : à restreindre plus tard) |
| Protection `main` (ruleset / branch protection) | **indisponible** sur compte perso Free + repo privé |
| Environnements `Production` / `Preview` | existent, **sans** reviewers |

Passage en privé : API GitHub `visibility=private` (compte `djasiqi`).
`allow_forking=false` n’est pas applicable aux dépôts perso (422).

## Secrets déjà vus par GitHub

14 alertes secret scanning **résolues** (aucune ouverte) :

- clés Google API : 8 marquées `revoked`, 4 `used_in_tests`
- clés OpenWeather : 2 `used_in_tests`

Règle : un secret qui a existé dans Git est **compromis**. Le retirer
du fichier ne suffit pas. Il faut **révoquer et régénérer** côté
fournisseur, puis ne plus le committer (GitHub Secrets / secret manager).

Ne pas republier ces valeurs. Ne pas « nettoyer l’historique » sans
plan de rotation : un rewrite Git ne retire pas un secret déjà copié.

## Recommandé, pas encore imposé

Ces points casseraient le flux actuel (push direct admin sur `main`)
ou exigent **GitHub Pro** / une **organisation** :

```text
MFA obligatoire sur le compte GitHub
protection de main (PR obligatoire, CI required, pas de force-push)
pas de push direct sur main (sauf bypass admin temporaire)
environnements staging / production avec reviewers
Actions : allowed_actions = github + verified seulement
revue périodique des collaborateurs et clés SSH
révocation immédiate des accès à la sortie d’un collaborateur
```

Tant que le compte reste Free + perso :

- la visibilité privée est la barrière principale ;
- `main` n’a **pas** de lock GitHub (force-push possible par l’admin) ;
- passer en **org LIRIE + GitHub Team/Pro** dès qu’il y a plus d’un
  contributeur, puis activer rulesets.

## Accès

Aujourd’hui : **un seul admin**. C’est le bon point de départ.

À chaque changement d’équipe, vérifier :

```text
gh api repos/djasiqi/atmr --jq "{private,visibility}"
gh api repos/djasiqi/atmr/collaborators --jq ".[] | {login,role_name}"
gh api repos/djasiqi/atmr/keys
```

## Actions et secrets

Permissions workflow par défaut : **lecture**. Conserver.

Les secrets d’Actions / Vercel / EAS restent hors Git. Les
environnements GitHub `Production – atmr` et `Production – lirie-app`
sont créés par Vercel **sans** règle de protection : les ajouter
seulement après un test de deploy, pour ne pas bloquer la prod.

## Dépôts éventuellement publics plus tard

Uniquement s’ils sont conçus pour l’extérieur, **hors** de ce monorepo :

```text
lirie-public-docs
lirie-sdk
lirie-api-examples
```

Pas le backend, pas le mobile, pas l’infra, pas ce dépôt.

## Réaction

```text
nouveau collaborateur → rôle minimal, MFA, revue
départ → retirer l’accès le jour même
nouvelle alerte secret → révoquer + régénérer, ne pas seulement dismiss
repo redevenu public → incident : repasser privé immédiatement
```

Baseline CodeQL : `docs/security/codeql-baseline.md`.
