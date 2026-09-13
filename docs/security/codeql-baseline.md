# Baseline sécurité CodeQL — remise à zéro

Le chantier de remise à zéro de la dette CodeQL est **clos**.
Ce document fige l’état reproductible. Il n’autorise aucun lot
« d’amélioration » sur du code sain sans nouvelle menace identifiée.

## Référence

```text
REFERENCE SHA :
4fe9d96db16dc31630eef6b44559acc89a066108

DATE DE SCAN :
2026-09-12 (analyses CodeQL sur ce SHA)

HEAD au moment de la baseline :
4fe9d96db16dc31630eef6b44559acc89a066108
```

```text
BASELINE SECURITY :
CodeQL .................... 0 open
Dependabot ................ 0 open
Secret scanning ........... 0 open
Bandit / Semgrep .......... GREEN
pip-audit ................. GREEN
Secret Pattern Scan ....... GREEN
CI ........................ GREEN
```

## Exécution CodeQL (à conserver)

CodeQL n’est **pas** un workflow YAML du dépôt. Il s’exécute via
GitHub code scanning (setup default / analyses `codeql:analyze`).

Sur le SHA de référence, les trois analyses ont bien tourné :

| Langage | Catégorie | Résultat |
|---------|-----------|----------|
| Python | `/language:python` | GREEN |
| JavaScript/TypeScript | `/language:javascript-typescript` | GREEN |
| Actions | `/language:actions` | GREEN |

Si une de ces trois analyses disparaît d’un scan `main`, ce n’est pas
une « amélioration » : c’est une **régression de couverture**.
La rétablir avant tout autre lot sécurité.

## Autres gates (dépôt)

| Gate | Où | Déclenchement |
|------|----|----------------|
| pip-audit + Semgrep (hebdo / requirements) | `.github/workflows/security-scan.yml` | push/PR sur requirements, cron lundi 03:00 UTC |
| Bandit + Semgrep + pip-audit | `.github/workflows/backend-tests.yml` | push/PR `backend/**` |
| Secret Pattern Scan (clés Google) | `.github/workflows/secret-pattern-scan.yml` | tout push / PR |
| Secret scanning GitHub | GitHub Advanced Security | continu |

## Familles CodeQL traitées (0 open)

Corrigées ou classées. Ne pas les rouvrir sans **nouvelle** preuve
de régression sur le SHA courant.

```text
py/clear-text-logging-sensitive-data
py/path-injection
py/stack-trace-exposure
py/reflective-xss
py/bad-tag-filter
py/polynomial-redos
py/weak-sensitive-data-hashing
py/overly-large-range
py/incomplete-url-substring-sanitization

js/clear-text-storage-of-sensitive-data
js/incomplete-url-substring-sanitization
js/insecure-randomness
js/xss-through-dom
```

Faux positifs démontrés (dismiss GitHub, preuves distinctes) :

| ID | Règle | Preuve |
|----|-------|--------|
| 26 | `js/unvalidated-dynamic-method-call` | `loader()` = `import()` issus de `ROUTE_CHUNK_LOADERS` (map fermée). `item.path` sidebar hardcodé. Pas `object[userInput]()`. |
| 42 | `js/incomplete-multi-character-sanitization` | `prerender-public-pages.mjs` build-time. Dédup JSON-LD, pas sanitizer XSS. HTML = React local, routes figées. |

Preuves hashing : `docs/security/py-weak-sensitive-data-hashing.md`.

## Dette de conception documentée (hors CodeQL)

`#245` / `_get_password_hash_version` : l’extrait 16 caractères d’un
hash Werkzeug est le préfixe d’algorithme, pas un signal de changement
de mot de passe. L’autorité d’invalidation reste `token_version` +
révocation des sessions. **Ne pas « réparer » sans besoin produit.**

## Politique pour une nouvelle alerte

Ne pas corriger « parce que CodeQL a parlé ».

```text
nouveau finding
→ comparer à cette baseline (SHA + familles)
→ tracer source → validation → sink
→ la sécurité du système dépend-elle de ce finding ?
→ classifier :
    REAL SECURITY RISK
    QUALITY / HARDENING
    FALSE POSITIVE
    BUILD / TEST ONLY
→ corriger seulement si nécessaire
→ tests de non-régression des familles ci-dessus
→ CodeQL / Dependabot / secrets restent à 0 sauf alerte réellement ouverte
```

Interdit :

```text
changer un hash / une regex / un parser
uniquement pour faire descendre un compteur
```

```text
MD5 ≠ vulnérable par son nom
SHA-256 ≠ sûr par son nom
```

## Objectif après cette baseline

Revenir au produit LIRIE.

Un lot sécurité ne se justifie que par :

```text
nouvelle alerte réelle
ou
régression de couverture (langage CodeQL manquant)
ou
Dependabot / secret scanning > 0
```
