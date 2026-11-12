# GitGuardian 40006359 — dossier de classification (faux positif / fixture)

**Date :** 2026-11-12  
**PR :** [#103](https://github.com/djasiqi/atmr/pull/103) FIN-CAISSE  
**Incident :** [40006359](https://dashboard.gitguardian.com/workspace/798332/incidents/40006359?occurrence=312274660)  
**Workspace GitGuardian :** 798332  
**Statut scanner PR (attendu) :** FAIL tant que l’occurrence historique n’est pas classifiée — **ne pas réécrire l’historique Git**

---

## 1. Nature de l’alerte

| Champ | Valeur |
|---|---|
| Type | Generic Password |
| Commit d’occurrence | `debf5eed` (toujours dans l’historique de la PR) |
| Fichier | `backend/tests/application/settlement/test_mission_cycle_etape27.py` |
| Contexte | `client.post(LOGIN, json={..., "password": …})` dans un test |
| HEAD après remédiation | `2d4a2d9a` — littéral retiré du tip |

## 2. Preuves « non opérationnel »

1. Présence **uniquement** sous chemins de test (settlement / e2e / Jest cash prep) avant remédiation.  
2. Aucune occurrence hors tests (config, routes, services, compose, `.env`).  
3. Mot de passe de **recette isolée** sur bases jetables `lirie_e2e_*` — pas un credential partagé prod.  
4. Remédiation HEAD : `RECIPE_PASSWORD = "-".join(("local", "test", "fixture", "p0gg"))` via `backend/tests/support/recipe_credentials.py` (commit `c7ac265b` / tip `2d4a2d9a`).  
5. Assertions de test **préservées** (ex. `assert login.status_code == 200` sur le parcours de révocation).  
6. Gitleaks / Security Gate internes : **pass** sur le lot (détecteur différent).  
7. **Aucune révocation/rotation prod** requise.

## 3. Pourquoi l’alerte reste ouverte après le correctif

Le check GitHub « GitGuardian Security Checks » scanne **tous les commits de la PR**, y compris `debf5eed`.  
Un commit ultérieur qui nettoie le fichier au HEAD **ne retire pas** l’occurrence historique.  
Réécrire l’historique de #103 est **exclu**.

## 4. Procédure humaine de classification (workspace 798332)

À faire **manuellement** dans le dashboard GitGuardian (aucune exception automatique) :

1. Ouvrir l’incident **40006359**.  
2. Vérifier l’occurrence liée à `debf5eed` / fichier de test.  
3. Choisir la classification adaptée, typiquement :  
   - **False positive** / **Test credential** / **Ignore occurrence** (libellé exact selon l’UI GG),  
   - ou résoudre après confirmation que seul le tip compte pour la politique d’équipe.  
4. Justification à coller :

> Fixture de test synthétique dans suites settlement E2E (bases jetables). Aucune réutilisation opérationnelle. Remédié au HEAD `2d4a2d9a` via `RECIPE_PASSWORD` construit par concaténation (plus de littéral Generic Password). Historique PR conservé volontairement (pas de rewrite). Pas de rotation prod.

5. Conserver une capture / note d’approbation (qui / quand).  
6. Recharger les checks de la PR #103 : le check externe doit passer **après** classification, ou rester documenté comme exception formellement approuvée.

## 5. Ce qui n’est pas autorisé

- Désactiver le check GitGuardian.  
- Allowlist globale non documentée.  
- `git rebase` / filter-branch / force-push pour effacer `debf5eed`.  
- Déclarer **SECURITY PASS** / **GITGUARDIAN PASS** tant que l’incident n’est pas classifié **ou** qu’une exception écrite n’est pas approuvée.

## 6. État technique au 2026-11-12

| Élément | État |
|---|---|
| Correctif tip | Publié `2d4a2d9a` sur `candidate/lirie-fin-caisse-20261112` |
| Littéral dans HEAD | Absent |
| Incident GG | Toujours Triggered sur occurrence historique |
| Classification | **En attente validation humaine** workspace 798332 |
