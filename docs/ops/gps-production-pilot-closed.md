# Production pilote fermée — GPS / flotte (5 chauffeurs)

Décision figée : **2026-08-23**.

Références : [`gps-fleet-e2e-certification.md`](./gps-fleet-e2e-certification.md) · [`gps-driver-state-certification.md`](./gps-driver-state-certification.md) · [`gps-product-contract-canary-matrix.md`](./gps-product-contract-canary-matrix.md).

## Décision

```text
PRODUCTION PILOTE 5 CHAUFFEURS     = GO ✅
DÉPLOIEMENT LARGE                  = HOLD
NOUVELLES ENTREPRISES              = au cas par cas / pilote
GO PROD GÉNÉRAL                    = NO-GO (pour l’instant)
```

**Position** : la version actuelle est suffisamment validée (mono-device + premiers comportements multi-chauffeurs) pour **observer le système en conditions réelles** avec les **5 chauffeurs existants**, entreprise et institution actuelles — **canary réel**, pas certification « flotte illimitée ».

## Périmètre pilote

| Élément | Règle |
|---------|--------|
| Chauffeurs | **Exactement les 5 actuels** (pas d’expansion silencieuse) |
| Entreprise / institution | Périmètre actuel uniquement |
| Type | Pilote **fermé** et **surveillé** |
| Prospection | **Non bloquée** — viser 1–2 nouvelles entreprises/institutions **pilotes** en parallèle |

## Surveillance pilote (priorités)

- Chauffeurs en service réellement visibles sur la carte entreprise
- Fraîcheur `recorded_at` / ages DB
- Disparition prolongée d’un chauffeur (sans cause métier)
- Recovery automatique après réseau / HOME / verrouillage
- `CROSS_DRIVER` / `CROSS_SESSION` = 0
- Transitions mission correctes
- Interventions manuelles nécessaires (ops / support)
- Retours chauffeurs : batterie, permissions, compréhension du suivi
- Stabilité carte côté entreprise (`2/N en direct`, cohérence projection)

## Règle de décision

```text
Si les 5 chauffeurs fonctionnent plusieurs jours
  sans défaut critique ni intervention répétée
  → pilote confirmé ✅

Si un défaut critique apparaît
  → expansion HOLD
  → RCA
  → les 5 peuvent rester en pilote si le risque est maîtrisé
```

**Défaut critique** (exemples) : crossover identité, tracking arrêté sans cause, carte fausse durable, perte de données mission, incident sécurité / consentement.

## Distinction GO PROD

```text
GO PROD PILOTE FERMÉ (5)     ≠ GO PROD GÉNÉRAL

GO PROD GÉNÉRAL exige encore :
  FLEET-2 E2E complet (F05–F20)
  flotte 10/10 puis 20/20
  matrice recovery / iOS
  critères docs/ops/gps-fleet-e2e-certification.md
```

## Statut certification (lien)

```text
DRIVER STATE CERT (C01–C11)     = PASS ✅
FLEET-2 (lab)                   = OPEN (F00–F04 PASS · F05+ en cours)
DEVICE B2/B3                    = CLOSED ✅
```

Le pilote production **ne remplace pas** la certification flotte large ; il **alimente** la décision suivante avec données réelles.

**Avant / pendant pilote** : rejouer le canary Samsung 30 min — [`gps-samsung-canary-30min-plan.md`](./gps-samsung-canary-30min-plan.md).
