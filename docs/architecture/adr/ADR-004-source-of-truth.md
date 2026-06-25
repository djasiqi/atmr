# ADR-004 — Source of Truth flux GPS

## Statut

Accepté — 2026-06-25

## Contexte

Audit 2026-06-25 : pipeline aval sain, rupture mobile amont.

## Décision

Documenter et faire respecter le flux canonique (voir `tracking-architecture-contract.md`). Aucun composant hors flux ne modifie lat/lng en production.

## Conséquences

- Architecture contract tests CI bloquants
- ADR lié aux invariants INV-4, INV-5, INV-6
- Suppression progressive des chemins legacy Sprint 4
