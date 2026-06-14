# STOP GATE TR-01 — TransportRequestDisplayModel v1

```txt
Status: PASS (auto)
Date: 2026-06-12
Prérequis: P2.1 build_transport_request_display_blocks()
Bloque: branchement serializer TR, offres entreprise, exports institution
```

## Objectif

Certifier `display_model: "transport_request"`, `display_model_version: 1` avant branchement consommateurs.

## Cas de validation

| ID | Configuration | Résultat attendu | Auto |
|----|---------------|------------------|------|
| TR-01a | Départ confirmé 13:15 | `departure.display_time` = `13:15`, `time_defined=true` | ✅ |
| TR-01b | Départ indicatif | `departure.time_defined=false` | ✅ |
| TR-01c | Retour non confirmé | `return.display_time` = « À définir », summary contient « Retour à définir » | ✅ |
| TR-01d | Multi-stop avec legs confirmés | `legs[]` ordonnés, `display_time` par leg | ✅ |
| TR-01e | Identité institution | `primary_label` patient, `secondary_label` institution | ✅ |

## Validation

- Tests auto : `backend/tests/unit/test_transport_request_display.py`
- Checklist staging `[manuel]` : portail institution liste + détail demande

## Consommateurs débloqués après PASS

- `transport_request.serialize` blocs canoniques
- `InstitutionOffersTable` via `scheduling.summary`
- Exports institution via `scheduling.summary`
