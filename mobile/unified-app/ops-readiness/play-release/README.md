# Livrables Play Console — Release obligatoires

| Artefact | Fichier / dossier | Statut |
| -------- | ----------------- | ------ |
| Narratif de soumission | `play-submission-narrative.md` | ✅ prêt |
| Vidéo BG location | `bg-location-demo.mp4` (26s, 17,5 Mo) | ✅ capturée (2026-06-10, S23) |
| Captures disclosure | `captures-disclosure/` | ✅ disclosure + notification persistante |
| Texte justification BG | `bg-location-justification.txt` | ✅ validé (Cas A) |
| Export Data Safety | `data-safety-mapping.md` | ⏳ à vérifier |
| URL suppression compte | `account-deletion-url.txt` | ✅ |
| URL privacy | `privacy-url.txt` | ✅ |
| Checklist cohérence | `checklist-coherence.md` | ⏳ revue finale |
| Export Play Console | `play-console-export/` | ⏳ captures formulaires après upload |

## Texte justification BG

Validé (décision LOC-01 Cas A). Voir `bg-location-justification.txt` (texte court à coller) et
`play-submission-narrative.md` (narratif complet : disclosure, fonctionnalité cœur, preuves).

## Preuve FGS (système)

FGS confirmé au niveau système Android le 2026-06-10 (STOP GATE #2 = PASS) :
`LocationTaskService isForeground=true`, type location, notification persistante NO_CLEAR.
Voir `../evidence/stop-gate-2/CLOSURE.txt`.

## Checklist cohérence (triangulation)

Comparer avant upload AAB :

1. Formulaire Play Console (captures `play-console-export/`)
2. Manifest AAB (`evidence/stop-gate-manifest/`)
3. Comportement app (vidéo + code)
