# Nettoyage entreprises clones e2e (dev local)

Exécuté le 2026-08-02 sur la DB Docker locale.

## Action

Suppression de **319** entreprises clones / tests ; conservation de :

| id | name |
|----|------|
| 1 | Emmenez-moi |
| 2 | Roger |
| 3 | Diaz |
| 6 | Marques |

## API

`GET /admin/platform-billing/companies/config` ne renvoie plus que les entreprises **approuvées** par défaut (`include_unapproved=true` pour inclure le reste).

## Rejouer (dev uniquement)

Voir historique agent / script SQL temporaire — ne jamais exécuter en production sans revue.
