# Gate — Session permanente chauffeur + GPS

Document de gate versionné (F3). Critères GO avant pilote / rollout.

## Base SHA

Parent initial : `eeee2d613e51c30236d8a8f32749653b89947484`

Chaque PR F1a / F1b / F2 / F3 déclare son SHA parent.

## Niveaux GO

### GO code

- [ ] F1a mergé (logout preuves, revoke-pending idempotent, security revoke, GPS fail-closed)
- [ ] F1b mergé (unicité partielle, générations, TX SQL, guards, routes UUID)
- [ ] F2 mergé (coordinateur branché, logout crash-safe, PendingRefreshOperation)
- [ ] F3 mergé (`expo-background-task`, `processing` iOS, CI)
- [ ] Migration Alembic appliquée via Docker
- [ ] Builds Android / iOS OK (EAS — OTA insuffisant pour native)

### GO pilote chauffeur (matrice physique exécutée)

Appareils : ≥1 Android réel + ≥1 iPhone réel.

| Scénario | Attente |
| --- | --- |
| OS suspend l'app | session + queue GPS préservées ; reprise au foreground |
| OS tue pour mémoire | idem après réouverture |
| Force-quit / force-stop utilisateur | **pas** d'exigence GPS auto en arrière-plan |
| Après nouvelle ouverture (force-stop inclus) | profil sans login si session OK ; queue intacte ; auth auto ; tracking si mission active + permissions |
| Reboot | session restaurée |
| Mode avion | offline authenticated ; tombstone logout si déconnexion |
| Nuit écran verrouillé | session intacte ; GPS mission via FGS/location native |
| Access expiré | refresh / session-resume sans login |
| Logout offline | tombstone → ACK au retour réseau |
| Révocation 2e appareil | rejet immédiat (SLO ≤ 5 s) |
| BackgroundTask | vérifié sur device (pas simulateur iOS) |

### GO rollout large

- [ ] Pilote stable
- [ ] Multi-OEM Android
- [ ] Pas de pic `storage_locked` / `rotation_recovery_required` / `refresh_replay_detected`

## Non-exigences (plateforme)

- GPS toujours actif après force-stop utilisateur : **non garanti**
- BackgroundTask toutes les 15 min exactes : **best-effort OS**

## Vérification git du présent document

```bash
git check-ignore -v docs/gate-session-permanente-gps.md
git ls-files docs/gate-session-permanente-gps.md
```

Le second doit lister ce fichier après commit.
