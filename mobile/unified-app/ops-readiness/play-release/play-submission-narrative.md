# Play Console — Narratif de soumission BG Location (LIRIE Opérations `ch.liri.operations`)

**Dernière mise à jour :** 2026-06-10
**Décision LOC-01 :** Cas A signé (2026-06-09) — présence flotte 07h–19h + mission active.
**Statut preuve FGS :** PASS (preuve système Android obtenue 2026-06-10, voir §4).

Ce document regroupe les textes à coller dans le formulaire Play Console et les références
de preuve associées. Source de vérité du texte court : `bg-location-justification.txt`.

---

## 1. Justification Background Location (Play Console — Sensitive permissions)

Texte à soumettre (identique à `bg-location-justification.txt`) :

> LIRIE collecte la position de l'appareil en arrière-plan uniquement pendant une mission de
> transport active assignée au chauffeur, afin de permettre le suivi opérationnel temps réel
> pour l'entreprise et le client. Une notification persistante indique que la localisation est
> active (« Mission en cours — localisation active »). L'application reste utilisable sans cette
> autorisation pour consulter les missions.
>
> Lorsque le chauffeur est déclaré disponible pendant la plage opérationnelle (typiquement
> 07h–19h), LIRIE peut également utiliser la localisation en arrière-plan afin de permettre à
> l'entreprise de visualiser les chauffeurs disponibles sur la flotte et d'attribuer les missions
> plus efficacement. Cette collecte est limitée à la gestion opérationnelle de la flotte ; une
> notification persistante indique lorsque la localisation est active (« Disponibilité active —
> localisation en cours »). Le chauffeur peut refuser cette autorisation : il reste utilisable
> pour consulter ses missions, mais n'est pas visible pour le dispatch flotte tant que la
> disclosure n'est pas acceptée.

**Pourquoi l'arrière-plan est indispensable :** le suivi opérationnel (dispatch, ETA client,
sécurité course) doit continuer lorsque le chauffeur conduit, écran éteint ou autre app au
premier plan. Un suivi au premier plan uniquement ne couvre pas le cas d'usage métier.

---

## 2. Description du disclosure affiché (prominent disclosure in-app)

Avant toute demande de permission de localisation arrière-plan, l'app affiche une modale de
disclosure dédiée (`PresenceAvailabilityDisclosureModal`, montée via `DriverPresenceDisclosureHost`) :

- **Titre :** « Disponibilité flotte »
- **Sous-titre :** « Localisation · gestion opérationnelle »
- **Corps :** explique que, lorsque le chauffeur est déclaré disponible pendant la plage
  opérationnelle, l'app peut utiliser la localisation en arrière-plan pour rendre le chauffeur
  visible au dispatch et attribuer les missions efficacement ; usage limité à la gestion de flotte.
- **Encart notification :** « Une notification persistante indique que la localisation est active
  (« Disponibilité active — localisation en cours »). »
- **Actions :** « Annuler » (refus) / « Continuer » (acceptation).

Le flux respecte l'ordre Play : **disclosure → consentement explicite → demande de permission
système → démarrage du Foreground Service**. Le refus est pleinement supporté : l'app reste
fonctionnelle, le chauffeur n'est simplement pas visible au dispatch.

Preuves : `captures-disclosure/disclosure-disponibilite-flotte.png`,
`captures-disclosure/PresenceAvailabilityDisclosureModal.png`.

---

## 3. Description de la fonctionnalité cœur (core functionality)

LIRIE Opérations est l'application des chauffeurs d'une entreprise de transport de personnes.
La localisation arrière-plan sert **deux fonctions opérationnelles** :

1. **Suivi de mission active** — pendant une course assignée (EN_ROUTE / IN_PROGRESS), la
   position alimente le suivi temps réel pour le dispatch et le client (ETA, sécurité).
   Notification : « Mission en cours — localisation active ».
2. **Disponibilité flotte (Cas A, 07h–19h)** — quand le chauffeur se déclare disponible dans la
   plage opérationnelle, sa position le rend visible sur la carte dispatch pour l'attribution.
   Notification : « Disponibilité active — localisation en cours ».

**Bornage strict :**
- Hors de la fenêtre 07h–19h **et** sans mission active → aucun suivi de présence, FGS non démarré.
- Pilotage par `driver_tracking_work_window_enabled` (fenêtre) + acceptation disclosure.
- Foreground Service typé **location** uniquement, avec notification persistante non-effaçable.

---

## 4. Références aux preuves (vidéo + captures + preuve système)

| Preuve | Fichier | Contenu |
| ------ | ------- | ------- |
| Vidéo démo BG location | `bg-location-demo.mp4` (26 s, 17,5 Mo) | Dashboard « Disponible » → passage arrière-plan, FGS persistant |
| Disclosure (modale) | `captures-disclosure/disclosure-disponibilite-flotte.png`, `PresenceAvailabilityDisclosureModal.png` | Modale « Disponibilité flotte » + boutons Annuler/Continuer |
| Notification persistante | `captures-disclosure/fgs-notification-presence.png` | « Lirie Unified est active — Disponibilité active — localisation en cours » (NO_CLEAR) |
| Preuve système Android | `../evidence/stop-gate-2/fgs-active-after-flag-override.txt` | `ServiceRecord LocationTaskService isForeground=true types=FGS_LOCATION`, stable > 1 h |
| Root cause + closure | `../evidence/stop-gate-2/CLOSURE.txt`, `fgs-diagnostic-rootcause.md` | Verdict PASS + cause racine documentée |

**Niveau de preuve :** au-delà de l'auto-report applicatif, la preuve `dumpsys activity services`
établit au niveau système l'existence d'un Foreground Service typé location avec notification
persistante — niveau attendu pour défendre l'usage devant Google.

---

## 5. Cohérence Cas A (triangulation)

```
Disponibilité flotte 07h–19h     -> driver_tracking_work_window_enabled + fenêtre trackingWindow.ts
        +
mission active                   -> pipeline mission (EN_ROUTE / IN_PROGRESS)
        +
notification persistante         -> FGS LocationTaskService, NO_CLEAR (PREUVE SYSTÈME)
        +
désactivation hors fenêtre       -> presence-outside-window-runtime.txt (aucun FGS hors 07h-19h sans mission)
```

| Élément | Code / app.json | Manifest AAB | Comportement prouvé |
| ------- | --------------- | ------------ | ------------------- |
| BG location | expo-location + disclosure | FGS location | Vidéo + dumpsys |
| FGS type location | withAndroidTrackingForegroundService | FOREGROUND_SERVICE_LOCATION | `types=0x00000008` |
| Notification | resolveForegroundServiceNotification | — | `fgs-notification-presence.png` |
| Refus disclosure | DriverPresenceDisclosureHost | — | `fgs-window-refus.txt` (pas de FGS) |
| Hors fenêtre | trackingWindow.ts | — | `presence-outside-window-runtime.txt` |

---

## 6. Réserve build (NON bloquant Play, à fermer avant retrait override)

La démonstration FGS a nécessité un override runtime backend
`MOBILE_FEATURE_FLAGS={"tracking_background_enabled":true}` car la variable
`EXPO_PUBLIC_ENABLE_BG_LOCATION` n'a pas été inlinée dans le build `production-apk`.
C'est un problème de chaîne de build EAS (ticket `BUILD-EXPO-001`), pas de conformité Play.
L'override est conservé en production comme mitigation jusqu'à ce qu'un build embarque
correctement le flag. Le comportement applicatif soumis à Play est exactement celui démontré.
