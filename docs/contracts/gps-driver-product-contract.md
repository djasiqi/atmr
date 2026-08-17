# Contrat produit GPS chauffeurs — ATMR

**Statut :** Source of Truth produit pour le suivi GPS chauffeurs.  
**Référence d’implémentation :** plan conformité FINAL v4 (`OFF` / `BLOCKED` / `PRESENCE` / `LIVE`).

Ce document doit servir de référence aux développements, aux tests canary et aux critères de validation du suivi GPS ATMR.

---

## 1. Objectif

Le suivi GPS d’ATMR doit permettre de connaître la position d’un chauffeur **lorsqu’il est en service**, qu’il soit momentanément disponible ou engagé sur une mission.

Le fonctionnement doit rester simple pour le chauffeur : il ne doit pas avoir à garder l’application ouverte, redémarrer manuellement le GPS ou intervenir après une perte de réseau.

Le principe général est :

**le statut de travail du chauffeur détermine si sa position doit être suivie ; la mission détermine le niveau de suivi nécessaire.**

---

## 2. Les trois états GPS du chauffeur

### État 1 — Hors service

Le chauffeur est considéré hors service lorsqu’il :

* n’est pas connecté (session authentifiée absente) ;
* s’est explicitement déconnecté ;
* a terminé sa journée et n’est plus disponible (`Driver.is_available=false`) ;
* n’est plus dans son espace chauffeur ;
* ou a effectué un véritable arrêt forcé de l’application depuis Android.

### Comportement attendu

**GPS ATMR arrêté.**

ATMR :

* ne demande plus de nouvelles positions ;
* n’envoie plus de positions au serveur ;
* ne maintient plus le suivi en arrière-plan ;
* ne présente plus le chauffeur comme disponible en temps réel.

Le chauffeur ne doit pas continuer à être géolocalisé inutilement lorsqu’il n’est plus en service.

---

## 3. Chauffeur en service mais sans mission active

Un chauffeur peut être :

* connecté (session authentifiée) ;
* disponible pour travailler (`Driver.is_available=true`) ;
* en attente d’une prochaine course ;
* entre deux missions ;

sans avoir de mission active à cet instant.

Dans cette situation, **le GPS ne doit pas être complètement coupé**.

Il passe en **mode présence**.

### Objectif du mode présence

Permettre au dispatch de savoir approximativement où se trouve un chauffeur disponible afin de :

* connaître les chauffeurs présents sur le terrain ;
* faciliter l’attribution d’une prochaine course ;
* voir leur zone géographique ;
* éviter de considérer à tort un chauffeur comme hors ligne.

Le mode présence n’a cependant pas besoin de la même fréquence qu’un transport en cours.

### Comportement attendu

```text
CHAUFFEUR CONNECTÉ + EN SERVICE
+ PAS DE MISSION ACTIVE
+ permissionsReady (FG + BG)

→ GPS PRÉSENCE ACTIF
→ chauffeur visible sur la carte
→ position actualisée régulièrement
→ consommation batterie raisonnable
```

> **Note produit :** la fenêtre horaire 07–19 Europe/Zurich **n’est plus** une gate produit. C’est l’état de service (`Driver.is_available`) qui commande la présence GPS, pas l’heure de la journée.

Une position peut par exemple être actualisée à une fréquence modérée ou lorsqu’un déplacement significatif est constaté — sans que la distance seule soit la preuve de vie GPS (voir §9 et B4).

---

## 4. Démarrage d’une mission

Lorsqu’une mission devient active et nécessite le suivi du chauffeur, ATMR doit automatiquement passer du **mode présence** au **mode suivi en direct**.

Le chauffeur ne doit avoir aucune manipulation particulière à effectuer.

```text
GPS présence
→ mission démarre
→ GPS mission en direct
```

Il faut éviter autant que possible de couper puis de redémarrer complètement le GPS lors de cette transition.

Le passage doit être fluide (aucun `stopLocationUpdatesAsync` / unregister / trou de capture / rotation de session).

---

## 5. Chauffeur en route ou mission en cours

Lorsqu’un chauffeur :

* part vers une prise en charge ;
* est indiqué « En route » ;
* réalise effectivement le transport ;
* ou se trouve dans tout autre état considéré comme mission active ;

ATMR passe en **suivi GPS en direct**.

### Comportement attendu

```text
MISSION ACTIVE
→ GPS LIVE
→ nouvelles positions régulières
→ position serveur continuellement actualisée
→ chauffeur visible en direct sur la carte
```

---

## 6. Application visible à l’écran

```text
mission active
→ GPS LIVE

chauffeur disponible sans mission
→ GPS PRÉSENCE
```

Le simple fait d’ouvrir ou fermer un écran de l’application ne doit pas créer de nouvelle session GPS ni réinitialiser le suivi.

---

## 7. Application mise en arrière-plan avec HOME

```text
MISSION ACTIVE + HOME
→ GPS LIVE CONTINUE

DISPONIBLE + HOME
→ GPS PRÉSENCE CONTINUE
```

La position du chauffeur ne doit pas disparaître de la carte simplement parce qu’ATMR n’est plus l’application affichée à l’écran.

Cela exige `permissionsReady` incluant la permission arrière-plan.

---

## 8. Écran du téléphone verrouillé

```text
écran verrouillé + mission active
→ GPS LIVE continue

écran verrouillé + disponible
→ GPS présence continue
```

---

## 9. Chauffeur immobile

Un chauffeur qui ne se déplace pas ne doit jamais être considéré automatiquement comme ayant perdu son GPS.

ATMR doit distinguer :

**« la voiture ne bouge pas »** de **« nous n’avons réellement plus aucune information GPS »**.

Deux heartbeats distincts :

1. **Location heartbeat** — vrai fix / nouvelle capture / nouvel `event_id` / peut avancer `recorded_at`.
2. **Device-health heartbeat** — vie du runtime ; **ne mute jamais** `recorded_at`.

---

## 10. Perte temporaire d’Internet

```text
GPS disponible
Internet indisponible

→ continuer à produire les positions utiles
→ conserver localement ce qui ne peut pas être envoyé
```

Lorsque Internet revient : transmettre les positions en attente ; reprendre immédiatement. Aucune position déjà créée ne doit changer après coup.

`networkConnected=false` / `socketConnected=false` **≠** hors service.

---

## 11. Une position créée est immuable

Chaque position représente un événement unique. Retry = renvoyer exactement la même position. Interdit de réutiliser un `event_id` pour de nouvelles coordonnées.

---

## 12. Reconnexion Internet ou temps réel

```text
connexion perdue
→ session GPS X

connexion revient
→ session GPS X continue
```

Pas de rotation gratuite de session à chaque reconnexion.

---

## 13. Une seule session active

À un instant donné, un chauffeur doit disposer d’**une seule session GPS active** correspondant à son activité courante.

---

## 14. Perte momentanée d’information dans l’application

```text
information momentanément absente
→ vérifier
→ laisser un court délai de confirmation
→ récupérer la mission
→ continuer le GPS sans coupure
```

---

## 15. Récupération automatique

```text
quelque chose paraît incohérent
→ vérifier
→ essayer de rétablir (L1 non destructif)
→ conserver le suivi existant autant que possible
```

Éviter les boucles cut/restart.

---

## 16. Fin d’une mission

```text
MISSION LIVE
→ mission terminée
→ GPS PRÉSENCE  (si encore en service)
```

Une fin de mission n’est **jamais** à elle seule une raison de STOP natif.

---

## 17. Nouvelle mission après une mission terminée

```text
GPS PRÉSENCE
→ GPS LIVE
```

Transition fluide, sans fermer/rouvrir l’application.

---

## 18. Fin de journée / chauffeur hors service

```text
GPS LIVE ou PRÉSENCE
→ GPS OFF
```

`Driver.is_available=false` → statut métier **Hors service**, pas « GPS hors ligne ».

---

## 19. Déconnexion

```text
LOGOUT
→ GPS OFF
```

Empêcher un ancien état local de continuer à envoyer des positions sous l’identité du chauffeur.

---

## 20. Force-stop Android

```text
FORCE-STOP
→ GPS ATMR arrêté
→ aucun nouvel envoi
→ aucun nouveau suivi
→ aucune auto-reprise tant que l’utilisateur n’a pas relancé l’application
```

Hors FSM produit JS (le runtime JS ne s’exécute plus).

---

## 21. Swipe depuis les applications récentes

Idéalement, si le chauffeur est encore en service, le suivi continue. Comportement à valider définitivement sur les appareils cibles avant garantie produit.

---

## 22. Position « envoyée » et position « confirmée »

Libellés produit :

```text
GPS actif · Position à jour
GPS actif · Synchronisation…
GPS indisponible
EN SERVICE · GPS BLOQUÉ — AUTORISATION REQUISE
Localisation en cours…
```

Ne pas présenter un ACK pending comme panne GPS.

---

## 23–24. Carte du dispatch et fraîcheur

La carte affiche selon la situation réelle. Fraîcheur = `recorded_at` de la dernière vraie position.

Cinq situations :

1. **Hors service** — jamais « GPS hors ligne »
2. **Acquisition / BLOCKED** — « Localisation en cours… » / autorisation
3. **Live / Recent** — suivi normal
4. **Stale / Last known** — « Dernière position : il y a X » ; `visualStatus ≠ offline`
5. **GPS hors ligne** — uniquement chauffeur censé être tracké + seuil fort + preuve **indépendante** de panne pipeline (pas `last_fix_age` seul)

---

## 25. Règles finales

```text
CHAUFFEUR HORS SERVICE
→ GPS OFF

CHAUFFEUR CONNECTÉ + EN SERVICE + permissionsReady
SANS MISSION
→ GPS PRÉSENCE

MISSION À SUIVRE
→ GPS LIVE

HOME / LOCK / FG
→ conserver le mode actuel

CHAUFFEUR IMMOBILE
→ GPS reste considéré vivant (location + device-health heartbeats)

PERTE INTERNET / RECONNEXION
→ même session ; flush ; pas OFF métier

FIN DE MISSION + encore en service
→ GPS PRÉSENCE

FIN DE SERVICE / LOGOUT
→ GPS OFF

FORCE-STOP ANDROID
→ GPS OFF (OS) ; pas d’auto-reprise JS
```

---

## 26. Principe directeur

> **Tant qu’un chauffeur est en service, ATMR maintient sa présence GPS. Lorsqu’il réalise une mission, cette présence devient un suivi en direct. Les changements d’écran, pertes de réseau, reconnexions et anomalies temporaires ne doivent jamais suffire à interrompre ou recréer inutilement le suivi. Le GPS n’est arrêté que lorsqu’une raison explicite et durable le justifie : fin de service, déconnexion, sortie du contexte chauffeur, révocation d’identité, ou arrêt forcé de l’application.**

### Machine d’état produit

```text
OFF ↔ BLOCKED ↔ PRESENCE ↔ LIVE
```

- `BLOCKED` : en service mais contrat non garanti (`capabilityReady` = permissions FG+BG **et** disclosure) — ≠ hors service. Une mission **ne contourne pas** `BLOCKED`.
- Fin de mission seule → `PRESENCE` si encore en service.
- Force-stop : hors FSM JS.
- Disponibilité mobile : `UNKNOWN` tant que `Driver.is_available` n’est pas hydraté (cache/DB) — pas PRESENCE/LIVE, pas hors service.
- Preuve de vie carte : heartbeat device-health ≤ 120 s. `last_fix_age` n’est pas une preuve indépendante. `is_available=false` → HORS SERVICE, jamais « GPS hors ligne ».

---

## Implémentation (v4)

✅ **Implémenté** sur `feat/gps-product-contract-v4` :

- SoT `Driver.is_available` → fanout / live-locations / mobile
- Mobile : état `UNKNOWN | AVAILABLE | UNAVAILABLE` (pas de défaut `true` avant hydratation)
- FSM `BLOCKED` + `capabilityReady` (permissions FG+BG **et** disclosure) ; mission ne contourne pas `BLOCKED`
- Retrait gate présence 07–19 (fenêtre mission T−30 intacte)
- Transition présence↔LIVE sans stop FGS (B2) + options natives durables (B3)
- Labels chauffeur + 5 situations carte (`off_duty` prioritaire sur `gps_offline` ; heartbeat frais obligatoire)
- Matrice canary : [`docs/ops/gps-product-contract-canary-matrix.md`](../ops/gps-product-contract-canary-matrix.md)

**Gate release :** A1 mobile + A3 + C2 doivent être verts avant canary. B2+B3+canary batterie requis avant candidat prod.
