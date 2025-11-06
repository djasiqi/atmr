# Workshops & Cadrage – Application Mobile Enterprise Dispatch

## 1. Calendrier des ateliers (2 × 1h30 chacun)

| Atelier | Public cible | Objectifs | Date/heure proposés | Lieu / Support |
| --- | --- | --- | --- | --- |
| A1 – Dispatch terrain | Dispatchers (3-4), support planning | Recueil usages actuels, irritants, besoins mobilité | Mardi 12/11/2025 – 09:30-11:00 (Europe/Zurich) | Visioconférence (Teams) + Miro |
| A2 – Supervision & décision | Superviseurs (2), responsable opérationnel | KPIs, monitoring, modes auto, alertes | Mercredi 13/11/2025 – 14:00-15:30 | Salle Atlas / Teams hybride |
| A3 – Sécurité & conformité | Admin sécurité, DPO, IT MDM | Exigences SSO/MFA, RGPD, MDM, audit | Jeudi 14/11/2025 – 10:00-11:30 | Visioconférence (Teams) |

- Invitations à envoyer (Outlook) avec ordre du jour & artefacts pré-lecture.
- Prévoir prise de notes en direct (scribe) + enregistrement si accord participants.

## 2. Support d’interview – Dispatch terrain (US-01)

### Introduction (5 min)
- Contexte : nouvel outil mobile pour dispatchers.
- Objectif atelier : cartographier parcours, moments clés, douleurs.

### Questions / prompts
1. **Parcours actuels**  
   - “Décris ta journée type de dispatch.”  
   - “Quelles actions fais-tu le plus souvent en dehors du poste fixe ?”  
   - “Quelles infos te manquent en mobilité (ex : statut agent, OSRM, retards) ?”
2. **Gestion des modes**  
   - “Dans quelles situations passes-tu de manuel à semi-auto / fully-auto ?”  
   - “Quelles confirmations / validations sont indispensables avant bascule ?”
3. **Assignations & réassignations**  
   - “Comment choisis-tu un chauffeur quand plusieurs sont possibles ?”  
   - “Quels signaux te font reconsidérer une affectation existante ?”
4. **Urgences & incidents**  
   - “Quand utilises-tu un chauffeur d’urgence ?”  
   - “Quelles infos te faut-il pour décider rapidement ?”  
5. **Notifications & alertes**  
   - “Quelles alertes voudrais-tu recevoir sur mobile (contenu, urgence) ?”  
   - “Que ferais-tu directement depuis une notification ?”
6. **Expérience mobile actuelle**  
   - “As-tu déjà utilisé la driver app ou un autre outil mobile ?”  
   - “Qu’est-ce qui fonctionne / ne fonctionne pas sur petit écran ?”

### Clôture (10 min)
- Synthèse rapide, validation des principaux besoins.
- Prochaines étapes (maquettes, tests pilote).

## 3. Support d’interview – Supervision & décision (US-02)

### Thèmes
1. **Monitoring temps réel**  
   - KPIs clés (taux assignation, retards, usage urgence, fairness).  
   - Grains temporels (jour, 2h, tick par tick).
2. **Modes de dispatch**  
   - Règles de stabilité, qui décide des bascules.  
   - Besoin d’approbation / garde-fous.
3. **Alerting & notifications**  
   - Priorisation des alertes (impact client, service médical).  
   - Canal préféré (push, email, Slack).
4. **Reporting & audit**  
   - Infos nécessaires pour comité qualité / conformité.  
   - Exigences sur historique des actions (qui, quand, pourquoi).
5. **Vision mobile**  
   - Scénarios mobilité (astreinte, astreinte de nuit, weekend).  
   - Équipements (smartphone pro, tablette).

### Sorties attendues
- Tableau des KPIs prioritaires.
- Scénarios d’usage à couvrir dans la V1.
- Liste des alertes / actions critiques.

## 4. Support d’interview – Sécurité & conformité (US-03)

### Topics
1. **Authentification & SSO**  
   - OIDC/SAML disponibles, providers internes.  
   - MFA existant (TOTP, push), exigences minimum.
2. **Gestion des sessions**  
   - Durée, rotation refresh tokens, révocation (MDM).  
   - Contrainte “appareil partagé” vs personnel.
3. **MDM & politiques device**  
   - Solutions déployées (Intune, JAMF…).  
   - Contrôles requis (PIN, biométrie, jailbreak/root detection, wipe).
4. **RGPD & protection des données**  
   - Catégories PII autorisées sur mobile.  
   - Retention, droit à l’effacement, consentements.
5. **Audit & observabilité**  
   - Niveau de détail attendu (events, correlation IDs).  
   - Intégration SOC/SIEM.

### Checklist livrable
- Liste des exigences SSO/MFA signée.
- Matrice RGPD (base légale, finalité, données).
- Politique MDM appliquée (documentée).

## 5. Template de prise de notes

```
# Atelier A{N} – {Titre} – {Date} (Europe/Zurich)
Participants : …
Animateur : …

## Contextes & objectifs
- …

## Points saillants
- …

## Parcours / processus décrits
- …

## Douleurs & besoins
- …

## KPIs / alertes / sécurité
- …

## Actions & prochaines étapes
- …
```

- Stocker chaque compte-rendu dans `docs/workshops/` (un fichier par atelier).
- Ajouter résumés clés dans `docs/enterprise_dispatch_mobile_plan.md` (section S1-S2) après validation.

