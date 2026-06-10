# Data Safety — Mapping données ↔ code ↔ formulaire Play Console

**Dernière mise à jour :** 2026-06-10
**App :** `ch.liri.operations`
**Statut :** FIGÉ (4 points tranchés et vérifiés — voir §4). Reste 1 acte legal : confirmer Sentry comme sous-traitant.
**Transport :** HTTPS uniquement (`https://api.lirie.ch`) → chiffrement en transit = OUI.

## 1. Mapping technique (donnée ↔ source)

| Donnée déclarée | Collectée | SDK / source | Finalité |
| --------------- | --------- | ------------ | -------- |
| Position précise | Oui (mission / présence) | expo-location | Suivi opérationnel chauffeur |
| Identifiants compte | Oui | Auth JWT | Connexion multi-rôles |
| Adresse e-mail | Oui | Profil client/chauffeur | Compte, notifications |
| Téléphone | Oui | Profil | Contact course |
| Messages | Oui | Chat in-app | Coordination équipe |
| Diagnostics crash | Oui | Sentry | Stabilité app |
| Push token | Oui | Expo / FCM | Notifications métier |

## 2. Formulaire Play Console — Data Safety (prêt à saisir)

Légende : **C** = Collected, **S** = Shared (transféré à un tiers), **R/O** = Required/Optional,
**Éph.** = traité de façon éphémère (non stocké).

| Catégorie Play | Type de donnée | C | S | R/O | Éph. | Finalités Play |
| -------------- | -------------- | - | - | --- | ---- | -------------- |
| Location | Approximate location | Oui | Non | **R** | Non | App functionality |
| Location | Precise location | Oui | Non | **R** | Non | App functionality (suivi mission + disponibilité flotte) |
| Personal info | Name | Oui | Non | R | Non | Account management |
| Personal info | Email address | Oui | Non | R | Non | Account management, communications |
| Personal info | Phone number | Oui | Non | R | Non | App functionality (contact course) |
| Personal info | User IDs | Oui | Non | R | Non | Account management |
| Messages | In-app messages | Oui | Non | R | Non | App functionality (coordination dispatch) |
| App info & perf | Crash logs | Oui | **Non**¹ | O | Non | Crash prevention / diagnostics |
| App info & perf | Diagnostics | Oui | **Non**¹ | O | Non | Performance / diagnostics |
| Device or other IDs | Device/push token | Oui | **Non**² | R | Non | App functionality (notifications) |

> **Ligne App activity / Analytics : SUPPRIMÉE** — aucun SDK analytics utilisateur présent
> (vérifié 2026-06-10, voir §4). Les logs opérationnels / device-health sont de la télémétrie
> interne, pas des « Analytics » au sens Play.

**Location = Required** (décision figée) : la fonctionnalité cœur chauffeur (dispatch, suivi
mission, disponibilité flotte) repose sur la géolocalisation. Certaines vues restent consultables
sans GPS, mais l'app n'est pas utilisable normalement sans localisation → « Required ».

¹ **Sentry (crash/diagnostics) = Collected / Not Shared** (décision figée) : Sentry est utilisé
strictement comme sous-traitant technique (processor) traitant les données pour le compte de
LIRIE, sans revente ni partage avec des tiers indépendants. ⚠️ Validation juridique interne
finale recommandée avant soumission.

² **Push token = Collected / Not Shared** : transmis à Expo Push / FCM (Google) comme
infrastructure d'acheminement (service provider), pas un partage commercial/publicitaire.

## 3. Pratiques de sécurité (section Security practices)

| Question Play | Réponse | Preuve |
| ------------- | ------- | ------ |
| Données chiffrées en transit ? | **Oui** | `EXPO_PUBLIC_API_BASE_URL=https://api.lirie.ch` (HTTPS) |
| L'utilisateur peut demander la suppression des données ? | **Oui** | Client : suppression in-app `DELETE /clients/me`. Chauffeur : support `https://www.lirie.ch/contact` |
| Données supprimables sur demande ? | **Oui** | idem |
| Engagement Play Families Policy ? | N/A | App professionnelle, pas destinée aux enfants |

## 4. Points figés (vérifiés 2026-06-10)

- [x] **Sentry = Collected / Not Shared.** Sous-traitant technique. ⚠️ Validation juridique
      interne finale recommandée avant soumission (seul résidu à acter côté legal).
- [x] **Location = Required.** Fonctionnalité cœur (dispatch, mission, disponibilité flotte).
- [x] **Analytics = Non.** Vérifié dans `package.json` : aucun SDK analytics utilisateur
      (pas de Firebase Analytics, Amplitude, Mixpanel, PostHog, Segment, AppsFlyer, Adjust).
      Présents : `@sentry/react-native` (diagnostics), `@react-native-firebase/app` +
      `messaging` (FCM push uniquement, PAS analytics).
- [x] **Aucun nouveau SDK / permission depuis le gel (2026-06-09).** Vérifié via git :
      depuis le gel, `app.json` n'a changé que par des bumps `versionCode` (108→112) et la
      reformulation des `NSLocation*UsageDescription` iOS (texte Cas A, pas une nouvelle
      collecte). `package.json` / `package-lock.json` : aucune dépendance ajoutée/modifiée.
      → Aucune nouvelle collecte de données.

## 5. Cohérence

- Aligné avec la justification BG : `bg-location-justification.txt` + `play-submission-narrative.md`.
- Aligné avec les permissions manifest : `play-compliance.md` §Permissions.
- Suppression compte : `account-deletion-url.txt`, `privacy-url.txt`.
