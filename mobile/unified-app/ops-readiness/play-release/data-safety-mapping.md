# Data Safety — Mapping données ↔ code ↔ formulaire Play Console

**Dernière mise à jour :** 2026-06-10
**App :** `ch.liri.operations`
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
| Location | Approximate location | Oui | Non | R* | Non | App functionality |
| Location | Precise location | Oui | Non | R* | Non | App functionality (suivi mission + disponibilité flotte) |
| Personal info | Name | Oui | Non | R | Non | Account management |
| Personal info | Email address | Oui | Non | R | Non | Account management, communications |
| Personal info | Phone number | Oui | Non | R | Non | App functionality (contact course) |
| Personal info | User IDs | Oui | Non | R | Non | Account management |
| Messages | In-app messages | Oui | Non | R | Non | App functionality (coordination dispatch) |
| App activity | App interactions | Oui | Non | O | Non | Analytics / app functionality |
| App info & perf | Crash logs | Oui | Oui** | O | Non | Crash prevention / diagnostics |
| App info & perf | Diagnostics | Oui | Oui** | O | Non | Performance / diagnostics |
| Device or other IDs | Device/push token | Oui | Oui*** | R | Non | App functionality (notifications) |

\* **R\*** (Location) : requis pour la fonctionnalité de suivi/présence, mais l'app reste
utilisable pour consulter les missions sans accorder la localisation arrière-plan (le chauffeur
n'est alors pas visible au dispatch). À déclarer « Required » pour la feature, refus géré côté app.

\** **Sharing Sentry** : Sentry est un sous-traitant (service provider) pour les diagnostics
crash/perf. Play traite généralement les service providers à part, mais par prudence on peut
déclarer « Shared » si le transfert sort de l'organisation. **À confirmer juridiquement.**

\*** **Push token** : transmis à Expo Push / FCM (Google) pour l'acheminement des notifications.

## 3. Pratiques de sécurité (section Security practices)

| Question Play | Réponse | Preuve |
| ------------- | ------- | ------ |
| Données chiffrées en transit ? | **Oui** | `EXPO_PUBLIC_API_BASE_URL=https://api.lirie.ch` (HTTPS) |
| L'utilisateur peut demander la suppression des données ? | **Oui** | Client : suppression in-app `DELETE /clients/me`. Chauffeur : support `https://www.lirie.ch/contact` |
| Données supprimables sur demande ? | **Oui** | idem |
| Engagement Play Families Policy ? | N/A | App professionnelle, pas destinée aux enfants |

## 4. Points à confirmer avant saisie (human-in-the-loop)

- [ ] **Sentry = « Shared » ou non ?** Trancher selon la politique (service provider vs tiers).
- [ ] **Location Required vs Optional** dans le formulaire (refus supporté côté app → cohérent
      avec « Required pour la feature, app utilisable sans »).
- [ ] **App interactions / analytics** : confirmer s'il y a une collecte analytics réelle (sinon
      retirer la ligne App activity).
- [ ] Vérifier qu'aucun SDK ajouté depuis le gel permissions (2026-06-09) n'introduit de nouvelle
      collecte (voir `play-compliance.md` — gel SDK).

## 5. Cohérence

- Aligné avec la justification BG : `bg-location-justification.txt` + `play-submission-narrative.md`.
- Aligné avec les permissions manifest : `play-compliance.md` §Permissions.
- Suppression compte : `account-deletion-url.txt`, `privacy-url.txt`.
