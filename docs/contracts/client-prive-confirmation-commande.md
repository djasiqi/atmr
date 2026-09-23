# Cahier des charges — Compte client privé : cadre contractuel et confirmation de commande

**Statut** : source de vérité **cible**, **non implémentée**. L’impayé ne démarre qu’après la bascule (étape 5), elle-même seulement quand les étapes 1 à 4 sont implémentées et testées.

**Périmètre** : compte **client privé** (rôle CLIENT, hors institution, hors réservation invité). Le portail institution et le paiement guest ne sont pas couverts ici.

**Deux documents, deux rôles, jusqu’à la migration complète.**

| Document | Rôle |
| --- | --- |
| Ce fichier | Cible produit |
| [AUTH-SMS-02](../ops/auth-sms-02-decouple-activation.md) | Comportement de production **actuel** |

Tant que les étapes 1 à 4 ne sont pas complètement implémentées et testées, AUTH-SMS-02 et `assert_portal_can_confirm_transport` (`backend/services/auth/portal_phone_verification.py`) restent en production. L’OTP à la confirmation de transport n’est retiré qu’à l’étape 5.

## Règle d’or

**Aucune preuve contractuelle n’est mutable.**

Acceptation des CGV, confirmation de commande, modification et annulation produisent toujours de nouveaux événements. Elles ne réécrivent jamais l’historique. Cette règle s’applique à toute la suite, y compris l’impayé.

Sources de cadrage retenues pour la conception (pas un avis juridique) :

- Code des obligations, principe de liberté de forme du contrat ordinaire, sauf disposition spéciale — [Fedlex, CO](https://www.fedlex.admin.ch/eli/cc/27/317_321_377/fr).
- SECO, commerce électronique : le client doit savoir quelle action envoie la commande ; l’acceptation des CGV peut résulter d’un clic de confirmation ; confirmation électronique immédiate, notamment par e-mail — [Commerce électronique](https://www.seco.admin.ch/fr/commerce-electronique), [Avant l’achat et conclusion du contrat](https://www.seco.admin.ch/fr/avant-l-achat-et-conclusion-du-contrat).

## Principe

Un OTP n’est pas le moyen de conclure chaque réservation. Il sert à **vérifier le compte** et à **renforcer** quelques situations sensibles.

Deux couches :

| Couche | Moment | Action du client |
| --- | --- | --- |
| Contrat-cadre | Création / activation du compte | E-mail vérifié, téléphone vérifié par OTP, acceptation explicite des CGU et des CGV |
| Commande individuelle | Chaque réservation | Un clic sur **Confirmer la réservation – CHF X.–** |

Pas de case CGV à recocher à chaque course. Pas de code SMS à chaque course.

## 1. Activation du compte

À la première création, pour un client privé :

- identité : civilité, prénom, nom ;
- adresse ;
- e-mail vérifié ;
- téléphone vérifié par OTP ;
- date de naissance si le produit la demande.

Puis une acceptation explicite, une seule fois pour la version en vigueur :

> J’accepte les Conditions générales d’utilisation et les Conditions générales de réservation et de transport de LIRIE.

Le compte n’est **contractuellement activé** qu’après cette acceptation. On conserve un enregistrement, pas un booléen.

Une nouvelle acceptation **ajoute une ligne**. Elle ne modifie jamais la ligne précédente. Même règle pour une commande modifiée ou annulée : nouvel événement, jamais de réécriture de la preuve historique.

```text
user_id
terms_version          ex. 1.3
accepted_at            horodatage avec fuseau
email                  adresse vérifiée au moment de l’acceptation
verified_phone         E.164 vérifié par OTP
verification_method    ex. otp_sms
cgv_hash               empreinte du texte exact accepté
ip / session           seulement selon la politique de conservation
```

Les CGU et les CGV doivent être lisibles **avant** le clic d’acceptation. La version applicable est celle dont le hash a été enregistré.

✅ **Implémenté** : registre append-only `legal_document_version` et `client_terms_acceptance` (`backend/models/client_terms_acceptance.py`). CGU (`terms_of_service`) et CGV de transport (`transport_terms`) sont deux documents. Le corps figé est dans `backend/legal/canonical/fr/`, pas dans la page frontend. `POST/GET /api/v1/clients/me/terms-acceptances` insère et lit seulement. Les comptes existants ne sont pas backfillés. Le blocage `TERMS_REACCEPTANCE_REQUIRED` reste à l’étape 4. La page `frontend/src/pages/Legal/TermsOfService.jsx` n’est pas la source canonique.

✅ **Implémenté** : une nouvelle inscription PORTAL (`activation_session.portal_terms_required`) suit `inscription → e-mail confirmé → lecture des deux textes canoniques → case explicite non précochée → finalisation`. `POST /api/v1/auth/activation/finalize` avec `accept_current_portal_terms: true` écrit les deux `ClientTermsAcceptance` et active le compte dans la même transaction. Le navigateur ne choisit ni version ni empreinte. `GET /api/v1/auth/activation/portal-terms` sert les mêmes corps que le catalogue. Sans téléphone vérifié, `verification_method = not_verified`. Un OTP ultérieur ne réécrit pas ces lignes. Les sessions antérieures (`portal_terms_required = false`) restent activables sans acceptation, et AUTH-SMS-02 n’exige toujours pas le SMS pour activer le compte.

## 2. Chaque réservation

Récapitulatif visible avant le bouton :

- trajet (prise en charge, destination, date et heure) ;
- prix, par exemple CHF 90.– ;
- personne facturée (le client privé) ;
- mode de paiement affiché (par exemple à 15 jours).

Au-dessus du bouton, une phrase, avec lien vers les conditions :

> En confirmant cette réservation, vous passez une commande soumise aux Conditions générales de réservation et de transport acceptées lors de la création de votre compte.

Bouton unique, libellé exact comprenant le montant :

> Confirmer la réservation – CHF 90.–

Ce clic est l’acte de commande. Il produit un **enregistrement probatoire immuable** lié à la version exacte des CGV déjà acceptées. On n’y met pas une nouvelle case à cocher, ni un OTP.

Champs minimum de cet enregistrement :

```text
user_id authentifié
booking_id
confirmed_at
libellé exact du bouton
prix accepté et devise
débiteur affiché
trajet et horaire affichés
terms_version
cgv_hash
```

Aucune nouvelle réservation si le compte n’a pas d’acceptation valide pour la version substantielle en vigueur (`TERMS_REACCEPTANCE_REQUIRED`).

✅ **Implémenté** : événement append-only `client_booking_contract_event` à la création PORTAL, dans la même transaction que le booking (`BOOKING_CREATED` seulement). Le montant figé est `estimated_amount_snapshot` avec `amount_is_contractual = false`. Sans acceptation, les clés vers `ClientTermsAcceptance` restent vides. Le bouton et `TERMS_REACCEPTANCE_REQUIRED` ne sont pas faits ici.

✅ **Implémenté** : pour un nouveau `BOOKING_CREATED` PORTAL, le débiteur nominal est le titulaire authentifié (`debtor_type_snapshot = account_holder`, `debtor_user_id`, nom, e-mail et téléphone du `User`, et `Client.billing_address` seulement si cette colonne est renseignée). `billed_to_type = patient` reste une catégorie legacy et n’est pas cette identité. Le domicile, `contact_email` et le nom saisi dans la demande ne sont pas recopiés comme débiteur. Les événements déjà `partial` ne sont pas réécrits. Le navigateur ne peut pas soumettre `debtor_*`.

✅ **Implémenté** : le parcours PORTAL affiche un récapitulatif puis le bouton « Confirmer la demande de transport », sans montant dans le bouton. L’estimation reste indicative. Les liens affichent le catalogue serveur `GET /clients/me/portal-terms`, pas `TermsOfService.jsx`. Sans acceptation enregistrée, le texte ne prétend pas que les conditions ont été acceptées. L’activation ne crée toujours pas de `ClientTermsAcceptance`.

## 3. E-mail : confirmation, pas acceptation

✅ **Implémenté** : après le commit de la commande PORTAL, un e-mail de confirmation est tenté. Son échec n’annule pas le booking ni `BOOKING_CREATED`. Chaque tentative est tracée dans `portal_booking_confirmation_email`.

Trois rôles distincts :

1. **Acceptation** — dans LIRIE, par l’utilisateur connecté (cadre à l’activation, commande au clic).
2. **Preuve** — l’enregistrement écrit par LIRIE au moment du clic.
3. **Confirmation** — e-mail envoyé immédiatement après la commande.

L’e-mail n’est pas le mécanisme d’acceptation. LIRIE conserve une copie de ce qui a été envoyé (corps ou empreinte, horodatage, identifiant de réservation).

Exemple de contenu :

> Réservation confirmée — LIRIE #39869
>
> Bonjour Monsieur Dupont,
>
> Votre réservation a été confirmée.
>
> Date : 29.09.2026
> Trajet : Presinge → HUG
> Prix : CHF 90.–
> Facturé à : Monsieur X Dupont
> Conditions applicables : CGV version 1.3
>
> Vous pouvez retrouver votre réservation et les conditions applicables depuis votre compte LIRIE.

## 4. Changement des CGV

| Nature du changement | Version | Effet |
| --- | --- | --- |
| Correction sans conséquence contractuelle importante | 1.3 → 1.3.1 | Information éventuelle. Pas de blocage. |
| Délai de paiement, annulation, no-show, frais, intérêts, responsabilité, suspension pour impayés, règles importantes de facturation | version substantielle | `TERMS_REACCEPTANCE_REQUIRED` |

À la prochaine connexion, si réacceptation exigée :

> Mise à jour de nos conditions
>
> Nos Conditions générales ont été mises à jour.
>
> ☐ J’ai pris connaissance et j’accepte les nouvelles Conditions générales.
>
> Accepter et continuer

Tant que cette acceptation n’est pas enregistrée (nouvelle ligne, nouveau hash, nouvel horodatage), **aucune nouvelle réservation**.

## 5. OTP : sécurité, pas rituel de commande

| Situation | OTP |
| --- | --- |
| Création du compte (vérification du téléphone) | Obligatoire |
| Changement de numéro | Obligatoire |
| Changement d’e-mail sensible | Vérification |
| Récupération du compte | Obligatoire |
| Nouveau terminal ou comportement inhabituel | Possible |
| Montant exceptionnel | Possible |
| Réservation habituelle depuis un compte déjà vérifié | **Aucun** |

La première réservation peut demander un OTP si la politique de risque le décide. Ce n’est pas le chemin par défaut.

`phone_verified_at` reste la trace qu’un OTP a réellement vérifié le numéro. On ne le fabrique pas.

## 6. Dossier de preuve (cible, pas l’écran impayés)

Pour une course facturée et non payée, le dossier doit pouvoir se relire ainsi, à partir des enregistrements ci-dessus :

```text
CLIENT
identité, date de création du compte
e-mail vérifié : oui
téléphone vérifié par OTP : oui

CADRE
CGV version
acceptées le : horodatage
hash du texte

COMMANDE
identifiant de réservation
date et heure du clic
trajet
prix accepté
débiteur affiché
libellé exact du bouton

CONFIRMATION
e-mail envoyé : horodatage
copie ou empreinte du message

PRESTATION
course effectuée : oui / non

FACTURATION
numéro, montant, échéance

PAIEMENT
reçu ou non
```

## 7. Ordre de développement

Chaque étape attend que la précédente soit en place. L’étape 5 est la seule qui retire l’OTP systématique de la réservation.

1. **Registre d’acceptation des CGV** — version, hash du contenu, `accepted_at`, utilisateur, e-mail vérifié, téléphone vérifié, méthode de vérification. Ajout seul : une nouvelle acceptation n’altère pas l’ancienne ligne.
2. **Événement immuable de confirmation de commande** — `transport_id`, `user_id`, montant affiché, devise, texte du bouton, version CGV applicable, horodatage, snapshot des éléments contractuels importants. Modification ou annulation = nouvel événement.
3. **Parcours client** — bouton **Confirmer la réservation – CHF X.–** ; le clic conclut la commande ; l’e-mail qui suit est une confirmation, pas l’acceptation.
4. **Réacceptation** — changement substantiel → `TERMS_REACCEPTANCE_REQUIRED` ; aucune nouvelle réservation tant que la nouvelle ligne d’acceptation n’existe pas ; les acceptations antérieures restent.
5. **Migration AUTH-SMS-02** — retirer l’OTP systématique à la réservation seulement quand les étapes 1 à 4 sont couvertes par les tests. Conserver l’OTP pour la création de compte, le changement de numéro, la récupération et les événements sensibles.
6. **Impayés** — facture échue, blocage des nouvelles réservations, historique probatoire exploitable, rappel et recouvrement.

## 8. Hors de ce document

- Institutions et transporteurs.
- Réservation invité / paiement immédiat guest.
- Rédaction juridique du texte des CGU et des CGV (le produit enregistre la version et le hash ; le texte vient du juridique).
- Implémentation. Le schéma, les écrans et l’e-mail décrits ici ne sont pas en production.

## Reste à faire

Les six étapes de la section 7. Aucune n’est implémentée. AUTH-SMS-02 reste le comportement de production.
