# 7A — Audit de réalité contractuelle (client privé PORTAL)

```text
7A = CURRENT CONTRACT REALITY AUDIT
Périmètre : client privé PORTAL uniquement
Statut : lecture produit / code — pas une rédaction juridique
TARGET LOCKED = double validation client + plafond distinct
```

**Hors scope 7A** : modification du code, mutation des canoniques 1.0,
choix de barèmes d’annulation, implémentation 7B, rédaction CGU/CGV 2.0 (7C).

Documents liés :

- [client-prive-confirmation-commande.md](client-prive-confirmation-commande.md)
- [portal-receivable.md](portal-receivable.md)
- [portal-payment-hold-policy.md](portal-payment-hold-policy.md)
- Canoniques 1.0 (lecture seule) :
  `backend/legal/canonical/fr/terms_of_service_v1.0.txt`,
  `transport_terms_v1.0.txt`

---

## 1. Cible produit verrouillée (référence d’écart)

```text
1er clic client
→ demande marché
→ estimation LIRIE éventuelle
→ plafond maximum explicitement accepté (donnée DISTINCTE)
→ PAS de contrat de transport
→ company_id = NULL

Transporteur X
→ propose son prix + conditions annulation / no-show / attente
→ prix_X <= plafond  → offre recevable → présentée pour 2e clic
→ prix_X >  plafond  → offre INÉLIGIBLE
                      → PAS de présentation confirmable
                      → PAS de contrat
                      → PAS de dépassement silencieux
                      → PAS de sollicitation pour relever le plafond

2e clic client
→ voit X + prix X + conditions X (+ rappel plafond)
→ « Confirmer le transport à CHF … »
→ CLIENT_TRANSPORT_CONFIRMED
→ contrat formé
→ prix contractuel = prix X (jamais le plafond, jamais l’estimation)

Après exécution
→ X facture directement → X créancier → paiement client → X
```

### Rôles

| Acteur | Rôle cible | Aujourd’hui (code) |
| --- | --- | --- |
| LIRIE | Infra : transmet, affiche, fige preuves, suit facture/contestation. Ne fixe pas le prix, ne facture pas, n’encaisse pas, n’est pas créancier. | Aligné en partie (pas d’encaissement PORTAL ; créance = `PortalReceivable` côté company). Manque présentation prix/annulation X + 2e clic. |
| Titulaire compte | Demande + plafond + confirmation du transport | Demande oui ; plafond non ; 2e clic non. |
| Passager | Éventuel, pas débiteur | Débiteur = titulaire (`BOOKING_CREATED`). |
| Entreprise X | Prix, annulation, exécution, facture, créancier | Accepte et s’assigne ; ne propose pas un prix distinct pour le client ; facture via `PortalReceivable` après course. |

Cohérent avec le positionnement public (exécution par entreprises partenaires,
LIRIE plateforme) — ex. pages Professionnel / Conduire.

### Montants — vocabulaire strict

| Élément | Signification | Champ cible |
| --- | --- | --- |
| Estimation LIRIE | Indicative | `estimated_amount_snapshot` |
| Prix maximum client | Plafond du 1er clic | `maximum_accepted_amount_snapshot` (**distinct**) |
| Prix proposé par X | Offre transporteur | sur `CARRIER_OFFERED` |
| Prix contractuel | Confirmé au 2e clic | sur `CLIENT_TRANSPORT_CONFIRMED` |
| Facture | Après prestation | `PortalReceivable.total_amount` |

Le plafond n’est **jamais** une dette. L’estimation non plus.

### Preuve cible (7B)

```text
BOOKING_CREATED
→ estimated_amount_snapshot
→ maximum_accepted_amount_snapshot
→ currency, company_id = NULL, contract_formed = false
→ amount_is_contractual = false  (attaché à l’estimation)

CARRIER_OFFERED
→ company + price_x + cancellation_policy_{id,version,snapshot,hash}
→ admissible ssi price_x <= maximum_accepted

CLIENT_TRANSPORT_CONFIRMED
→ price_contractual = price_x
→ snapshot/hash annulation présentés
→ = formation définitive du contrat
```

Ne **jamais** réécrire `BOOKING_CREATED` pour y coller le prix ferme.

---

## 2. État actuel — 1er clic (demande marché)

### Ce que fait le code

| Fait | Preuve |
| --- | --- |
| Bouton UI : « Confirmer la demande de transport » (pas de montant dans le libellé) | `ClientDashboard.jsx` |
| `company_id = None` à la création PORTAL | `resolve_booking_owner_company_id_for_create` → `PORTAL → None` |
| Événement `BOOKING_CREATED` append-only | `ClientBookingContractEvent` ; contrainte unique par booking |
| Montant figé = `estimated_amount_snapshot` depuis `booking.amount` | `record_booking_contract_event.py` |
| `amount_is_contractual = false` imposé (CHECK SQL) | modèle + migration |
| `pricing_status = 'estimated'` seul autorisé | CHECK sur la table |
| `carrier_status = not_assigned` si pas de company | même service |
| Estimation UI via `POST /clients/me/indicative-fare/estimate` ou config plateforme / `PORTAL_CLIENT_PREVIEW_COMPANY_ID` (preview tarifaire seulement) | `clients.py`, `portal-payment-hold-policy.md` |
| E-mail post-création : « demande enregistrée », estimation « indicative, non contractuelle », « Transporteur : attribué après confirmation » | `send_portal_booking_confirmation.py` |

### Écarts vs cible

| Cible | Actuel | Gravité |
| --- | --- | --- |
| Plafond distinct `maximum_accepted_amount_snapshot` | **Absent** — un seul montant (estimation / `booking.amount`) | Bloquant 7B |
| Client accepte explicitement un max | Non — pas de contrôle UI ni champ | Bloquant |
| Demande ≠ contrat | Partiellement clair côté e-mail ; CGV 1.0 dit encore « confirme sa réservation » ; UI parle parfois de « commande » | Moyen (texte + wording) |
| `contract_formed = false` explicite | Implicite via `amount_is_contractual=false` + pas d’événement de confirmation transport | Moyen |

**Constat** : le 1er clic actuel est déjà une **demande de marché**, pas un contrat avec un transporteur identifié. Il manque le **plafond** comme donnée contractuelle distincte.

---

## 3. État actuel — « acceptation » transporteur

### Ce que fait le code

| Fait | Preuve |
| --- | --- |
| Seed d’offres marché ouvert sans company assignée | `seed_dispatch_offers_for_unassigned_booking` |
| Filtre hold par créancier (X tenu, Y/Z ok) | même fichier + `portal_payment_hold` |
| Acceptation : `AcceptReservationUseCase` pose `booking.company_id` et statut `accepted` | `accept_reservation.py` |
| Gate hold à l’acceptation | même use-case (`portal_client_payment_hold`) |
| Pas de saisie d’un prix d’offre distinct pour PORTAL dans ce use-case | `execute()` ne lit/écrit pas un `offer_price` |
| Route accept : peut écraser `booking.amount` avec un **tarif clinique séjour** (cas non PORTAL typique) | `companies.py` AcceptReservation |
| Aucun événement `CARRIER_OFFERED` / `CLIENT_TRANSPORT_CONFIRMED` | types d’événements limités à `BOOKING_CREATED` / `MODIFIED` / `CANCELLED` |
| Client peut voir `company_name` une fois assigné (liste / détail réservations) | `ReservationsPage.jsx` |
| Pas d’écran récap « X + CHF prix + annulation » ni 2e clic | absents |

### Visibilité montant côté transporteur

- Le booking porte un `amount` (souvent l’estimation LIRIE / indicative).
- Les offres marché (`DispatchOffer`) ne portent pas un modèle « prix proposé par X vs plafond client ».
- **Aucun plafond client n’existe** aujourd’hui ; la question « le transporteur voit-il le plafond ? » est donc **N/A**.
- Il voit en pratique le booking et son `amount` (estimation) selon les écrans entreprise — risque que l’estimation soit prise pour un prix cible.

**Pour 7B (non décidé ici, signalé)** : X peut saisir librement son prix sans voir le plafond ; LIRIE compare côté serveur `prix_X <= plafond`.

### Écarts vs cible

| Cible | Actuel | Gravité |
| --- | --- | --- |
| `CARRIER_OFFERED` avec prix X | **Absent** — acceptation = assignation company | Bloquant |
| Gate `prix_X <= plafond` (rejet strict si >) | **Absent** | Bloquant |
| Conditions d’annulation X versionnées présentées | **Absent** au moment de l’offre | Bloquant |
| Offre > plafond inéligible, pas de 3e sollicitation | N/A (pas de plafond) | — |

**Constat** : l’action entreprise actuelle signifie « je prends la course » (assignation), **pas** « je propose CHF Y sous conditions Z au client pour qu’il confirme ». Avec la cible double validation, le nom **`CARRIER_OFFERED`** est le bon ; **`CARRIER_ACCEPTED` tromperait**.

---

## 4. État actuel — 2e clic / formation du contrat

| Question | Réponse actuelle |
| --- | --- |
| Existe-t-il un 2e clic « Confirmer le transport à CHF … » ? | **Non** |
| Le client doit-il accepter explicitement X + prix + annulation avant exécution ? | **Non** |
| Quel événement marque la formation du contrat de transport ? | **Aucun** — seul `BOOKING_CREATED` (demande + estimation) |
| Après acceptation X, le client reçoit-il un récap prix ferme ? | **Non** constaté (pas d’événement ni flux UI dédié) |

### Écart

Le point contractuel central cible **`CLIENT_TRANSPORT_CONFIRMED`** n’existe pas. Aujourd’hui, après acceptation entreprise, le booking a un transporteur et un `amount` historique (souvent estimatif) **sans** second consentement client sur un prix de X.

---

## 5. Facture vs prix

| Fait | Preuve |
| --- | --- |
| Créance PORTAL = `PortalReceivable` après `COMPLETED` / `RETURN_COMPLETED` | `portal_receivable.py` / doc 6B |
| Créancier = `booking.company_id` ; débiteur = snapshot `BOOKING_CREATED` | idem |
| Montant facturé **saisi** ; **≠** `booking.amount` | docstring + service |
| LIRIE n’est pas créancier | hold / receivable par company |

### Tension avec la cible

- Cible : prix contractuel figé au 2e clic ; facture après course **reprend** ce prix (+ suppléments seulement s’ils étaient prévus au 2e clic).
- Actuel : facture saisie librement après course, sans lien à un prix contractuel client confirmé (parce que ce prix n’existe pas).

Écart **bloquant** pour l’alignement preuve ↔ créance (à traiter en 7B : facture ancrée sur `CLIENT_TRANSPORT_CONFIRMED`, ajustements explicitement prévus).

---

## 6. Annulation / no-show / attente

| Fait | Preuve |
| --- | --- |
| CGV 1.0 | Aucun barème d’annulation / no-show |
| `compute_cancellation_fee` | Politique JSON **entreprise** (`CompanyBillingSettings.cancellation_policy`), basée sur `booking.amount` — orientée facturation S1/S2 |
| Annulation client PORTAL | Événement `BOOKING_CANCELLED` append-only ; pas de snapshot de barème présenté au client avant engagement |
| No-show | Enums / messages métier existent ; **pas** de règle tarifaire PORTAL présentée au client |
| Attente / suppléments | Non exposés au client privé comme conditions d’engagement |

### Écart vs cible

Les règles doivent être **celles de X**, stockées / présentées / hashées au moment de `CARRIER_OFFERED` puis prouvées au `CLIENT_TRANSPORT_CONFIRMED`. Aujourd’hui : politiques entreprise éventuelles pour la facturation interne, **jamais** le parcours « client voit les règles de X puis dit oui ».

7A **n’invente aucun montant**.

---

## 7. Écarts annexes (textes 1.0 ↔ système)

| Texte / sujet | Code / produit | Écart |
| --- | --- | --- |
| « confirme sa réservation » (CGV 1.0) | 1er clic = demande marché | Formulation trop forte |
| « estimation indicative » seule | OK pour LIRIE ; incomplet vs plafond + prix X + prix contractuel | Incomplet pour la cible |
| « Une version ultérieure fait l'objet d'une nouvelle acceptation » | `requires_reacceptance` peut être `false` | Contradiction partielle |
| PAYMENT_HOLD / contestation / recouvrement | Politiques 6C+ implémentées / documentées | Absents des 1.0 |
| Identité exploitant / fors | Absents des canoniques 1.0 | Constat SECO ; hors rédaction 7A |

---

## 8. Matrice d’écarts (synthèse)

| # | Élément cible | État code | Gravité | Étape |
| --- | --- | --- | --- | --- |
| 1 | Plafond distinct au 1er clic | Absent | Bloquant | 7B |
| 2 | Demande = pas contrat (vocabulaire) | Partiel (e-mail OK, CGV/UI mitigés) | Moyen | 7C (+ UI 7B) |
| 3 | `CARRIER_OFFERED` + prix X | Absent (accept = assign) | Bloquant | 7B |
| 4 | Gate strict `prix_X <= plafond` | Absent | Bloquant | 7B |
| 5 | UI 2e clic + conditions X | Absent | Bloquant | 7B |
| 6 | `CLIENT_TRANSPORT_CONFIRMED` | Absent | Bloquant | 7B |
| 7 | Ne pas muter `BOOKING_CREATED` | Respecté aujourd’hui | À préserver | 7B |
| 8 | Facture ancrée sur prix contractuel | Montant saisi libre | Haut | 7B |
| 9 | Policies annulation versionnées X | Pas dans le parcours client | Bloquant | 7B |
| 10 | Masquer le plafond à X (option) | N/A | Décision UX 7B | 7B |

---

## 9. Réponses aux trois questions centrales (état actuel)

### Qui contracte avec qui ?

- **Aujourd’hui** : le client passe une **demande** à LIRIE ; un transporteur peut s’**assigner** ; la **créance** naît plus tard via `PortalReceivable` (X créancier, titulaire débiteur). Il n’y a **pas** d’événement prouvant un contrat de transport formé par consentements croisés prix + annulation.
- **Cible** : contrat de transport entre client et X au `CLIENT_TRANSPORT_CONFIRMED` ; LIRIE reste infrastructure.

### À quel moment le contrat de transport est-il formé ?

- **Aujourd’hui** : **indéterminé** dans le code (pas de point de formation explicite). Le 1er clic n’est pas ce point ; l’acceptation entreprise non plus au sens de la cible.
- **Cible** : exclusivement le **2e clic** `CLIENT_TRANSPORT_CONFIRMED`.

### Comment le prix réellement dû est-il déterminé ?

- **Aujourd’hui** : estimation LIRIE / `booking.amount` non contractuel ; facture saisie après course par X. **Pas** de prix de X communiqué et confirmé avant exécution.
- **Cible** : prix dû de base = prix X confirmé au 2e clic (≤ plafond) ; facture après course alignée ; plafond et estimation ne sont jamais la dette.

---

## 10. Entrées pour 7B (liste d’implémentation — ne pas coder ici)

1. Champ / snapshot `maximum_accepted_amount` au 1er clic (distinct de l’estimation).
2. UI 1er clic : afficher estimation + faire accepter le plafond explicitement.
3. Modèle `CARRIER_OFFERED` (prix X + policy annulation id/version/snapshot/hash).
4. Gate serveur strict `prix_X <= plafond` (rejet si > ; **pas** de parcours « augmenter le plafond »).
5. Décider si X voit le plafond (recommandation produit : **non** — comparaison serveur seule).
6. Écran client 2e clic : X, prix, rappel plafond, annulation / no-show / attente.
7. Événement `CLIENT_TRANSPORT_CONFIRMED` immuable = formation du contrat.
8. Ancrer `PortalReceivable` sur le prix contractuel (+ règles d’ajustement prévues).
9. Préserver append-only : ne jamais réécrire `BOOKING_CREATED`.

Puis **7C** : Conditions transport / CGU 2.0 (`requires_reacceptance = true`) alignées sur cette mécanique — sans toucher aux fichiers 1.0.
