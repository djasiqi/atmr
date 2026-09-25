# Politique canal LIRIE — plafonds d’annulation (pas de taux imposés)

```text
STATUS : LEGAL DESK PASS WITH SAFEGUARDS — IMPLEMENTATION GO (flags OFF)
PARENT : portal-double-validation-7b.md (7B.3 conservé) + portal-single-click-conditional-order-7b5.md
EXTERNAL LAWYER : NOT A CODE BLOCKER
PRODUCTION : NO jusqu’aux gates d’implémentation
```

## Freeze produit → code

```text
7B.3
→ structure existante conservée (politique entreprise versionnée / publiée / hashée)

CHANNEL CAPS
→ contrat produit verrouillé
→ implémentation 7B.5 (refus explicite, jamais min)
```

**7B.3 n’est pas remplacé** : la politique d’annulation reste **propre à chaque entreprise**,
encadrée ensuite par des plafonds canal LIRIE.

## Contrat produit (verrouillé)

```text
PRIX TRANSPORT
→ librement défini par chaque entreprise

POLITIQUE ANNULATION ENTREPRISE
→ librement définie par chaque entreprise

CAPS CANAL LIRIE
→ bornes maximales
→ politique entreprise doit respecter :
   company_rate <= channel_cap

0 % reste toujours autorisé
```

Exemple :

```text
Cap LIRIE 4–24 h = 50 %

Entreprise A = 30 %  → admissible
Entreprise B = 50 %  → admissible
Entreprise C = 0 %   → admissible
Entreprise D = 70 %  → non admissible
```

Exemple de grille canal (valeurs à figer en revue, pas encore code) :

```text
POLITIQUE CANAL LIRIE (MAXIMA)

> 24 h avant départ     → maximum 0 %
4 h – 24 h              → maximum 50 %
< 4 h                   → maximum 100 %
Annulation imputable transporteur → 0 %
No-show                 → maximum 100 %
```

## Interdit : clamp silencieux

**Mauvaise** implémentation — à ne jamais faire :

```text
min(company_rate, channel_cap)
```

qui corrigerait silencieusement la politique de l’entreprise.

**Correct** :

```text
company_rate > channel_cap
→ publication refusée
→ erreur explicite (ex. portal_cancellation_exceeds_channel_cap)
```

Ainsi l’entreprise sait exactement ce qu’elle publie, et le client ne reçoit jamais
une règle contractuelle transformée silencieusement par LIRIE.

Même règle au moment de l’offre / formation du contrat : **refuser**,
ne pas réécrire.

## Séparation des artefacts (futur code)

```text
CompanyPortalCancellationPolicy   (déjà 7B.3)
→ politique choisie par l’entreprise
→ versionnée / publiée / hashée

LirieChannelCancellationPolicy    (à créer après PASS juridique)
→ plafonds du canal
→ versionnée / publiée / hashée
```

### Lors d’une offre / formation

```text
company policy (version + hash)
+
channel caps (version + hash)
↓
validation stricte : chaque company_rate <= channel_cap
↓
snapshot contractuel exact (les deux artefacts)
```

Preuve claire de :

```text
politique entreprise applicable
version des caps LIRIE applicable
conformité au moment de la formation du contrat
```

## Clause d’adhésion transporteur (esprit)

> Pour les demandes provenant du canal Client privé LIRIE, l’entreprise demeure
> libre de déterminer le prix de ses prestations de transport. Elle accepte
> toutefois de respecter les plafonds de frais d’annulation, de non-présentation
> et autres protections du client définis par la politique du canal LIRIE en vigueur.

```text
hors LIRIE     → politique commerciale propre du transporteur
via LIRIE      → règles protectrices du canal (plafonds max)
```

Condition d’accès au canal — pas une réécriture unilatérale de toutes les CG entreprise.

## Correction de cadre (CH / concurrence)

**Ne pas** imposer un pourcentage d’annulation **identique** à toutes les entreprises
(ex. « &lt; 24 h = exactement 80 % ») sans avis spécialisé LCart / COMCO.

Les frais d’annulation ont une **dimension tarifaire**. Harmoniser un taux fixe
entre concurrents via le canal peut être traité comme une condition commerciale
sensible, même si le prix du transport reste libre.

Références de contexte (non exhaustives) :

- [COMCO — recommandations droit des cartels](https://www.weko.admin.ch/fr/recommandations-et-precisions-relatives-au-droit-des-cartels)
- [COMCO — note explicative accords verticaux (2022)](https://www.weko.admin.ch/dam/weko/fr/dokumente/2022/erlaeuterungen_zur_vertikalbekanntmachung_vom_12_dezember_2022.pdf.download.pdf/Note%20explicative%20sur%20la%20Communication%20concernant%20l%27appr%C3%A9ciation%20des%20accords%20verticaux%20du%2012%20d%C3%A9cembre%202022.pdf)
- [SECO — avant l’achat / conclusion](https://www.seco.admin.ch/fr/avant-l-achat-et-conclusion-du-contrat)
- [SECO — CG abusives / art. 8 LCD](https://www.seco.admin.ch/fr/conditions-generales-abusives)

## Lien avec allègement / suppression du 2e clic

Compatible : au 1er clic le client accepte **prix maximum + cadre maximal d’annulation**
(caps canal). La première entreprise compatible forme ensuite le contrat à **son**
prix et avec **sa** politique, sous réserve `company_rate <= channel_cap`.

Affichage indicatif avant 1er clic :

```text
Trajet / date / heure
Prix maximum accepté : CHF 52

Annulation (plafonds canal) :
>24 h : sans frais
4–24 h : maximum 50 %
<4 h : maximum 100 %

Le prix définitif sera celui du transporteur qui accepte,
sans dépasser CHF 52 ; ses frais d’annulation ne dépasseront
pas ces plafonds.
```

CTA indicatif : **Commander jusqu’à CHF 52.–**

Voir aussi [portal-preferred-carrier-autoconfirm-7b4.md](portal-preferred-carrier-autoconfirm-7b4.md).

## État code actuel (écart)

| Élément | Aujourd’hui (7B.3) | Cible (après PASS) |
| --- | --- | --- |
| Policy entreprise | `CompanyPortalCancellationPolicy` | **Inchangé** (conservé) |
| Caps canal | Absent | `LirieChannelCancellationPolicy` |
| Publication si `rate > cap` | N/A | **Refus explicite** (jamais `min()`) |
| Snapshot offre | Policy entreprise seule | Policy entreprise **+** version caps |
| UI 1er clic | Plafond prix | Plafond prix **+** caps annulation canal |

✅ **Implémenté** : 7B.3 publication entreprise ; textes 2.0 sans barème LIRIE générique.

**Reste à faire** (uniquement après `LEGAL REVIEW = PASS`) :

1. Modèle + versioning `LirieChannelCancellationPolicy`
2. Gate publication / offre : refus si `company_rate > channel_cap`
3. Snapshot dual sur offre / confirmation
4. UI 1er clic : caps canal
5. Clause adhésion transporteur

## Points juridiques à faire valider (ciblés)

```text
1) FORMATION DU CONTRAT
   Commande conditionnelle adressée au pool de transporteurs LIRIE éligibles,
   le cocontractant devenant identifié au moment de son acceptation
   (identité non connue au 1er clic).

2) LCART / ACCORDS VERTICAUX
   Plafonds communs de frais d’annulation / no-show du canal
   (maxima protecteurs, entreprise libre en dessous, refus si dépassement)
   — vs risque de fixation de fait si les caps deviennent des taux de référence.
```

Ces deux points figurent aussi dans [portal-legal-review-package-7d-l.md](portal-legal-review-package-7d-l.md).
