# STOP GATE institution-external-01 — Compatibilité frontend institution

## Objectif

Garantir que le portail institution affiche correctement les missions transporteur externe sans régression LIRIE.

## Règles

1. **Aucun crash** sur une mission `carrier_source = external` : pas d'accès non gardé à `booking_summary`.
2. **Helpers centralisés** : `requestStatus.js` et `carrierDisplay.js` sont la source unique pour statut et transporteur.
3. **Liste** : badge Mode, filtre `carrier_source`, statuts `EXTERNAL_ASSIGNED` / `EXTERNAL_DECLARED_COMPLETED`, pill `EXTERNE` sur le transporteur.
4. **Détail** : section « Transporteur externe » distincte du bloc LIRIE (booking/chauffeur/véhicule).
5. **Actions LIRIE masquées** en externe : Envoyer, chat booking, édition opérationnelle booking.
6. **Actions externes** :
   - DRAFT/SENT : « Affecter transporteur externe »
   - EXTERNAL_ASSIGNED : « Déclarer réalisée »
7. **Timeline** : événements `external_carrier_assigned`, `external_carrier_switched`, `external_mission_completed` via labels API ; jalons locaux en fallback.

✅ **Implémenté** :
- `[frontend/src/utils/requestStatus.js](frontend/src/utils/requestStatus.js)`
- `[frontend/src/utils/carrierDisplay.js](frontend/src/utils/carrierDisplay.js)`
- `[frontend/src/pages/institution/Requests/InstitutionRequests.jsx](frontend/src/pages/institution/Requests/InstitutionRequests.jsx)`
- `[frontend/src/pages/institution/Requests/RequestDetailPanel.jsx](frontend/src/pages/institution/Requests/RequestDetailPanel.jsx)`

## STOP GATE INSTITUTION-EXTERNAL-02 — Création / mode d'exécution

Depuis le formulaire de création (`InstitutionRequestCreate.jsx`) :

| Mode | Comportement attendu |
|---|---|
| Brouillon | Crée une demande `DRAFT` |
| Envoyer LIRIE | Crée puis envoie (flux offres existant) |
| Transporteur externe | Crée en `DRAFT` puis `assignExternalCarrier` → `EXTERNAL_ASSIGNED` |

### Garde-fous

- Aucun `RequestOffer` ni `Booking` en mode externe.
- Si `assignExternalCarrier` échoue après création : message explicite « La demande a été créée, mais le transporteur externe n'a pas été affecté » + navigation vers le détail DRAFT pour réessayer.
- Bouton dynamique selon le mode : brouillon / LIRIE / externe.
- Message UX sous le choix externe (non inscrit LIRIE, pas d'offre/réservation, déclaration manuelle).

✅ **Implémenté** :
- `[frontend/src/pages/institution/Requests/InstitutionRequestCreate.jsx](frontend/src/pages/institution/Requests/InstitutionRequestCreate.jsx)`
- `[frontend/src/components/institution/ExternalCarrierFields.jsx](frontend/src/components/institution/ExternalCarrierFields.jsx)`
- Hooks `useAssignExternalCarrier` dans `[frontend/src/hooks/useInstitutionData.js](frontend/src/hooks/useInstitutionData.js)`

## Non-régression LIRIE

- Flux DRAFT → SENT → ACCEPTED → CONVERTED inchangé.
- Affichage booking_summary / chat / édition opérationnelle conservés pour les missions LIRIE converties.
