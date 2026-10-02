# Sélecteur patient de « Nouvelle facture »

L'ouverture du modal **Direct patient** appelle `GET /api/v1/invoices/companies/{id}/invoices/invoice-candidates?payer_type=patient&period=AAAA-MM`.

Cette lecture agrège les courses patient déjà facturables du mois (statut terminé ou annulation facturable, sans ligne de facture, hors revendication active, hors retenue Market LIRIE). Elle renvoie le nom, le nombre de courses et le montant indiqué (`booking.amount`, ou les frais si la course est annulée). Elle ne calcule pas les cliniques, ne prépare pas de lignes, et n'écrit rien en base.

Le menu affiche ce total comme **montant estimé**. La prévisualisation et **Préparer la facture** relisent la base et recalculent le tarif : le cache n'est pas la source de vérité.

Le résultat est mis en cache Redis 60 secondes (`billing:candidates:patient:company_{id}:{période}`). La création d'une facture patient n'invalide que cette clé : même entreprise, même mois, type patient.

L'index existant `ix_booking_company_scheduled` (`company_id`, `scheduled_time`) borne la recherche au mois de l'entreprise. Institutions et partenaires ne sont chargés que si leur onglet est ouvert.

✅ **Implémenté** : le menu patient lit `pendingValidation.count` et `disputed.count` via `presentPatientInvoiceSummary`. Ces deux objets sont toujours renvoyés (`frontend/src/utils/payerInvoiceSummaryUi.js`). L’accès dans `BillPeriodModal.jsx` tolère leur absence, pour ne plus remplacer toute la page factures par « Cannot read properties of undefined (reading 'count') ».

✅ **Implémenté** : le sélecteur reste en lecture seule, mais le payeur de chaque ligne est le payeur effectif (`resolve_effective_patient_billing_party`), pas aveuglément `booking.billing_party_id`. Un BillingParty `PATIENT` technique cède à un tiers par défaut (curatelle). Un override verrouillé, un bon valide ou un séjour actif gardent la priorité. La génération de facture refait la même résolution côté serveur. Un brouillon encore mutable est réconcilié à la régénération du PDF ; une facture déjà figée ne change pas de destinataire.
