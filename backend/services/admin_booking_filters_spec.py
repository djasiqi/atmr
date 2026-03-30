"""Définitions métier des filtres plateforme admin — réservations.

Ces règles sont la source de vérité pour aligner la liste, les compteurs `summary`
et les filtres URL. Toute évolution doit être reflétée ici et dans
`admin_platform_bookings.apply_admin_booking_filters`.
"""

# unassigned: réservation sans entreprise propriétaire assignée (pas encore prise par une entreprise).
# SQL: company_id IS NULL ET executing_company_id IS NULL.
UNASSIGNED_DEFINITION = (
    "Non assignée = aucune entreprise propriétaire ni exécutante (company_id et "
    "executing_company_id tous deux NULL)."
)

# with_transfer: au moins un transfert partenaire accepté ou complété pour cette réservation.
# SQL: EXISTS booking_transfers WHERE booking_id = … AND status IN (ACCEPTED, COMPLETED).
WITH_TRANSFER_DEFINITION = (
    "Avec transfert = existe un BookingTransfer en statut ACCEPTED ou COMPLETED pour la réservation."
)

# incomplete_data: données minimales manquantes pour exploitation sereine.
INCOMPLETE_DATA_DEFINITION = (
    "Données incomplètes = scheduled_time NULL OU customer_name vide OU pickup_location vide "
    "OU dropoff_location vide."
)

# needs_investigation: signal léger pour le support (sous-ensemble minimal, extensible).
NEEDS_INVESTIGATION_DEFINITION = (
    "À investiguer = données incomplètes OU (statut PENDING et transport prévu dépassé depuis > 24h) "
    "OU transfert uniquement en PENDING (bloqué)."
)

# institution_q: filtre texte (ILIKE %terme%) sur le nom de l'institution liée au client.
INSTITUTION_Q_DEFINITION = (
    "Filtre nom institution = sous-chaîne insensible à la casse sur institutions.name "
    "via client.linked_institution."
)

# company_q: filtre texte sur le nom de l'entreprise propriétaire ou exécutante.
COMPANY_Q_DEFINITION = (
    "Filtre nom entreprise = sous-chaîne insensible à la casse sur companies.name "
    "pour company ou executing_company."
)
