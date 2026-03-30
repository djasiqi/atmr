"""Codes d'erreur stables pour l'API chauffeur (JSON `code`, observabilité)."""

# 403 — la course n'est plus assignée à ce chauffeur (ex. réassignation)
BOOKING_ASSIGNED_TO_OTHER_DRIVER = "BOOKING_ASSIGNED_TO_OTHER_DRIVER"

# 403 — entreprise / exécution (à distinguer du wrong driver)
BOOKING_COMPANY_FORBIDDEN = "BOOKING_COMPANY_FORBIDDEN"
