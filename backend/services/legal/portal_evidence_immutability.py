"""Immutabilité des preuves contractuelles PORTAL v2 (7B.1).

Les listeners SQLAlchemy sont enregistrés sur les modèles :

- ``models.portal_carrier_offer`` — champs contractuels immuables ;
  seul ``status`` évolue (offered → stale|confirmed|withdrawn) ;
  DELETE physique interdit.
- ``models.portal_client_transport_confirmation`` — append-only strict
  (aucun UPDATE / DELETE).

Ce module documente la politique ; ne pas y re-enregistrer d'events
(risque de double listener).
"""
