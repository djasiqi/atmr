from flask_restx import Namespace

geocode_ns = Namespace(
    "geocode", description="Autocomplete & géocodage avec Google Places API"
)
