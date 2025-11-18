"""Schémas de sérialisation pour les modèles de dispatch.

Utilise Marshmallow pour une sérialisation cohérente et typée.
Remplace les méthodes .serialize() et .to_dict() dispersées dans les modèles.
"""

from marshmallow import Schema, fields


class DriverSchema(Schema):
    """Schéma pour Driver (version simplifiée)."""

    id = fields.Int(dump_only=True)
    first_name = fields.Str()
    last_name = fields.Str()
    phone = fields.Str(allow_none=True)
    email = fields.Email(allow_none=True)

    # Status
    is_available = fields.Bool()
    is_active = fields.Bool()
    is_emergency = fields.Bool(allow_none=True)

    # Métriques
    punctuality_score = fields.Float(allow_none=True)
    current_load = fields.Int(allow_none=True)

    # Position actuelle
    current_lat = fields.Float(allow_none=True)
    current_lon = fields.Float(allow_none=True)

    class Meta:  # type: ignore
        ordered = True


class BookingSchema(Schema):
    """Schéma pour Booking."""

    id = fields.Int(dump_only=True)

    # Timestamps
    scheduled_time = fields.DateTime(format="iso", allow_none=True)
    created_at = fields.DateTime(format="iso", dump_only=True)
    completed_at = fields.DateTime(format="iso", allow_none=True)

    # Adresses
    pickup_address = fields.Str(allow_none=True)
    dropoff_address = fields.Str(allow_none=True)
    pickup_location = fields.Str(allow_none=True)
    dropoff_location = fields.Str(allow_none=True)

    # Coordonnées GPS
    pickup_lat = fields.Float(allow_none=True)
    pickup_lon = fields.Float(allow_none=True)
    dropoff_lat = fields.Float(allow_none=True)
    dropoff_lon = fields.Float(allow_none=True)

    # Status
    status = fields.Str()

    # Caractéristiques
    is_medical = fields.Bool()
    is_urgent = fields.Bool()
    is_return = fields.Bool(allow_none=True)
    is_round_trip = fields.Bool(allow_none=True)
    priority = fields.Float(allow_none=True)

    # Relations
    company_id = fields.Int()
    client_id = fields.Int(allow_none=True)

    # Informations client (si disponibles)
    client_name = fields.Str(allow_none=True)
    client_phone = fields.Str(allow_none=True)

    # Métriques
    distance_km = fields.Float(allow_none=True)
    duration_minutes = fields.Float(allow_none=True)

    class Meta:  # type: ignore
        ordered = True


class AssignmentSchema(Schema):
    """Schéma pour Assignment."""

    id = fields.Int(dump_only=True)

    # Relations
    booking_id = fields.Int(required=True)
    driver_id = fields.Int(required=True)
    dispatch_run_id = fields.Int(allow_none=True)

    # Timestamps
    created_at = fields.DateTime(format="iso", dump_only=True)
    updated_at = fields.DateTime(format="iso", allow_none=True)
    actual_pickup_at = fields.DateTime(format="iso", allow_none=True)
    actual_dropoff_at = fields.DateTime(format="iso", allow_none=True)

    # Status
    status = fields.Str()
    confirmed = fields.Bool()

    # Métriques
    distance_km = fields.Float(allow_none=True)
    duration_minutes = fields.Float(allow_none=True)
    cost = fields.Float(allow_none=True)

    # Relations nested (optionnelles)
    booking = fields.Nested(BookingSchema, allow_none=True)
    driver = fields.Nested(DriverSchema, allow_none=True)

    class Meta:  # type: ignore
        ordered = True


class DispatchRunSchema(Schema):
    """Schéma pour DispatchRun."""

    id = fields.Int(dump_only=True)
    company_id = fields.Int(required=True)

    # Timestamps
    created_at = fields.DateTime(format="iso", dump_only=True)
    for_date = fields.Date()

    # Configuration
    mode = fields.Str()

    # Métriques
    quality_score = fields.Float(allow_none=True)
    total_bookings = fields.Int(allow_none=True)
    assigned_bookings = fields.Int(allow_none=True)
    unassigned_bookings = fields.Int(allow_none=True)
    total_drivers = fields.Int(allow_none=True)

    # Timing
    solver_time_seconds = fields.Float(allow_none=True)
    total_time_seconds = fields.Float(allow_none=True)

    # Assignments (si demandés)
    assignments = fields.Nested(AssignmentSchema, many=True, exclude=("dispatch_run",), allow_none=True)

    class Meta:  # type: ignore
        ordered = True


class DispatchSuggestionSchema(Schema):
    """Schéma pour suggestions du RealtimeOptimizer."""

    action = fields.Str(required=True)  # 'assign', 'reassign', 'notify', 'unassign'

    # IDs concernés
    assignment_id = fields.Int(allow_none=True)
    booking_id = fields.Int(allow_none=True)
    driver_id = fields.Int(allow_none=True)
    alternative_driver_id = fields.Int(allow_none=True)

    # Contexte
    reason = fields.Str(allow_none=True)
    priority = fields.Str(allow_none=True)  # 'low', 'medium', 'high', 'critical'
    impact_score = fields.Float(allow_none=True)

    # Prédictions
    predicted_delay_minutes = fields.Float(allow_none=True)
    gain_minutes = fields.Float(allow_none=True)
    confidence = fields.Float(allow_none=True)

    class Meta:  # type: ignore
        ordered = True


class DispatchProblemSchema(Schema):
    """Schéma pour le problème de dispatch complet."""

    # Méta
    company_id = fields.Int()
    for_date = fields.Str()
    mode = fields.Str()

    # Données
    bookings = fields.Nested(BookingSchema, many=True)
    drivers = fields.Nested(DriverSchema, many=True)

    # Métriques
    total_bookings = fields.Int()
    total_drivers = fields.Int()

    class Meta:  # type: ignore
        ordered = True


class DispatchResultSchema(Schema):
    """Schéma pour le résultat d'un dispatch."""

    # Run info
    dispatch_run_id = fields.Int(allow_none=True)
    mode = fields.Str()

    # Résultats
    assignments = fields.Nested(AssignmentSchema, many=True)
    unassigned_bookings = fields.Nested(BookingSchema, many=True)

    # Métriques
    total_bookings = fields.Int()
    assigned_count = fields.Int()
    unassigned_count = fields.Int()
    quality_score = fields.Float(allow_none=True)

    # Timing
    solver_time_seconds = fields.Float(allow_none=True)
    total_time_seconds = fields.Float(allow_none=True)

    # Suggestions (si mode semi_auto ou fully_auto)
    suggestions = fields.Nested(DispatchSuggestionSchema, many=True, allow_none=True)

    class Meta:  # type: ignore
        ordered = True


# ============================================================
# Instances des schémas (singleton pattern)
# ============================================================

# Drivers
driver_schema = DriverSchema()
drivers_schema = DriverSchema(many=True)

# Bookings
booking_schema = BookingSchema()
bookings_schema = BookingSchema(many=True)

# Assignments
assignment_schema = AssignmentSchema()
assignments_schema = AssignmentSchema(many=True)

# Assignments avec relations nested
assignment_with_relations_schema = AssignmentSchema()
assignments_with_relations_schema = AssignmentSchema(many=True)

# Dispatch runs
dispatch_run_schema = DispatchRunSchema()
dispatch_runs_schema = DispatchRunSchema(many=True)

# Suggestions
suggestion_schema = DispatchSuggestionSchema()
suggestions_schema = DispatchSuggestionSchema(many=True)

# Problème & Résultat
dispatch_problem_schema = DispatchProblemSchema()
dispatch_result_schema = DispatchResultSchema()
