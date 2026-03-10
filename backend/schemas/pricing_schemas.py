from marshmallow import Schema, fields, validate


class PricingSimulateBookingSchema(Schema):
    pickup_at = fields.Str(required=True)
    is_round_trip = fields.Bool(load_default=False)
    pickup_geo_unit_id = fields.Int(required=False, allow_none=True)
    dropoff_geo_unit_id = fields.Int(required=False, allow_none=True)
    pickup_admin_token = fields.Str(required=False, allow_none=True, validate=validate.Length(max=64))
    dropoff_admin_token = fields.Str(required=False, allow_none=True, validate=validate.Length(max=64))
    pickup_lat = fields.Float(required=False, allow_none=True, validate=validate.Range(min=-90, max=90))
    pickup_lng = fields.Float(required=False, allow_none=True, validate=validate.Range(min=-180, max=180))
    dropoff_lat = fields.Float(required=False, allow_none=True, validate=validate.Range(min=-90, max=90))
    dropoff_lng = fields.Float(required=False, allow_none=True, validate=validate.Range(min=-180, max=180))
    pickup_zip = fields.Str(required=False, allow_none=True, validate=validate.Length(max=16))
    dropoff_zip = fields.Str(required=False, allow_none=True, validate=validate.Length(max=16))
    distance_km = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0))
    route_points = fields.List(fields.Raw(), required=False, allow_none=True)
    route_geometry = fields.Raw(required=False, allow_none=True)
    requires_waiting = fields.Bool(load_default=False)


class PricingSimulateRequestSchema(Schema):
    pricing_profile_version_id = fields.Int(required=True, validate=validate.Range(min=1))
    booking = fields.Nested(PricingSimulateBookingSchema, required=True)


class PricingZoneMatrixSettingsSchema(Schema):
    model = fields.Str(load_default="zone_matrix")
    zones = fields.List(fields.Raw(), load_default=list)
    matrix = fields.Dict(keys=fields.Str(), values=fields.Raw(), load_default=dict)
    matrix_symmetry = fields.Bool(load_default=False)
    default_same_zone_price = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0))
    extras = fields.List(fields.Raw(), load_default=list)
    minimum = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0))
