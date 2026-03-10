from marshmallow import Schema, fields, validate


class ServiceAreaCreateSchema(Schema):
    geo_unit_id = fields.Int(required=True, validate=validate.Range(min=1))
    coverage_mode = fields.Str(
        required=True,
        validate=validate.OneOf(
            ["A_STRICT", "B_PICKUP_ONLY", "C_INTRA_ONLY", "D_NATIONAL"]
        ),
    )
    weight = fields.Int(load_default=0)
    is_active = fields.Bool(load_default=True)


class ServiceAreaUpdateSchema(Schema):
    coverage_mode = fields.Str(
        required=False,
        validate=validate.OneOf(
            ["A_STRICT", "B_PICKUP_ONLY", "C_INTRA_ONLY", "D_NATIONAL"]
        ),
    )
    weight = fields.Int(required=False)
    is_active = fields.Bool(required=False)
