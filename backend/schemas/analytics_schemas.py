"""✅ Schemas Marshmallow pour validation des query params des endpoints analytics."""

from marshmallow import Schema, fields, validate

from schemas.validation_utils import ISO8601_DATE_REGEX


class AnalyticsDashboardQuerySchema(Schema):
    """Schema pour validation query params GET /api/analytics/dashboard."""
    
    period = fields.Str(
        validate=validate.OneOf(
            ["7d", "30d", "90d", "1y"],
            error="period doit être: 7d, 30d, 90d ou 1y"
        ),
        load_default="30d"
    )
    start_date = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="start_date doit être au format YYYY-MM-DD"
        ),
        allow_none=True,
        load_default=None
    )
    end_date = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="end_date doit être au format YYYY-MM-DD"
        ),
        allow_none=True,
        load_default=None
    )


class AnalyticsInsightsQuerySchema(Schema):
    """Schema pour validation query params GET /api/analytics/insights."""
    
    lookback_days = fields.Int(
        validate=validate.Range(min=1, max=365, error="lookback_days doit être entre 1 et 365"),
        load_default=30
    )


class AnalyticsWeeklySummaryQuerySchema(Schema):
    """Schema pour validation query params GET /api/analytics/weekly-summary."""
    
    week_start = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="week_start doit être au format YYYY-MM-DD"
        ),
        allow_none=True,
        load_default=None
    )


class AnalyticsExportQuerySchema(Schema):
    """Schema pour validation query params GET /api/analytics/export."""
    
    start_date = fields.Str(
        required=True,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="start_date doit être au format YYYY-MM-DD"
        )
    )
    end_date = fields.Str(
        required=True,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="end_date doit être au format YYYY-MM-DD"
        )
    )
    format = fields.Str(
        validate=validate.OneOf(["csv", "json"], error="format doit être: csv ou json"),
        load_default="csv"
    )

