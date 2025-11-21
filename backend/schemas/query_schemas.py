"""✅ Schemas Marshmallow réutilisables pour validation des query parameters GET.

Fournit des schémas communs pour pagination, filtres, dates, recherche, etc.
"""

from marshmallow import Schema, fields, validate

from schemas.validation_utils import ISO8601_DATE_REGEX


class PaginationQuerySchema(Schema):
    """Schema réutilisable pour pagination (page, per_page)."""

    page = fields.Int(
        load_default=1,
        validate=validate.Range(min=1, error="page doit être >= 1"),
    )
    per_page = fields.Int(
        load_default=50,
        validate=validate.Range(min=1, max=500, error="per_page doit être entre 1 et 500"),
    )


class DateRangeQuerySchema(Schema):
    """Schema réutilisable pour plage de dates (start_date, end_date)."""

    start_date = fields.Str(
        load_default=None,
        validate=validate.Regexp(ISO8601_DATE_REGEX, error="start_date doit être au format YYYY-MM-DD"),
        allow_none=True,
    )
    end_date = fields.Str(
        load_default=None,
        validate=validate.Regexp(ISO8601_DATE_REGEX, error="end_date doit être au format YYYY-MM-DD"),
        allow_none=True,
    )


class SearchQuerySchema(Schema):
    """Schema réutilisable pour recherche textuelle (search)."""

    search = fields.Str(
        load_default=None,
        validate=validate.Length(max=200, error="search doit faire max 200 caractères"),
        allow_none=True,
    )


class FilterQuerySchema(Schema):
    """Schema réutilisable pour filtres communs (company_id, status, etc.)."""

    company_id = fields.Int(
        load_default=None,
        validate=validate.Range(min=1, error="company_id doit être > 0"),
        allow_none=True,
    )
    status = fields.Str(
        load_default=None,
        validate=validate.Length(max=50, error="status doit faire max 50 caractères"),
        allow_none=True,
    )


class LimitOffsetQuerySchema(Schema):
    """Schema réutilisable pour limit/offset (alternative à pagination)."""

    limit = fields.Int(
        load_default=50,
        validate=validate.Range(min=1, max=100, error="limit doit être entre 1 et 100"),
    )
    offset = fields.Int(
        load_default=0,
        validate=validate.Range(min=0, error="offset doit être >= 0"),
    )
