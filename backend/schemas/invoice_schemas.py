"""✅ Schemas Marshmallow pour validation des endpoints invoices."""

from marshmallow import Schema, fields, validate

# ISO8601_DATE_REGEX et ISO8601_DATETIME_REGEX non nécessaires pour ces schemas


class BillingSettingsUpdateSchema(Schema):
    """Schema pour mise à jour des paramètres de facturation (PUT /api/invoices/companies/<id>/billing-settings)."""
    
    payment_terms_days = fields.Int(
        validate=validate.Range(min=0, max=365, error="payment_terms_days doit être entre 0 et 365 jours")
    )
    overdue_fee = fields.Float(
        validate=validate.Range(min=0, error="overdue_fee doit être >= 0")
    )
    reminder1fee = fields.Float(
        validate=validate.Range(min=0, error="reminder1fee doit être >= 0")
    )
    reminder2fee = fields.Float(
        validate=validate.Range(min=0, error="reminder2fee doit être >= 0")
    )
    reminder3fee = fields.Float(
        validate=validate.Range(min=0, error="reminder3fee doit être >= 0")
    )
    auto_reminders_enabled = fields.Bool()
    email_sender = fields.Str(validate=validate.Length(max=254), allow_none=True)
    invoice_number_format = fields.Str(validate=validate.Length(max=50))
    invoice_prefix = fields.Str(validate=validate.Length(max=20))
    
    # IBAN: format CH + 19 chiffres/lettres
    iban = fields.Str(
        validate=validate.Regexp(
            r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$",
            error="IBAN invalide"
        ),
        allow_none=True
    )
    qr_iban = fields.Str(
        validate=validate.Regexp(
            r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$",
            error="QR IBAN invalide"
        ),
        allow_none=True
    )
    esr_ref_base = fields.Str(validate=validate.Length(max=26), allow_none=True)
    invoice_message_template = fields.Str(validate=validate.Length(max=1000), allow_none=True)
    reminder1template = fields.Str(validate=validate.Length(max=1000), allow_none=True)
    reminder2template = fields.Str(validate=validate.Length(max=1000), allow_none=True)
    reminder3template = fields.Str(validate=validate.Length(max=1000), allow_none=True)
    legal_footer = fields.Str(validate=validate.Length(max=2000), allow_none=True)
    pdf_template_variant = fields.Str(validate=validate.Length(max=50))
    
    # reminder_schedule_days peut être une liste
    reminder_schedule_days = fields.Raw(allow_none=True)
    vat_applicable = fields.Bool()
    vat_rate = fields.Float(
        allow_none=True,
        validate=validate.Range(min=0, max=100, error="vat_rate doit être entre 0 et 100")
    )
    vat_label = fields.Str(validate=validate.Length(max=50), allow_none=True)
    vat_number = fields.Str(validate=validate.Length(max=50), allow_none=True)
    
    class Meta:  # type: ignore
        unknown = "INCLUDE"


class InvoiceGenerateSchema(Schema):
    """Schema pour génération de facture (POST /api/invoices/companies/<id>/invoices/generate)."""
    
    client_id = fields.Int(validate=validate.Range(min=1))
    client_ids = fields.List(
        fields.Int(validate=validate.Range(min=1)),
        validate=validate.Length(min=1, error="client_ids doit contenir au moins un ID")
    )
    bill_to_client_id = fields.Int(validate=validate.Range(min=1), allow_none=True)
    period_year = fields.Int(
        required=True,
        validate=validate.Range(min=2000, max=2100, error="Année invalide (2000-2100)")
    )
    period_month = fields.Int(
        required=True,
        validate=validate.Range(min=1, max=12, error="Mois invalide (1-12)")
    )
    # Sélection manuelle de réservations: { client_id: [reservation_ids] }
    client_reservations = fields.Dict(allow_none=True)
    reservation_ids = fields.List(fields.Int(), allow_none=True)
    overrides = fields.Dict(
        keys=fields.Str(),
        values=fields.Dict(
            keys=fields.Str(),
            values=fields.Raw()
        ),
        allow_none=True
    )

