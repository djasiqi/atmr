from __future__ import annotations

from sqlalchemy import or_

import manage
from ext import db
from models import (
    Booking,
    Client,
    ClientBillingParty,
    ClientStay,
    Company,
    CompanyBillingSettings,
    DelayEvent,
    Driver,
    Invoice,
    InvoiceLine,
    InvoicePayment,
    InvoiceReminder,
    Payment,
    TransportRequest,
    TransportVoucher,
    User,
)


def _is_demo_email_expr(column):
    return or_(
        column.ilike("%@demo.local"),
        column.ilike("%@demo.lirie.ch"),
        column.ilike("demo-%@%"),
        column.ilike("%@internal.atmr.local"),
    )


def run() -> dict[str, int]:
    app = manage.app
    with app.app_context():
        demo_users = User.query.filter(_is_demo_email_expr(User.email)).all()
        user_ids = [u.id for u in demo_users]
        if not user_ids:
            return {"demo_users_found": 0, "deleted": 0}

        demo_clients = Client.query.filter(Client.user_id.in_(user_ids)).all()
        client_ids = [c.id for c in demo_clients]
        demo_companies = Company.query.filter(Company.user_id.in_(user_ids)).all()
        company_ids = [c.id for c in demo_companies]

        deleted_transport_vouchers = TransportVoucher.query.filter(
            TransportVoucher.client_id.in_(client_ids)
        ).delete(synchronize_session=False)
        deleted_client_billing_links = ClientBillingParty.query.filter(
            ClientBillingParty.client_id.in_(client_ids)
        ).delete(synchronize_session=False)
        deleted_client_stays = ClientStay.query.filter(
            ClientStay.client_id.in_(client_ids)
        ).delete(synchronize_session=False)
        deleted_payments = Payment.query.filter(
            Payment.client_id.in_(client_ids)
        ).delete(synchronize_session=False)
        invoice_ids_subquery = db.session.query(Invoice.id).filter(
            or_(
                Invoice.client_id.in_(client_ids),
                Invoice.bill_to_client_id.in_(client_ids),
                Invoice.company_id.in_(company_ids),
            )
        )
        deleted_invoice_payments = InvoicePayment.query.filter(
            InvoicePayment.invoice_id.in_(invoice_ids_subquery)
        ).delete(synchronize_session=False)
        deleted_invoice_reminders = InvoiceReminder.query.filter(
            InvoiceReminder.invoice_id.in_(invoice_ids_subquery)
        ).delete(synchronize_session=False)
        deleted_invoice_lines = InvoiceLine.query.filter(
            InvoiceLine.invoice_id.in_(invoice_ids_subquery)
        ).delete(synchronize_session=False)
        deleted_invoices = Invoice.query.filter(
            or_(
                Invoice.client_id.in_(client_ids),
                Invoice.bill_to_client_id.in_(client_ids),
                Invoice.company_id.in_(company_ids),
            )
        ).delete(synchronize_session=False)
        booking_ids_subquery = db.session.query(Booking.id).filter(
            or_(
                Booking.client_id.in_(client_ids),
                Booking.user_id.in_(user_ids),
                Booking.company_id.in_(company_ids),
            )
        )
        deleted_delay_events = DelayEvent.query.filter(
            DelayEvent.booking_id.in_(booking_ids_subquery)
        ).delete(synchronize_session=False)
        deleted_bookings = Booking.query.filter(
            or_(
                Booking.client_id.in_(client_ids),
                Booking.user_id.in_(user_ids),
                Booking.company_id.in_(company_ids),
            )
        ).delete(synchronize_session=False)
        deleted_transport_requests = TransportRequest.query.filter(
            or_(
                TransportRequest.created_by_user_id.in_(user_ids),
                TransportRequest.accepted_by_company_id.in_(company_ids),
            )
        ).delete(synchronize_session=False)
        deleted_drivers = Driver.query.filter(Driver.user_id.in_(user_ids)).delete(
            synchronize_session=False
        )
        deleted_clients = Client.query.filter(Client.id.in_(client_ids)).delete(
            synchronize_session=False
        )
        deleted_company_billing_settings = CompanyBillingSettings.query.filter(
            CompanyBillingSettings.company_id.in_(company_ids)
        ).delete(synchronize_session=False)
        deleted_companies = Company.query.filter(Company.id.in_(company_ids)).delete(
            synchronize_session=False
        )
        deleted_users = User.query.filter(User.id.in_(user_ids)).delete(
            synchronize_session=False
        )

        db.session.commit()
        return {
            "demo_users_found": len(user_ids),
            "demo_clients_found": len(client_ids),
            "deleted_clients": int(deleted_clients or 0),
            "deleted_users": int(deleted_users or 0),
            "deleted_bookings": int(deleted_bookings or 0),
            "deleted_delay_events": int(deleted_delay_events or 0),
            "deleted_invoices": int(deleted_invoices or 0),
            "deleted_invoice_payments": int(deleted_invoice_payments or 0),
            "deleted_invoice_reminders": int(deleted_invoice_reminders or 0),
            "deleted_invoice_lines": int(deleted_invoice_lines or 0),
            "deleted_transport_requests": int(deleted_transport_requests or 0),
            "deleted_drivers": int(deleted_drivers or 0),
            "deleted_company_billing_settings": int(
                deleted_company_billing_settings or 0
            ),
            "deleted_companies": int(deleted_companies or 0),
            "deleted_transport_vouchers": int(deleted_transport_vouchers or 0),
            "deleted_client_billing_links": int(deleted_client_billing_links or 0),
            "deleted_client_stays": int(deleted_client_stays or 0),
            "deleted_payments": int(deleted_payments or 0),
        }


if __name__ == "__main__":
    print(run())
