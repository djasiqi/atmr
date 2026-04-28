import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from celery import Celery  # pyright: ignore[reportMissingImports]
from celery.schedules import crontab  # pyright: ignore[reportMissingImports]

from application.invoices.check_overdue_invoices import (
    CheckOverdueInvoicesInput,
    CheckOverdueInvoicesUseCase,
)
from application.invoices.generate_invoice import GenerateInvoiceUseCase
from application.invoices.process_automatic_reminders import (
    ProcessAutomaticRemindersInput,
    ProcessAutomaticRemindersUseCase,
)
from models import Booking, Company, Invoice, InvoiceReminder, InvoiceStatus, db

Client: Any = None
try:
    from models import Client as _Client

    Client = _Client
except ImportError:
    # Fallback si Client n'existe pas
    Client = type("Client", (), {"id": None, "company_id": None, "is_active": True})()

# Note: notification_service n'a pas de classe NotificationService,
# ce sont des fonctions
NotificationService: Any = None

MONTH_ONE = 1
MONTH_DECEMBER = 12

app_logger = logging.getLogger("billing_tasks")

# Instance Celery (à adapter selon votre configuration)
celery_app = Celery("billing")


@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=240,
    task_soft_time_limit=180,
    max_retries=2,
    autoretry_for=(Exception,),
)
def check_overdues_and_trigger_reminders():
    """Tâche quotidienne pour vérifier les factures en retard
    et déclencher les rappels automatiques."""
    try:
        app_logger.info("Début de la vérification des factures en retard")

        # Initialiser les use cases
        check_overdue_uc = CheckOverdueInvoicesUseCase()
        process_reminders_uc = ProcessAutomaticRemindersUseCase()

        # Vérifier et marquer les factures en retard
        check_result = check_overdue_uc.execute(CheckOverdueInvoicesInput())
        if not check_result.success:
            app_logger.error(
                "Erreur lors de la vérification des factures en retard: %s",
                check_result.error,
            )

        # Traiter les rappels automatiques
        process_result = process_reminders_uc.execute(ProcessAutomaticRemindersInput())
        if not process_result.success:
            app_logger.error(
                "Erreur lors du traitement des rappels automatiques: %s",
                process_result.error,
            )

        app_logger.info(
            (
                "Vérification des factures en retard terminée: %s factures mises "
                "à jour, %s rappels générés"
            ),
            check_result.updated_count,
            process_result.reminders_generated,
        )

    except Exception as e:
        app_logger.error(
            "Erreur lors de la vérification des factures en retard: %s", str(e)
        )
        raise


@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=240,
    task_soft_time_limit=180,
    max_retries=2,
    autoretry_for=(Exception,),
)
def send_reminder_notifications() -> None:
    """Tâche pour envoyer automatiquement les rappels par email.

    Cette tâche :
    1. Récupère les rappels générés récemment qui n'ont pas été envoyés
    2. Vérifie que le client a un email valide
    3. Envoie le rappel par email via SendReminderByEmailUseCase
    4. Marque le rappel comme envoyé

    Rate limiting : Max 50 emails par exécution pour éviter d'être blacklisté
    """
    try:
        app_logger.info("Début de l'envoi automatique des rappels par email")

        from application.invoices.send_reminder_by_email import (
            SendReminderByEmailInput,
            SendReminderByEmailUseCase,
        )

        send_reminder_uc = SendReminderByEmailUseCase()

        # Récupérer les rappels générés récemment (dernières 24h)
        # qui n'ont pas encore été envoyés et qui ont un PDF
        yesterday = datetime.now(UTC) - timedelta(days=1)
        reminders_to_send = (
            db.session.query(InvoiceReminder)
            .filter(
                InvoiceReminder.generated_at >= yesterday,
                InvoiceReminder.sent_at.is_(None),
                InvoiceReminder.pdf_url.isnot(None),
            )
            .limit(50)  # Rate limiting : max 50 emails par batch
            .all()
        )

        if not reminders_to_send:
            app_logger.info("Aucun rappel à envoyer")
            return

        success_count = 0
        skip_count = 0
        error_count = 0

        for reminder in reminders_to_send:
            try:
                # Vérifier que le client a un email
                invoice = reminder.invoice
                if not invoice:
                    app_logger.warning(
                        "Rappel %s : facture introuvable, skip", reminder.id
                    )
                    skip_count += 1
                    continue

                client = invoice.client if hasattr(invoice, "client") else None
                if not client or not getattr(client, "contact_email", None):
                    app_logger.info(
                        "Rappel %s : client sans email, marquage pour envoi papier",
                        reminder.id,
                    )
                    # Note : le rappel reste avec sent_at=None pour envoi papier manuel
                    skip_count += 1
                    continue

                # Envoyer le rappel par email
                send_input = SendReminderByEmailInput(reminder_id=reminder.id)
                result = send_reminder_uc.execute(send_input)

                if result.success:
                    success_count += 1
                    app_logger.info(
                        "Rappel N°%s envoyé avec succès pour la facture %s à %s",
                        reminder.level,
                        invoice.invoice_number,
                        result.recipient,
                    )
                else:
                    error_count += 1
                    app_logger.error(
                        "Échec de l'envoi du rappel %s: %s", reminder.id, result.error
                    )

            except Exception as e:
                error_count += 1
                app_logger.exception(
                    "Erreur lors du traitement du rappel %s: %s", reminder.id, e
                )
                db.session.rollback()
                continue

        app_logger.info(
            (
                "Envoi automatique des rappels terminé: "
                "%s succès, %s ignorés, %s erreurs (total: %s)"
            ),
            success_count,
            skip_count,
            error_count,
            len(reminders_to_send),
        )

    except Exception as e:
        app_logger.exception("Erreur lors de l'envoi automatique des rappels: %s", e)
        raise


@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=240,
    task_soft_time_limit=180,
    max_retries=1,
    autoretry_for=(Exception,),
)
def generate_monthly_invoices():
    """Tâche mensuelle pour générer automatiquement les factures des clients actifs."""
    try:
        app_logger.info("Début de la génération mensuelle des factures")

        generate_invoice_uc = GenerateInvoiceUseCase()

        # Calculer la période précédente
        now = datetime.now(UTC)
        if now.month == MONTH_ONE:
            period_year = now.year - 1
            period_month = MONTH_DECEMBER
        else:
            period_year = now.year
            period_month = now.month - 1

        # Récupérer toutes les entreprises avec des clients actifs
        companies_with_clients = (
            db.session.query(Company)
            .join(Client)
            .filter(Client.is_active)
            .distinct()
            .all()
        )

        invoices_generated = 0

        for company in companies_with_clients:
            try:
                # Récupérer les clients actifs de cette entreprise
                active_clients = (
                    db.session.query(Client)
                    .filter(Client.company_id == company.id, Client.is_active)
                    .all()
                )

                for client in active_clients:
                    try:
                        # Vérifier qu'il y a des réservations pour cette période
                        from datetime import datetime as dt

                        start_date = dt(period_year, period_month, 1)
                        end_date = (
                            dt(period_year + 1, 1, 1)
                            if period_month == MONTH_DECEMBER
                            else dt(period_year, period_month + 1, 1)
                        )
                        from repositories.booking_repository import BookingRepository

                        booking_repo = BookingRepository()
                        reservations = (
                            booking_repo.find_by_company_and_client_and_period(
                                company_id=company.id,
                                client_id=client.id,
                                start_date=start_date,
                                end_date=end_date,
                                statuses=["COMPLETED", "RETURN_COMPLETED"],
                            )
                        )
                        # Filtrer celles déjà facturées
                        reservations = [
                            r
                            for r in reservations
                            if getattr(r, "invoice_line_id", None) is None
                        ]

                        if reservations:
                            # Générer les factures en "split" par destinataire.
                            # Priorité:
                            # 1) booking.billing_party_id (destinataire unifié explicite)
                            # 2) booking.billed_to_type != patient + booking.billed_to_company_id (clinique)
                            # 3) patient (défaut)
                            from application.invoices.generate_invoice import (
                                GenerateInvoiceInput,
                            )

                            groups: dict[str, dict[str, Any]] = {}
                            for r in reservations:
                                bp_id = getattr(r, "billing_party_id", None)
                                billed_to_type = str(getattr(r, "billed_to_type", "") or "patient").lower()
                                clinic_id = getattr(r, "billed_to_company_id", None)

                                if bp_id:
                                    key = f"bp:{int(bp_id)}"
                                    dest = {"billing_party_id": int(bp_id), "clinic_company_id": None}
                                elif billed_to_type != "patient" and clinic_id:
                                    key = f"clinic:{int(clinic_id)}"
                                    dest = {"billing_party_id": None, "clinic_company_id": int(clinic_id)}
                                else:
                                    key = "patient"
                                    dest = {"billing_party_id": None, "clinic_company_id": None}

                                if key not in groups:
                                    groups[key] = {"reservation_ids": [], **dest}
                                groups[key]["reservation_ids"].append(int(r.id))

                            for key, g in groups.items():
                                # ✅ S2: Validation du mapping clinique → billing_party (prérequis pour S2)
                                clinic_company_id = g.get("clinic_company_id")
                                if clinic_company_id:
                                    from models.enums import BillingReviewStatus
                                    from services.billing.billing_party_linker import (
                                        resolve_billing_party_for_clinic,
                                    )

                                    # Vérifier que le mapping existe
                                    billing_party = resolve_billing_party_for_clinic(
                                        company_id=company.id,
                                        clinic_company_id=clinic_company_id,
                                    )
                                    if not billing_party:
                                        # Mapping manquant : mettre les bookings en NEEDS_REVIEW
                                        booking_ids = g.get("reservation_ids", [])
                                        reason = (
                                            f"Mapping clinique → billing_party manquant pour "
                                            f"clinic_company_id={clinic_company_id}. "
                                            f"Veuillez configurer le mapping dans les paramètres de facturation."
                                        )
                                        for booking_id in booking_ids:
                                            booking = Booking.query.get(booking_id)
                                            if booking:
                                                try:
                                                    booking.billing_review_status = (
                                                        BillingReviewStatus.NEEDS_REVIEW
                                                    )
                                                    booking.billing_override_reason = reason
                                                except Exception:
                                                    pass
                                        app_logger.warning(
                                            (
                                                "Génération S2 refusée pour clinic_company_id=%s "
                                                "(mapping manquant). %s bookings mis en NEEDS_REVIEW."
                                            ),
                                            clinic_company_id,
                                            len(booking_ids),
                                        )
                                        db.session.commit()
                                        continue  # Passer au groupe suivant

                                # Empêcher les doublons: une facture par (client, période, destinataire)
                                existing_invoice_q = db.session.query(Invoice).filter(
                                    Invoice.company_id == company.id,
                                    Invoice.client_id == client.id,
                                    Invoice.period_year == period_year,
                                    Invoice.period_month == period_month,
                                )
                                if g.get("billing_party_id"):
                                    existing_invoice_q = existing_invoice_q.filter(
                                        Invoice.billing_party_id == int(g["billing_party_id"])
                                    )
                                elif g.get("clinic_company_id"):
                                    existing_invoice_q = existing_invoice_q.filter(
                                        Invoice.billed_to_company_id == int(g["clinic_company_id"])
                                    )
                                else:
                                    existing_invoice_q = existing_invoice_q.filter(
                                        Invoice.billing_party_id.is_(None),
                                        Invoice.bill_to_client_id.is_(None),
                                        Invoice.billed_to_company_id.is_(None),
                                    )

                                if existing_invoice_q.first():
                                    continue

                                generate_input = GenerateInvoiceInput(
                                    company_id=company.id,
                                    client_id=client.id,
                                    period_year=period_year,
                                    period_month=period_month,
                                    billing_party_id=g.get("billing_party_id"),
                                    clinic_company_id=g.get("clinic_company_id"),
                                    reservation_ids=g.get("reservation_ids") or None,
                                )
                                generate_result = generate_invoice_uc.execute(generate_input)
                                if generate_result.success and generate_result.invoice:
                                    invoice = generate_result.invoice
                                    invoices_generated += 1
                                    app_logger.info(
                                        "Facture générée: %s (dest=%s) pour client %s",
                                        invoice.invoice_number,
                                        key,
                                        client.id,
                                    )
                                elif not generate_result.success:
                                    # Si la génération a échoué (ex: mapping manquant), les bookings
                                    # ont déjà été mis en NEEDS_REVIEW par generate_invoice.py
                                    app_logger.warning(
                                        (
                                            "Génération de facture échouée pour client %s, "
                                            "dest=%s: %s"
                                        ),
                                        client.id,
                                        key,
                                        generate_result.error,
                                    )

                    except Exception as e:
                        app_logger.error(
                            (
                                "Erreur lors de la génération de facture "
                                "pour client %s: %s"
                            ),
                            client.id,
                            str(e),
                        )
                        continue

            except Exception as e:
                app_logger.error(
                    "Erreur lors du traitement de l'entreprise %s: %s",
                    company.id,
                    str(e),
                )
                continue

        app_logger.info(
            "Génération mensuelle terminée: %s factures générées", invoices_generated
        )

    except Exception as e:
        app_logger.error(
            "Erreur lors de la génération mensuelle des factures: %s", str(e)
        )
        raise


@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=240,
    task_soft_time_limit=180,
    max_retries=1,
    autoretry_for=(Exception,),
)
def cleanup_old_invoices():
    """Tâche de nettoyage pour archiver les anciennes factures."""
    try:
        app_logger.info("Début du nettoyage des anciennes factures")

        # Factures payées depuis plus de 7 ans
        cutoff_date = datetime.now(UTC) - timedelta(days=7 * 365)

        old_paid_invoices = (
            db.session.query(Invoice)
            .filter(
                Invoice.status == InvoiceStatus.PAID,
                Invoice.paid_at < cutoff_date,
            )
            .all()
        )

        archived_count = 0

        for invoice in old_paid_invoices:
            try:
                # Marquer comme archivé (ajouter un champ archived_at si nécessaire)
                # Pour l'instant, on peut juste logger
                app_logger.info(
                    "Facture %s éligible pour archivage", invoice.invoice_number
                )
                archived_count += 1

            except Exception as e:
                app_logger.error(
                    "Erreur lors de l'archivage de la facture %s: %s",
                    invoice.id,
                    str(e),
                )
                continue

        app_logger.info(
            "Nettoyage terminé: %s factures éligibles pour archivage", archived_count
        )

    except Exception as e:
        app_logger.error("Erreur lors du nettoyage des factures: %s", str(e))
        raise


@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=240,
    task_soft_time_limit=180,
    max_retries=2,
    autoretry_for=(Exception,),
)
def send_invoice_summary() -> None:
    """Tâche pour envoyer un résumé mensuel des factures aux entreprises."""
    try:
        app_logger.info("Début de l'envoi des résumés mensuels")

        # Note: NotificationService n'existe pas dans notification_service
        # Le code attend un objet avec des méthodes send_reminder_notification
        # et send_monthly_invoice_summary
        notification_service: Any = (
            NotificationService() if NotificationService else None
        )
        if notification_service is None:
            app_logger.warning(
                "NotificationService non disponible, résumés non envoyés"
            )
            return

        # Calculer la période précédente
        now = datetime.now(UTC)
        if now.month == MONTH_ONE:
            period_year = now.year - 1
            period_month = MONTH_DECEMBER
        else:
            period_year = now.year
            period_month = now.month - 1

        # Récupérer toutes les entreprises
        companies = db.session.query(Company).all()

        for company in companies:
            try:
                # Calculer les statistiques du mois
                invoices = (
                    db.session.query(Invoice)
                    .filter(
                        Invoice.company_id == company.id,
                        Invoice.period_year == period_year,
                        Invoice.period_month == period_month,
                    )
                    .all()
                )

                if not invoices:
                    continue

                # Calculer les totaux (exclure les factures annulées)
                total_issued = sum(
                    invoice.total_amount
                    for invoice in invoices
                    if invoice.status != InvoiceStatus.CANCELLED
                )
                total_paid = sum(invoice.amount_paid for invoice in invoices)
                total_balance = sum(invoice.balance_due for invoice in invoices)
                overdue_count = len(
                    [inv for inv in invoices if inv.status == InvoiceStatus.OVERDUE]
                )

                # Envoyer le résumé
                notification_service.send_monthly_invoice_summary(
                    company,
                    period_year,
                    period_month,
                    {
                        "total_invoices": len(invoices),
                        "total_issued": total_issued,
                        "total_paid": total_paid,
                        "total_balance": total_balance,
                        "overdue_count": overdue_count,
                    },
                )

                app_logger.info("Résumé mensuel envoyé à l'entreprise %s", company.id)

            except Exception as e:
                app_logger.error(
                    "Erreur lors de l'envoi du résumé pour l'entreprise %s: %s",
                    company.id,
                    str(e),
                )
                continue

        app_logger.info("Envoi des résumés mensuels terminé")

    except Exception as e:
        app_logger.error("Erreur lors de l'envoi des résumés mensuels: %s", str(e))
        raise


# Configuration des tâches périodiques

celery_app.conf.beat_schedule = {
    "check-overdues-daily": {
        "task": "billing_tasks.check_overdues_and_trigger_reminders",
        "schedule": crontab(hour="8", minute="0"),  # Tous les jours à 8h00
    },
    "send-reminder-notifications": {
        "task": "billing_tasks.send_reminder_notifications",
        "schedule": crontab(hour="9", minute="0"),  # Tous les jours à 9h00
    },
    "generate-monthly-invoices": {
        "task": "billing_tasks.generate_monthly_invoices",
        # Le 1er de chaque mois à 6h00
        "schedule": crontab(day_of_month="1", hour="6", minute="0"),
    },
    "send-invoice-summary": {
        "task": "billing_tasks.send_invoice_summary",
        # Le 2 de chaque mois à 10h00
        "schedule": crontab(day_of_month="2", hour="10", minute="0"),
    },
    "cleanup-old-invoices": {
        "task": "billing_tasks.cleanup_old_invoices",
        # Le 1er de chaque mois à 2h00
        "schedule": crontab(day_of_month="1", hour="2", minute="0"),
    },
}
