import logging
from datetime import UTC, datetime, timedelta

from celery import Celery

from models import Invoice, InvoiceStatus, db
from services.invoice_service import InvoiceService
from services.notification_service import NotificationService

# Configuration du logger
app_logger = logging.getLogger("billing_tasks")

# Instance Celery (à adapter selon votre configuration)
celery_app = Celery('billing')

@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=300,
    max_retries=2,
    autoretry_for=(Exception,)
)
def check_overdues_and_trigger_reminders():
    """
    Tâche quotidienne pour vérifier les factures en retard et déclencher les rappels automatiques
    """
    try:
        app_logger.info("Début de la vérification des factures en retard")

        # Initialiser le service de facturation
        invoice_service = InvoiceService()

        # Vérifier et marquer les factures en retard
        invoice_service.check_overdue_invoices()

        # Traiter les rappels automatiques
        invoice_service.process_automatic_reminders()

        app_logger.info("Vérification des factures en retard terminée avec succès")

    except Exception as e:
        app_logger.error(f"Erreur lors de la vérification des factures en retard: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=600,
    max_retries=2,
    autoretry_for=(Exception,)
)
def send_reminder_notifications():
    """
    Tâche pour envoyer les notifications de rappel par email
    """
    try:
        app_logger.info("Début de l'envoi des notifications de rappel")

        notification_service = NotificationService()

        # Récupérer les rappels générés aujourd'hui qui n'ont pas encore été envoyés
        today = datetime.now(UTC).date()
        reminders_to_send = db.session.query(InvoiceReminder).filter(
            InvoiceReminder.generated_at >= today,
            InvoiceReminder.sent_at.is_(None),
            InvoiceReminder.pdf_url.isnot(None)
        ).all()

        for reminder in reminders_to_send:
            try:
                # Envoyer la notification
                notification_service.send_reminder_notification(reminder)

                # Marquer comme envoyé
                reminder.sent_at = datetime.now(UTC)
                db.session.commit()

                app_logger.info(f"Notification de rappel envoyée pour la facture {reminder.invoice.invoice_number}")

            except Exception as e:
                app_logger.error(f"Erreur lors de l'envoi de la notification pour le rappel {reminder.id}: {str(e)}")
                db.session.rollback()

        app_logger.info(f"Envoi des notifications terminé: {len(reminders_to_send)} rappels traités")

    except Exception as e:
        app_logger.error(f"Erreur lors de l'envoi des notifications de rappel: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=900,  # 15 min (peut générer beaucoup de factures)
    max_retries=1,
    autoretry_for=(Exception,)
)
def generate_monthly_invoices():
    """
    Tâche mensuelle pour générer automatiquement les factures des clients actifs
    """
    try:
        app_logger.info("Début de la génération mensuelle des factures")

        invoice_service = InvoiceService()

        # Calculer la période précédente
        now = datetime.now(UTC)
        if now.month == 1:
            period_year = now.year - 1
            period_month = 12
        else:
            period_year = now.year
            period_month = now.month - 1

        # Récupérer toutes les entreprises avec des clients actifs
        companies_with_clients = db.session.query(Company).join(Client).filter(
            Client.is_active == True
        ).distinct().all()

        invoices_generated = 0

        for company in companies_with_clients:
            try:
                # Récupérer les clients actifs de cette entreprise
                active_clients = db.session.query(Client).filter(
                    Client.company_id == company.id,
                    Client.is_active == True
                ).all()

                for client in active_clients:
                    try:
                        # Vérifier qu'une facture n'existe pas déjà pour cette période
                        existing_invoice = db.session.query(Invoice).filter(
                            Invoice.company_id == company.id,
                            Invoice.client_id == client.id,
                            Invoice.period_year == period_year,
                            Invoice.period_month == period_month
                        ).first()

                        if existing_invoice:
                            continue

                        # Vérifier qu'il y a des réservations pour cette période
                        reservations = invoice_service._get_reservations_for_period(
                            company.id, client.id, period_year, period_month
                        )

                        if reservations:
                            # Générer la facture
                            invoice = invoice_service.generate_invoice(
                                company.id, client.id, period_year, period_month
                            )
                            invoices_generated += 1

                            app_logger.info(f"Facture générée: {invoice.invoice_number} pour client {client.id}")

                    except Exception as e:
                        app_logger.error(f"Erreur lors de la génération de facture pour client {client.id}: {str(e)}")
                        continue

            except Exception as e:
                app_logger.error(f"Erreur lors du traitement de l'entreprise {company.id}: {str(e)}")
                continue

        app_logger.info(f"Génération mensuelle terminée: {invoices_generated} factures générées")

    except Exception as e:
        app_logger.error(f"Erreur lors de la génération mensuelle des factures: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=600,
    max_retries=1,
    autoretry_for=(Exception,)
)
def cleanup_old_invoices():
    """
    Tâche de nettoyage pour archiver les anciennes factures
    """
    try:
        app_logger.info("Début du nettoyage des anciennes factures")

        # Factures payées depuis plus de 7 ans
        cutoff_date = datetime.now(UTC) - timedelta(days=7*365)

        old_paid_invoices = db.session.query(Invoice).filter(
            Invoice.status == InvoiceStatus.PAID,
            Invoice.paid_at < cutoff_date
        ).all()

        archived_count = 0

        for invoice in old_paid_invoices:
            try:
                # Marquer comme archivé (ajouter un champ archived_at si nécessaire)
                # Pour l'instant, on peut juste logger
                app_logger.info(f"Facture {invoice.invoice_number} éligible pour archivage")
                archived_count += 1

            except Exception as e:
                app_logger.error(f"Erreur lors de l'archivage de la facture {invoice.id}: {str(e)}")
                continue

        app_logger.info(f"Nettoyage terminé: {archived_count} factures éligibles pour archivage")

    except Exception as e:
        app_logger.error(f"Erreur lors du nettoyage des factures: {str(e)}")
        raise

@celery_app.task(
    bind=True,
    acks_late=True,
    task_time_limit=300,
    max_retries=2,
    autoretry_for=(Exception,)
)
def send_invoice_summary():
    """
    Tâche pour envoyer un résumé mensuel des factures aux entreprises
    """
    try:
        app_logger.info("Début de l'envoi des résumés mensuels")

        notification_service = NotificationService()

        # Calculer la période précédente
        now = datetime.now(UTC)
        if now.month == 1:
            period_year = now.year - 1
            period_month = 12
        else:
            period_year = now.year
            period_month = now.month - 1

        # Récupérer toutes les entreprises
        companies = db.session.query(Company).all()

        for company in companies:
            try:
                # Calculer les statistiques du mois
                invoices = db.session.query(Invoice).filter(
                    Invoice.company_id == company.id,
                    Invoice.period_year == period_year,
                    Invoice.period_month == period_month
                ).all()

                if not invoices:
                    continue

                # Calculer les totaux
                total_issued = sum(invoice.total_amount for invoice in invoices)
                total_paid = sum(invoice.amount_paid for invoice in invoices)
                total_balance = sum(invoice.balance_due for invoice in invoices)
                overdue_count = len([inv for inv in invoices if inv.status == InvoiceStatus.OVERDUE])

                # Envoyer le résumé
                notification_service.send_monthly_invoice_summary(
                    company, period_year, period_month, {
                        'total_invoices': len(invoices),
                        'total_issued': total_issued,
                        'total_paid': total_paid,
                        'total_balance': total_balance,
                        'overdue_count': overdue_count
                    }
                )

                app_logger.info(f"Résumé mensuel envoyé à l'entreprise {company.id}")

            except Exception as e:
                app_logger.error(f"Erreur lors de l'envoi du résumé pour l'entreprise {company.id}: {str(e)}")
                continue

        app_logger.info("Envoi des résumés mensuels terminé")

    except Exception as e:
        app_logger.error(f"Erreur lors de l'envoi des résumés mensuels: {str(e)}")
        raise

# Configuration des tâches périodiques
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'check-overdues-daily': {
        'task': 'billing_tasks.check_overdues_and_trigger_reminders',
        'schedule': crontab(hour=8, minute=0),  # Tous les jours à 8h00
    },
    'send-reminder-notifications': {
        'task': 'billing_tasks.send_reminder_notifications',
        'schedule': crontab(hour=9, minute=0),  # Tous les jours à 9h00
    },
    'generate-monthly-invoices': {
        'task': 'billing_tasks.generate_monthly_invoices',
        'schedule': crontab(day_of_month=1, hour=6, minute=0),  # Le 1er de chaque mois à 6h00
    },
    'send-invoice-summary': {
        'task': 'billing_tasks.send_invoice_summary',
        'schedule': crontab(day_of_month=2, hour=10, minute=0),  # Le 2 de chaque mois à 10h00
    },
    'cleanup-old-invoices': {
        'task': 'billing_tasks.cleanup_old_invoices',
        'schedule': crontab(day_of_month=1, hour=2, minute=0),  # Le 1er de chaque mois à 2h00
    },
}
