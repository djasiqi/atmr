"""recalculate_reminder_due_date_from_schedule

Revision ID: d7e4a1b92f03
Revises: c5847c06ae2d
Create Date: 2026-06-17 12:00:00.000000

Recalcule invoice_reminders.due_date selon reminder_schedule_days par niveau.
"""

from alembic import op


revision = "d7e4a1b92f03"
down_revision = "c5847c06ae2d"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        UPDATE invoice_reminders AS r
        SET due_date = r.generated_at + (
            COALESCE(
                NULLIF(cbs.reminder_schedule_days ->> r.level::text, '')::int,
                CASE r.level
                    WHEN 1 THEN 10
                    WHEN 2 THEN 5
                    WHEN 3 THEN 5
                    ELSE 10
                END
            ) || ' days'
        )::interval
        FROM invoices AS i
        LEFT JOIN company_billing_settings AS cbs ON cbs.company_id = i.company_id
        WHERE r.invoice_id = i.id
          AND r.generated_at IS NOT NULL
        """
    )


def downgrade():
    pass
