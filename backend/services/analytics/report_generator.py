# backend/services/analytics/report_generator.py

# Constantes pour éviter les valeurs magiques
import logging
from datetime import date, timedelta
from typing import Any, Dict, cast

QUALITY_THRESHOLD = 85
ON_TIME_RATE_THRESHOLD = 70

"""Service de génération de rapports automatiques.
Génère des rapports PDF/Email pour les analytics.
"""


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Générateur de rapports analytics."""

    def generate_daily_report(self, company_id: int, day: date) -> Dict[str, Any]:
        """Génère un rapport quotidien.

        Args:
            company_id: ID de l'entreprise
            day: Date du rapport

        Returns:
            Dict avec le contenu du rapport

        """
        from models import Company
        from services.analytics.aggregator import get_period_analytics
        from services.analytics.insights import generate_insights

        company = Company.query.get(company_id)
        if not company:
            return {"error": "Company not found"}

        # Récupérer les analytics du jour
        analytics = get_period_analytics(company_id, day, day)

        if not analytics or not analytics.get("trends"):
            return {"error": "No data for this day"}

        daily_data = analytics["trends"][0] if analytics["trends"] else {}
        insights = generate_insights(company_id, analytics)

        # Générer le contenu
        return {
            "company_name": company.name,
            "date": day.isoformat(),
            "metrics": {
                "total_bookings": daily_data.get("bookings", 0),
                "on_time_rate": daily_data.get("on_time_rate", 0),
                "avg_delay": daily_data.get("avg_delay", 0),
                "quality_score": daily_data.get("quality_score", 0),
            },
            "insights": insights,
            "summary": self._generate_daily_summary(daily_data),
        }

    def generate_weekly_report(
        self, company_id: int, week_start: date
    ) -> Dict[str, Any]:
        """Génère un rapport hebdomadaire.

        Args:
            company_id: ID de l'entreprise
            week_start: Date de début de semaine (lundi)

        Returns:
            Dict avec le contenu du rapport

        """
        from models import Company
        from services.analytics.aggregator import get_weekly_summary
        from services.analytics.insights import generate_insights

        company = Company.query.get(company_id)
        if not company:
            return {"error": "Company not found"}

        # Récupérer le résumé hebdomadaire
        analytics = get_weekly_summary(company_id, week_start)

        if not analytics or not analytics.get("trends"):
            return {"error": "No data for this week"}

        insights = generate_insights(company_id, analytics)

        # Calculer les highlights
        week_end = week_start + timedelta(days=6)
        summary = analytics.get("summary", {})

        return {
            "company_name": company.name,
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "summary": {
                "total_bookings": summary.get("total_bookings", 0),
                "avg_on_time_rate": summary.get("avg_on_time_rate", 0),
                "avg_delay": summary.get("avg_delay_minutes", 0),
                "avg_quality": summary.get("avg_quality_score", 0),
            },
            "insights": analytics.get("insights", {}),
            "trends": analytics.get("trends", []),
            "recommendations": self._generate_weekly_recommendations(
                analytics, insights
            ),
        }

    def _generate_daily_summary(self, daily_data: Dict[str, Any]) -> str:
        """Génère un résumé texte du jour."""
        bookings = daily_data.get("bookings", 0)
        on_time_rate = daily_data.get("on_time_rate", 0)
        quality = daily_data.get("quality_score", 0)

        if quality >= QUALITY_THRESHOLD:
            performance = "excellente"
            emoji = "🎉"
        elif quality >= QUALITY_THRESHOLD:
            performance = "bonne"
            emoji = "✅"
        elif quality >= QUALITY_THRESHOLD:
            performance = "moyenne"
            emoji = "⚠️"
        else:
            performance = "faible"
            emoji = "🔴"

        return (
            f"{emoji} Journée {performance} avec {bookings} courses effectuées. "
            f"Taux de ponctualité : {on_time_rate:.1f}%. "
            f"(Score qualité : {quality:.0f})"
        )

    def _generate_weekly_recommendations(
        self, analytics: Dict[str, Any], insights: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Génère des recommandations hebdomadaires."""
        recommendations = []

        summary = analytics.get("summary", {})
        on_time_rate = summary.get("avg_on_time_rate", 0)

        # Recommandation sur la ponctualité
        if on_time_rate < ON_TIME_RATE_THRESHOLD:
            recommendations.append(
                {
                    "priority": "high",
                    "title": "Améliorer la ponctualité",
                    "description": (
                        "Le taux de ponctualité est faible. "
                        "Analysez les causes récurrentes de retards."
                    ),
                }
            )

        # Recommandations des insights
        for insight in insights:
            if (
                insight.get("priority") == "critical"
                or insight.get("priority") == "high"
            ):
                recommendations.append(
                    cast(
                        Dict[str, str],
                        {
                            "priority": insight.get("priority") or "",
                            "title": insight.get("title") or "",
                            "description": insight.get("message") or "",
                        },
                    )
                )

        if not recommendations:
            recommendations.append(
                {
                    "priority": "low",
                    "title": "Maintenir la performance",
                    "description": (
                        "Continuez sur cette lancée ! Aucune action urgente requise."
                    ),
                }
            )

        return recommendations

    def generate_email_content(
        self, report: Dict[str, Any], report_type: str = "daily"
    ) -> Dict[str, str]:
        """Génère le contenu d'un email à partir d'un rapport.

        Args:
            report: Données du rapport
            report_type: "daily" ou "weekly"

        Returns:
            Dict avec subject et body

        """
        if report_type == "daily":
            return self._generate_daily_email(report)
        return self._generate_weekly_email(report)

    def _generate_daily_email(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Génère l'email quotidien."""
        company_name = report.get("company_name", "")
        date = report.get("date", "")
        metrics = report.get("metrics", {})
        summary = report.get("summary", "")

        subject = f"📊 Rapport Quotidien - {date}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <h2 style="color: #0f766e;">Rapport Quotidien - {company_name}</h2>
            <p><strong>Date :</strong> {date}</p>

            <div style="background: #f0fdfa; padding: 15px; border-radius: 8px; "
                 "margin: 20px 0;">
                <h3 style="color: #0f766e; margin-top: 0;">Résumé</h3>
                <p>{summary}</p>
            </div>

            <h3>Métriques Clés</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        <strong>Total Courses</strong>
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        {metrics.get("total_bookings", 0)}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        <strong>Taux de ponctualité</strong>
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        {metrics.get("on_time_rate", 0):.1f}%
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        <strong>Retard moyen</strong>
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        {metrics.get("avg_delay", 0):.1f} min
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px;">
                        <strong>Score de qualité</strong>
                    </td>
                    <td style="padding: 10px;">
                        {metrics.get("quality_score", 0):.0f}/100
                    </td>
                </tr>
            </table>

            <p style="margin-top: 30px; color: #666; font-size: 12px;">
                Ce rapport est généré automatiquement par le système ATMR Analytics.
            </p>
        </body>
        </html>
        """

        return {"subject": subject, "body": body}

    def _generate_weekly_email(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Génère l'email hebdomadaire."""
        company_name = report.get("company_name", "")
        week_start = report.get("week_start", "")
        week_end = report.get("week_end", "")
        summary = report.get("summary", {})
        recommendations = report.get("recommendations", [])

        subject = f"📊 Rapport Hebdomadaire - Semaine du {week_start}"

        # Générer la liste des recommandations
        reco_html = ""
        for reco in recommendations[:5]:  # Limiter à 5
            priority = reco.get("priority", "low")
            emoji = (
                "🔴" if priority == "critical" else "⚠️" if priority == "high" else "ℹ️"
            )
            reco_html += f"""
            <li style="margin-bottom: 10px;">
                {emoji} <strong>{reco.get("title", "")}</strong><br>
                <span style="color: #666; font-size: 14px;">"
                    "{reco.get("description", "")}</span>
            </li>
            """

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <h2 style="color: #0f766e;">Rapport Hebdomadaire - {company_name}</h2>
            <p><strong>Période :</strong> {week_start} au {week_end}</p>

            <h3>Résumé de la Semaine</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        <strong>Total Courses</strong>
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        {summary.get("total_bookings", 0)}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        <strong>Taux de ponctualité moyen</strong>
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        {summary.get("avg_on_time_rate", 0):.1f}%
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        <strong>Retard moyen</strong>
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #e5e5e5;">
                        {summary.get("avg_delay", 0):.1f} min
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px;">
                        <strong>Score de qualité moyen</strong>
                    </td>
                    <td style="padding: 10px;">
                        {summary.get("avg_quality", 0):.0f}/100
                    </td>
                </tr>
            </table>

            <h3 style="margin-top: 30px;">Recommandations</h3>
            <ul style="list-style: none; padding: 0;">
                {reco_html}
            </ul>

            <p style="margin-top: 30px; color: #666; font-size: 12px;">
                Ce rapport est généré automatiquement par le système ATMR Analytics.
            </p>
        </body>
        </html>
        """

        return {"subject": subject, "body": body}


# Instance globale
_report_generator = ReportGenerator()


def generate_daily_report(company_id: int, day: date) -> Dict[str, Any]:
    """Helper function."""
    return _report_generator.generate_daily_report(company_id, day)


def generate_weekly_report(company_id: int, week_start: date) -> Dict[str, Any]:
    """Helper function."""
    return _report_generator.generate_weekly_report(company_id, week_start)


def generate_email_content(
    report: Dict[str, Any], report_type: str = "daily"
) -> Dict[str, str]:
    """Helper function."""
    return _report_generator.generate_email_content(report, report_type)
