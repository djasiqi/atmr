"""
Routes API pour les paramètres avancés de l'entreprise
"""
from flask import request
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required
from models import Company, CompanyBillingSettings, CompanyPlanningSettings, UserRole
from ext import db, role_required
from routes.companies import get_company_from_token
import logging

logger = logging.getLogger(__name__)

settings_ns = Namespace('company-settings', description='Paramètres avancés entreprise')

# ==================== Models API ====================

billing_settings_model = settings_ns.model('BillingSettings', {
    'id': fields.Integer,
    'company_id': fields.Integer,
    'payment_terms_days': fields.Integer(description='Délai de paiement en jours'),
    'overdue_fee': fields.Float(description='Frais de retard'),
    'reminder1_fee': fields.Float(description='Frais 1er rappel'),
    'reminder2_fee': fields.Float(description='Frais 2e rappel'),
    'reminder3_fee': fields.Float(description='Frais 3e rappel'),
    'reminder_schedule_days': fields.Raw(description='Planning des rappels'),
    'auto_reminders_enabled': fields.Boolean(description='Rappels automatiques activés'),
    'email_sender': fields.String(description='Email expéditeur'),
    'invoice_number_format': fields.String(description='Format de numérotation'),
    'invoice_prefix': fields.String(description='Préfixe des factures'),
    'iban': fields.String(description='IBAN'),
    'qr_iban': fields.String(description='QR-IBAN'),
    'esr_ref_base': fields.String(description='Référence ESR'),
    'invoice_message_template': fields.String(description='Template email facture'),
    'reminder1_template': fields.String(description='Template 1er rappel'),
    'reminder2_template': fields.String(description='Template 2e rappel'),
    'reminder3_template': fields.String(description='Template 3e rappel'),
    'legal_footer': fields.String(description='Pied de page légal'),
    'pdf_template_variant': fields.String(description='Variante template PDF'),
})

operational_settings_model = settings_ns.model('OperationalSettings', {
    'service_area': fields.String(description='Zone de service'),
    'max_daily_bookings': fields.Integer(description='Limite courses/jour'),
    'dispatch_enabled': fields.Boolean(description='Dispatch automatique activé'),
    'latitude': fields.Float(description='Latitude du siège'),
    'longitude': fields.Float(description='Longitude du siège'),
})

# ==================== Routes ====================

@settings_ns.route('/operational')
class OperationalSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupérer les paramètres opérationnels"""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code
        
        return {
            "success": True,
            "data": {
                "service_area": company.service_area,
                "max_daily_bookings": company.max_daily_bookings,
                "dispatch_enabled": company.dispatch_enabled,
                "latitude": company.latitude,
                "longitude": company.longitude,
            }
        }, 200
    
    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(operational_settings_model)
    def put(self):
        """Mettre à jour les paramètres opérationnels"""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code
        
        data = request.get_json()
        
        try:
            if 'service_area' in data:
                company.service_area = data['service_area']
            if 'max_daily_bookings' in data:
                company.max_daily_bookings = int(data['max_daily_bookings'])
            if 'dispatch_enabled' in data:
                company.dispatch_enabled = bool(data['dispatch_enabled'])
            if 'latitude' in data:
                company.latitude = float(data['latitude']) if data['latitude'] else None
            if 'longitude' in data:
                company.longitude = float(data['longitude']) if data['longitude'] else None
            
            db.session.commit()
            logger.info(f"[Settings] Operational settings updated for company {company.id}")
            
            return {
                "success": True,
                "message": "Paramètres opérationnels mis à jour",
                "data": {
                    "service_area": company.service_area,
                    "max_daily_bookings": company.max_daily_bookings,
                    "dispatch_enabled": company.dispatch_enabled,
                    "latitude": company.latitude,
                    "longitude": company.longitude,
                }
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error(f"[Settings] Error updating operational settings: {e}")
            return {"success": False, "error": str(e)}, 500


@settings_ns.route('/billing')
class BillingSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.marshal_with(billing_settings_model)
    def get(self):
        """Récupérer les paramètres de facturation"""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code
        
        # Récupérer ou créer les billing settings
        billing = CompanyBillingSettings.query.filter_by(company_id=company.id).first()
        
        if not billing:
            # Créer avec valeurs par défaut
            billing = CompanyBillingSettings(
                company_id=company.id,
                payment_terms_days=10,
                overdue_fee=15.00,
                reminder1_fee=0.00,
                reminder2_fee=40.00,
                reminder3_fee=0.00,
                reminder_schedule_days={"1": 10, "2": 5, "3": 5},
                auto_reminders_enabled=True,
                invoice_number_format="{PREFIX}-{YYYY}-{MM}-{SEQ4}",
                invoice_prefix="EM",
                pdf_template_variant="default"
            )
            db.session.add(billing)
            db.session.commit()
        
        return billing.to_dict(), 200
    
    @jwt_required()
    @role_required(UserRole.company)
    @settings_ns.expect(billing_settings_model)
    def put(self):
        """Mettre à jour les paramètres de facturation"""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code
        
        data = request.get_json()
        
        try:
            billing = CompanyBillingSettings.query.filter_by(company_id=company.id).first()
            
            if not billing:
                billing = CompanyBillingSettings(company_id=company.id)
                db.session.add(billing)
            
            # Mise à jour des champs
            updatable_fields = [
                'payment_terms_days', 'overdue_fee', 'reminder1_fee', 'reminder2_fee',
                'reminder3_fee', 'reminder_schedule_days', 'auto_reminders_enabled',
                'email_sender', 'invoice_number_format', 'invoice_prefix',
                'iban', 'qr_iban', 'esr_ref_base',
                'invoice_message_template', 'reminder1_template',
                'reminder2_template', 'reminder3_template',
                'legal_footer', 'pdf_template_variant'
            ]
            
            for field in updatable_fields:
                if field in data:
                    setattr(billing, field, data[field])
            
            db.session.commit()
            logger.info(f"[Settings] Billing settings updated for company {company.id}")
            
            return {
                "success": True,
                "message": "Paramètres de facturation mis à jour",
                "data": billing.to_dict()
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error(f"[Settings] Error updating billing settings: {e}")
            return {"success": False, "error": str(e)}, 500


@settings_ns.route('/planning')
class PlanningSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupérer les paramètres de planning"""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code
        
        planning = CompanyPlanningSettings.query.filter_by(company_id=company.id).first()
        
        if not planning:
            planning = CompanyPlanningSettings(
                company_id=company.id,
                settings={}
            )
            db.session.add(planning)
            db.session.commit()
        
        return {
            "success": True,
            "data": planning.settings
        }, 200
    
    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        """Mettre à jour les paramètres de planning"""
        company, err, code = get_company_from_token()
        if err:
            return {"success": False, "error": err}, code
        
        data = request.get_json()
        
        try:
            planning = CompanyPlanningSettings.query.filter_by(company_id=company.id).first()
            
            if not planning:
                planning = CompanyPlanningSettings(
                    company_id=company.id,
                    settings=data.get('settings', {})
                )
                db.session.add(planning)
            else:
                planning.settings = data.get('settings', {})
            
            db.session.commit()
            logger.info(f"[Settings] Planning settings updated for company {company.id}")
            
            return {
                "success": True,
                "message": "Paramètres de planning mis à jour",
                "data": planning.settings
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error(f"[Settings] Error updating planning settings: {e}")
            return {"success": False, "error": str(e)}, 500

