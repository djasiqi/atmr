"""Utilitaires pour l'injection de signature email dans les emails de facturation."""

import html
import logging
import os
import re
from pathlib import Path
from typing import Any

from flask import current_app
from jinja2 import select_autoescape
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)


def _get_logo_file_path(logo_url: str | None) -> Path | None:
    """Récupère le chemin du fichier logo depuis logo_url.

    Si logo_url est une URL externe (http/https), retourne None.
    Si logo_url est relative (commence par /uploads/), retourne le Path du fichier.

    Args:
        logo_url: URL du logo (relative ou absolue)

    Returns:
        Path du fichier logo ou None si URL externe ou invalide
    """
    if not logo_url or not logo_url.strip():
        return None

    logo_url = logo_url.strip()

    # Si URL externe, on ne peut pas récupérer le fichier
    if logo_url.startswith(("http://", "https://")):
        return None

    # Si relative, construire le chemin
    if logo_url.startswith("/"):
        try:
            from flask import current_app

            uploads_dir = Path(current_app.config.get("UPLOAD_FOLDER", "/app/uploads"))
            # Nettoyer le chemin : /uploads/company_logos/logo.png -> company_logos/logo.png
            logo_url_clean = logo_url.lstrip("/")
            if logo_url_clean.startswith("uploads/"):
                logo_url_clean = logo_url_clean[8:]  # Supprimer 'uploads/'

            logo_path = uploads_dir / logo_url_clean
            if logo_path.exists():
                return logo_path
        except RuntimeError:
            # Hors contexte Flask (tests)
            return None

    return None


def _get_logo_bytes(logo_url: str | None) -> tuple[bytes | None, str | None]:
    """Récupère le logo en bytes depuis logo_url.

    Args:
        logo_url: URL du logo (relative ou absolue)

    Returns:
        Tuple (logo_bytes, mime_type) ou (None, None) si impossible
    """
    logo_path = _get_logo_file_path(logo_url)
    if not logo_path or not logo_path.exists():
        return (None, None)

    try:
        logo_bytes = logo_path.read_bytes()
        if len(logo_bytes) == 0:
            logger.warning("Logo %s est vide (0 bytes)", logo_path)
            return (None, None)
        # Déterminer le MIME type depuis l'extension (strict pour Outlook)
        suffix = logo_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
        mime_type = mime_types.get(suffix, "image/png")
        EMAIL_SIGNATURE_DEBUG = os.getenv("EMAIL_SIGNATURE_DEBUG", "0") == "1"
        if EMAIL_SIGNATURE_DEBUG:
            logger.info(
                "[EMAIL_SIGNATURE_DEBUG] _get_logo_bytes: path=%s, suffix=%s, mime_type=%s, bytes_len=%d",
                logo_path,
                suffix,
                mime_type,
                len(logo_bytes),
            )
        return (logo_bytes, mime_type)
    except Exception as e:
        logger.warning("Erreur lecture logo %s: %s", logo_path, e)
        return (None, None)


def _make_logo_url_absolute(logo_url: str | None) -> str | None:
    """Convertit une URL relative en URL absolue pour les emails.

    Si logo_url est déjà absolue (http:// ou https://), elle est retournée telle quelle.
    Si logo_url est relative (commence par /), elle est convertie en URL absolue
    en utilisant PUBLIC_BASE_URL ou APP_BASE_URL depuis la config Flask.

    Args:
        logo_url: URL du logo (relative ou absolue)

    Returns:
        URL absolue ou None si logo_url est vide
    """
    if not logo_url or not logo_url.strip():
        return None

    logo_url = logo_url.strip()

    # Si déjà absolue, retourner telle quelle
    if logo_url.startswith(("http://", "https://")):
        return logo_url

    # Si relative, convertir en absolue
    if logo_url.startswith("/"):
        # Utiliser PUBLIC_BASE_URL ou APP_BASE_URL depuis la config
        try:
            base_url = (
                current_app.config.get("PUBLIC_BASE_URL")
                or current_app.config.get("APP_BASE_URL")
                or current_app.config.get("PDF_BASE_URL", "http://127.0.0.1:5000")
            )
            # S'assurer que base_url ne se termine pas par /
            base_url = base_url.rstrip("/")
            return f"{base_url}{logo_url}"
        except RuntimeError:
            # Hors contexte Flask (tests), utiliser une valeur par défaut
            return f"http://127.0.0.1:5000{logo_url}"

    # Si ni absolue ni relative, retourner telle quelle (peut être un chemin relatif sans /)
    return logo_url


def render_signature_html_template(
    template_str: str,
    company: Any,
    billing_settings: Any | None = None,  # noqa: ARG001
) -> str:
    """Render un template HTML de signature avec variables whitelistées.

    Variables disponibles (whitelist):
    - name: Nom de l'entreprise (company.name)
    - contact_name: Nom du contact (optionnel)
    - phone: Téléphone (company.contact_phone)
    - email: Email (company.contact_email ou billing_email)
    - website: Site web (optionnel, depuis company ou settings)
    - address: Adresse complète formatée
    - logo_url: URL du logo (email_signature_logo_url ou company.logo_url)

    Sécurité:
    - SandboxedEnvironment (pas d'imports, pas d'appels arbitraires)
    - Auto-escape activé
    - Suppression des balises <script> et <iframe> après render

    Args:
        template_str: Template Jinja2 (HTML)
        company: Instance Company
        billing_settings: CompanyBillingSettings (optionnel, pour logo_url custom)

    Returns:
        HTML rendu et sécurisé

    Raises:
        ValueError: Si le template contient des expressions non autorisées
    """
    if not template_str or not template_str.strip():
        return ""

    # Préparer les variables whitelistées
    # Logo: utiliser uniquement company.logo_url (plus de signature_logo_url)
    logo_url = None
    if hasattr(company, "logo_url") and company.logo_url:
        logo_url = _make_logo_url_absolute(company.logo_url)

    # Formater l'adresse
    address_parts = []
    if hasattr(company, "domicile_address_line1") and company.domicile_address_line1:
        address_parts.append(company.domicile_address_line1)
    if hasattr(company, "domicile_zip") and hasattr(company, "domicile_city"):
        if company.domicile_zip and company.domicile_city:
            address_parts.append(f"{company.domicile_zip} {company.domicile_city}")
    elif hasattr(company, "address") and company.address:
        address_parts.append(company.address)
    address = "<br>".join(address_parts) if address_parts else ""

    # Variables whitelistées uniquement
    context = {
        "name": company.name if hasattr(company, "name") else "",
        "contact_name": "",  # Optionnel, peut être ajouté plus tard
        "phone": company.contact_phone if hasattr(company, "contact_phone") else "",
        "email": (
            company.billing_email
            if hasattr(company, "billing_email") and company.billing_email
            else (company.contact_email if hasattr(company, "contact_email") else "")
        ),
        "website": "",  # Optionnel, peut être ajouté plus tard
        "address": address,
        "logo_url": logo_url or "",
    }

    # Créer un environnement Jinja2 sandboxé (sécurité)
    env = SandboxedEnvironment(
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=False,
    )
    env.globals = {}  # Pas de globals arbitraires

    try:
        template = env.from_string(template_str.strip())
        rendered = template.render(**context)
    except Exception as e:
        # En cas d'erreur (template invalide, variable non autorisée, etc.)
        # Retourner une chaîne vide plutôt que de planter
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "Erreur lors du render du template signature HTML: %s (template: %s)",
            e,
            template_str[:100],
        )
        return ""

    # Sécurité supplémentaire: supprimer les balises dangereuses
    # (même si auto-escape est activé, on supprime <script> et <iframe> par précaution)
    rendered = re.sub(
        r"<script[^>]*>.*?</script>", "", rendered, flags=re.IGNORECASE | re.DOTALL
    )
    rendered = re.sub(
        r"<iframe[^>]*>.*?</iframe>", "", rendered, flags=re.IGNORECASE | re.DOTALL
    )
    # Supprimer onclick, onload, etc.
    return re.sub(r"on\w+\s*=", "", rendered, flags=re.IGNORECASE)


def generate_signature_html_from_form(
    name: str | None = None,
    title: str | None = None,
    company: str | None = None,
    phone_main: str | None = None,
    phone_mobile: str | None = None,
    email: str | None = None,
    website: str | None = None,
    address_line: str | None = None,
    zip_code: str | None = None,
    city: str | None = None,
    company_obj: Any | None = None,
) -> str:
    """Génère automatiquement le HTML de signature depuis un formulaire (mode "form").

    Crée un layout table-based compatible Outlook avec 2 colonnes + ligne verticale + ligne horizontale + logo.

    Structure:
    - Colonne gauche: nom (bold), titre, société
    - Colonne droite: téléphones, email (mailto), website (link), adresse
    - Ligne verticale entre colonnes (border-left: 2px solid #1b4b7a)
    - Ligne horizontale sous le bloc (border-top: 1px solid #1b4b7a)
    - Logo en bas si fourni (depuis company_obj.logo_url)

    Args:
        name: Nom complet (ex: "Khalid ALAOUI")
        title: Titre (ex: "Associé gérant")
        company: Nom de la société (ex: "Emmenez-moi Sàrl")
        phone_main: Téléphone principal
        phone_mobile: Téléphone mobile
        email: Email (sera dans un lien mailto:)
        website: Site web (sera dans un lien)
        address_line: Ligne d'adresse
        zip_code: Code postal
        city: Ville
        company_obj: Instance Company (pour récupérer logo_url automatiquement)

    Returns:
        HTML de signature généré automatiquement (email safe, tables only)
    """
    # Validation et normalisation des champs
    # Limites de longueur (selon les colonnes DB: String(200), String(50), String(100), String(10), String(500))
    MAX_LENGTH_NAME = 200
    MAX_LENGTH_TITLE = 200
    MAX_LENGTH_COMPANY = 200
    MAX_LENGTH_PHONE = 50
    MAX_LENGTH_EMAIL = 200
    MAX_LENGTH_WEBSITE = 200
    MAX_LENGTH_ADDRESS_LINE = 200
    MAX_LENGTH_ZIP = 10
    MAX_LENGTH_CITY = 100

    def truncate_field(value: str | None, max_length: int) -> str | None:
        """Tronque un champ à la longueur maximale."""
        if not value:
            return None
        value_stripped = value.strip()
        if not value_stripped:
            return None
        return (
            value_stripped[:max_length]
            if len(value_stripped) > max_length
            else value_stripped
        )

    def normalize_email(email_val: str | None) -> str | None:
        """Normalise et valide un email (strip + vérifie présence de @)."""
        if not email_val:
            return None
        email_stripped = email_val.strip()
        if not email_stripped:
            return None
        # Validation minimale: doit contenir @
        if "@" not in email_stripped:
            return None
        # Tronquer si trop long
        return truncate_field(email_stripped, MAX_LENGTH_EMAIL)

    def normalize_website(website_val: str | None) -> str | None:
        """Normalise un website (strip + ajoute https:// si absent)."""
        if not website_val:
            return None
        website_stripped = website_val.strip()
        if not website_stripped:
            return None
        # Tronquer si trop long
        website_truncated = truncate_field(website_stripped, MAX_LENGTH_WEBSITE)
        if not website_truncated:
            return None
        # Ajouter https:// si absent (déjà fait dans le code suivant, mais on peut le faire ici aussi)
        return website_truncated

    # Normaliser tous les champs
    name = truncate_field(name, MAX_LENGTH_NAME)
    title = truncate_field(title, MAX_LENGTH_TITLE)
    company = truncate_field(company, MAX_LENGTH_COMPANY)
    phone_main = truncate_field(phone_main, MAX_LENGTH_PHONE)
    phone_mobile = truncate_field(phone_mobile, MAX_LENGTH_PHONE)
    email = normalize_email(email)
    website = normalize_website(website)
    address_line = truncate_field(address_line, MAX_LENGTH_ADDRESS_LINE)
    zip_code = truncate_field(zip_code, MAX_LENGTH_ZIP)
    city = truncate_field(city, MAX_LENGTH_CITY)

    # Logo: utiliser uniquement company_obj.logo_url (obligatoire, pas de champ séparé)
    logo_url = None
    if company_obj and hasattr(company_obj, "logo_url") and company_obj.logo_url:
        logo_url = _make_logo_url_absolute(company_obj.logo_url)

    # Construire la colonne gauche
    left_col_parts = []
    if name:
        left_col_parts.append(
            f'<strong style="font-size: 12px;">{html.escape(name)}</strong>'
        )
    if title:
        left_col_parts.append(html.escape(title))
    if company:
        left_col_parts.append(html.escape(company))

    left_col_content = "<br>".join(left_col_parts) if left_col_parts else ""

    # Construire la colonne droite
    right_col_parts = []
    # Téléphones
    phones = []
    if phone_main:
        phones.append(html.escape(phone_main))
    if phone_mobile:
        phones.append(html.escape(phone_mobile))
    if phones:
        right_col_parts.append(" | ".join(phones))

    # Email (mailto link) - email est déjà validé et normalisé
    if email:
        email_escaped = html.escape(email)
        right_col_parts.append(
            f'<a href="mailto:{html.escape(email)}" style="color: #1b4b7a; text-decoration: none;">{email_escaped}</a>'
        )

    # Website (link) - website est déjà normalisé (strip), on ajoute https:// si absent
    if website:
        website_clean = website
        if not website_clean.startswith("http://") and not website_clean.startswith(
            "https://"
        ):
            website_clean = f"https://{website_clean}"
        # Pour l'affichage, on enlève le protocole
        website_display = website.replace("https://", "").replace("http://", "")
        website_escaped = html.escape(website_display)
        right_col_parts.append(
            f'<a href="{html.escape(website_clean)}" style="color: #1b4b7a; text-decoration: none;">{website_escaped}</a>'
        )

    # Adresse
    address_parts = []
    if address_line:
        address_parts.append(html.escape(address_line))
    if zip_code and city:
        address_parts.append(f"{html.escape(zip_code)} {html.escape(city)}")
    elif city:
        address_parts.append(html.escape(city))
    if address_parts:
        right_col_parts.append("<br>".join(address_parts))

    right_col_content = "<br>".join(right_col_parts) if right_col_parts else ""

    # Construire le HTML complet (table-based, compatible Outlook)
    # Wrapper fixe 520px pour éviter l'étirement sur grands écrans
    # align="left" pour Outlook Desktop
    html_parts = [
        '<table cellpadding="0" cellspacing="0" border="0" width="520" align="left" style="width:520px; max-width:520px; font-family: Arial, sans-serif; font-size: 11px; color: #333; margin-top: 12px;">',
        "  <tr>",
        "    <td>",
        '      <table cellpadding="0" cellspacing="0" border="0" style="width: 100%;">',
        "        <tr>",
        '          <td style="vertical-align: top; padding-right: 12px; width: 50%;">',
        left_col_content or "&nbsp;",
        "          </td>",
        '          <td width="1" style="border-left: 2px solid #1b4b7a; padding-left: 12px; vertical-align: top; width: 50%;">',
        right_col_content or "&nbsp;",
        "          </td>",
        "        </tr>",
        "      </table>",
        "    </td>",
        "  </tr>",
        "</table>",
    ]

    # Ajouter la ligne horizontale et le logo si présent
    # Utiliser une mini-table pour la ligne horizontale (meilleure compatibilité Outlook)
    # Logo: utiliser CID inline si possible, sinon URL absolue
    # Toutes les tables doivent être dans le wrapper 520px (pas 100% page)
    EMAIL_SIGNATURE_DEBUG = os.getenv("EMAIL_SIGNATURE_DEBUG", "0") == "1"

    if logo_url:
        company_name = getattr(company_obj, "name", "") if company_obj else ""
        alt_text = f"Logo {company_name}" if company_name else "Logo"
        # Utiliser CID inline strict pour meilleure compatibilité Outlook
        # CID doit être EXACTEMENT "company_logo" (sans chevrons)
        logo_src = "cid:company_logo"
        if EMAIL_SIGNATURE_DEBUG:
            logger.info(
                "[EMAIL_SIGNATURE_DEBUG] generate_signature_html_from_form: logo_src=%s (doit correspondre à contentId='company_logo')",
                logo_src,
            )
        html_parts.extend(
            [
                '<table width="520" align="left" cellpadding="0" cellspacing="0" border="0" style="width:520px; max-width:520px; margin-top: 12px;">',
                '  <tr><td style="border-top: 1px solid #1b4b7a; line-height: 1px; font-size: 1px;">&nbsp;</td></tr>',
                "</table>",
                '<table width="520" align="left" cellpadding="0" cellspacing="0" border="0" style="width:520px; max-width:520px; padding-top: 8px;">',
                "  <tr>",
                "    <td>",
                (
                    f'      <img src="{html.escape(logo_src)}" alt="{html.escape(alt_text)}" height="26" '
                    'style="display:block;border:0;outline:none;text-decoration:none;height:26px;width:auto;max-width:100%;" />'
                ),
                "    </td>",
                "  </tr>",
                "</table>",
            ]
        )
    else:
        # Ligne horizontale même sans logo (mini-table pour Outlook, largeur fixe, dans wrapper)
        html_parts.extend(
            [
                '<table width="520" align="left" cellpadding="0" cellspacing="0" border="0" style="width:520px; max-width:520px; margin-top: 12px;">',
                '<tr><td style="border-top: 1px solid #1b4b7a; line-height: 1px; font-size: 1px;">&nbsp;</td></tr>',
                "</table>",
            ]
        )

    return "\n".join(html_parts)


def inject_signature_into_html(
    html_content: str,
    signature_mode: str | None = None,
    company: Any | None = None,
    billing_settings: Any | None = None,
    *,
    logo_mode: str | None = None,
    cache_bust: str | int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Injecte une signature (texte ou HTML) dans un contenu HTML.

    Supporte trois modes:
    - "form": Génération automatique du HTML depuis formulaire (champs normalisés)
    - "text": Signature texte (échappée, \n → <br>)
    - "html": Template HTML Jinja2 rendu avec variables whitelistées

    Si aucune signature n'est fournie, retourne html_content inchangé.
    La signature est injectée AVANT la fermeture de </body> avec un séparateur propre.

    Args:
        html_content: Contenu HTML de l'email (doit contenir </body>)
        signature_mode: "form", "text" ou "html" (si None, lit depuis billing_settings.email_signature_mode)
        company: Instance Company (requis pour mode "html", optionnel pour "form" logo fallback)
        billing_settings: CompanyBillingSettings (contient tous les champs de signature)
        logo_mode: "cid" (inline CID) | "url" (URL absolue, ex. brevo_api). Si None, comportement par défaut (cid si possible).
        cache_bust: Valeur pour ?v= en mode url (ex. invoice_id, reminder_id).

    Returns:
        Tuple (html_content, logo_info) où:
        - html_content: HTML avec signature injectée (ou html_content inchangé si signature vide)
        - logo_info: Dict avec 'bytes', 'mime_type', 'cid' si logo inline (cid), ou None en mode url
    """
    EMAIL_SIGNATURE_DEBUG = os.getenv("EMAIL_SIGNATURE_DEBUG", "0") == "1"

    if not billing_settings:
        if EMAIL_SIGNATURE_DEBUG:
            logger.info(
                "[EMAIL_SIGNATURE_DEBUG] inject_signature: pas de billing_settings, retour html inchangé"
            )
        return (html_content, None)

    # Déterminer le mode (depuis billing_settings si non fourni)
    if signature_mode is None:
        signature_mode = (
            getattr(billing_settings, "email_signature_mode", "form") or "form"
        )

    signature_html = None

    if signature_mode == "form":
        # Mode form: génération automatique du HTML depuis formulaire
        # Lire les champs depuis billing_settings
        signature_name = getattr(billing_settings, "signature_name", None)
        signature_title = getattr(billing_settings, "signature_title", None)
        signature_company = getattr(billing_settings, "signature_company", None)
        signature_phone_main = getattr(billing_settings, "signature_phone_main", None)
        signature_phone_mobile = getattr(
            billing_settings, "signature_phone_mobile", None
        )
        signature_email = getattr(billing_settings, "signature_email", None)
        signature_website = getattr(billing_settings, "signature_website", None)
        signature_address_line = getattr(
            billing_settings, "signature_address_line", None
        )
        signature_zip = getattr(billing_settings, "signature_zip", None)
        signature_city = getattr(billing_settings, "signature_city", None)

        # Logo: utiliser uniquement company.logo_url (obligatoire, pas de signature_logo_url)
        # company est requis pour le mode "form" si on veut un logo
        signature_html = generate_signature_html_from_form(
            name=signature_name,
            title=signature_title,
            company=signature_company,
            phone_main=signature_phone_main,
            phone_mobile=signature_phone_mobile,
            email=signature_email,
            website=signature_website,
            address_line=signature_address_line,
            zip_code=signature_zip,
            city=signature_city,
            company_obj=company,  # Pour récupérer logo_url automatiquement
        )
        if (
            not signature_html
            or not signature_html.strip()
            or signature_html.strip() == "&nbsp;"
        ):
            # Si aucun champ rempli, fallback sur texte si disponible
            signature_text = getattr(billing_settings, "email_signature_text", None)
            if signature_text:
                signature_mode = "text"
            else:
                if EMAIL_SIGNATURE_DEBUG:
                    logger.info(
                        "[EMAIL_SIGNATURE_DEBUG] inject_signature: signature form vide, retour html inchangé"
                    )
                return (html_content, None)

    elif signature_mode == "html":
        signature_html_template = getattr(
            billing_settings, "email_signature_html_template", None
        )
        signature_text = getattr(billing_settings, "email_signature_text", None)

        if not signature_html_template:
            # Fallback sur texte si template vide
            if signature_text:
                signature_mode = "text"
            else:
                if EMAIL_SIGNATURE_DEBUG:
                    logger.info(
                        "[EMAIL_SIGNATURE_DEBUG] inject_signature: template HTML vide, retour html inchangé"
                    )
                return (html_content, None)
        elif not company:
            # Fallback sur texte si company manquante
            if signature_text:
                signature_mode = "text"
            else:
                if EMAIL_SIGNATURE_DEBUG:
                    logger.info(
                        "[EMAIL_SIGNATURE_DEBUG] inject_signature: company manquante, retour html inchangé"
                    )
                return (html_content, None)
        else:
            # Render le template HTML (signature_html_template est garanti non-None ici)
            assert signature_html_template is not None  # Type narrowing pour le linter
            signature_html = render_signature_html_template(
                signature_html_template, company, billing_settings
            )
            if not signature_html:
                # Si le render échoue, fallback sur texte si disponible
                if signature_text:
                    signature_mode = "text"
                else:
                    if EMAIL_SIGNATURE_DEBUG:
                        logger.info(
                            "[EMAIL_SIGNATURE_DEBUG] inject_signature: render template échoué, retour html inchangé"
                        )
                    return (html_content, None)

    if signature_mode == "text":
        signature_text = getattr(billing_settings, "email_signature_text", None)
        if signature_text:
            # Mode texte: échapper et convertir \n → <br>
            signature_escaped = html.escape(signature_text.strip())
            signature_html = signature_escaped.replace("\n", "<br>")

    if not signature_html or not signature_html.strip():
        if EMAIL_SIGNATURE_DEBUG:
            logger.info(
                "[EMAIL_SIGNATURE_DEBUG] inject_signature: signature_html vide, retour html inchangé"
            )
        return (html_content, None)

    # Récupérer les infos du logo pour attachement inline
    logo_info = None
    if company and hasattr(company, "logo_url") and company.logo_url:
        # Mode URL (brevo_api): forcer URL absolue + cache-busting, pas d'inline CID
        if logo_mode == "url":
            logo_url_absolute = _make_logo_url_absolute(company.logo_url)
            if logo_url_absolute:
                if cache_bust is not None:
                    logo_url_absolute = (
                        f"{logo_url_absolute.rstrip('?')}?v={cache_bust}"
                    )
                signature_html = signature_html.replace(
                    "cid:company_logo", logo_url_absolute
                )
                if EMAIL_SIGNATURE_DEBUG:
                    logger.info(
                        "[EMAIL_SIGNATURE_DEBUG] inject_signature: logo_mode=url, url absolue finale=%s",
                        logo_url_absolute,
                    )
            # logo_info reste None en mode url (pas d'attachement inline)
        else:
            # Mode CID (brevo_smtp ou défaut): logo inline avec cid:company_logo
            logo_bytes, mime_type = _get_logo_bytes(company.logo_url)
            if logo_bytes and mime_type:
                # Vérifier que les bytes ne sont pas vides
                if len(logo_bytes) == 0:
                    logger.warning(
                        "Logo bytes vides pour %s - fallback vers URL absolue",
                        company.logo_url,
                    )
                    logo_url_absolute = _make_logo_url_absolute(company.logo_url)
                    if logo_url_absolute:
                        if EMAIL_SIGNATURE_DEBUG:
                            logger.info(
                                "[EMAIL_SIGNATURE_DEBUG] inject_signature: logo bytes vides, fallback URL absolue - %s",
                                logo_url_absolute,
                            )
                        signature_html = signature_html.replace(
                            "cid:company_logo", logo_url_absolute
                        )
                # SVG guard: Outlook ne supporte pas bien les SVG inline
                elif mime_type == "image/svg+xml":
                    logger.warning(
                        "Logo SVG détecté (%s) - fallback vers URL absolue (Outlook incompatible)",
                        company.logo_url,
                    )
                    logo_url_absolute = _make_logo_url_absolute(company.logo_url)
                    if logo_url_absolute:
                        if EMAIL_SIGNATURE_DEBUG:
                            logger.info(
                                "[EMAIL_SIGNATURE_DEBUG] inject_signature: logo SVG fallback URL absolue - %s",
                                logo_url_absolute,
                            )
                        signature_html = signature_html.replace(
                            "cid:company_logo", logo_url_absolute
                        )
                else:
                    # CID strict: exactement "company_logo" pour Outlook (sans chevrons)
                    logo_info = {
                        "bytes": logo_bytes,
                        "mime_type": mime_type,
                        "cid": "company_logo",
                        "filename": f"logo{Path(company.logo_url).suffix or '.png'}",
                    }
                    if EMAIL_SIGNATURE_DEBUG:
                        img_src_match = re.search(
                            r'<img[^>]+src=["\']([^"\']+)["\']', signature_html
                        )
                        html_img_src = (
                            img_src_match.group(1) if img_src_match else "NON TROUVÉ"
                        )
                        logger.info(
                            (
                                "[EMAIL_SIGNATURE_DEBUG] inject_signature: logo inline disponible - "
                                "html_img_src=%s, cid=%s, filename=%s, mime_type=%s, bytes_len=%d"
                            ),
                            html_img_src,
                            logo_info["cid"],
                            logo_info["filename"],
                            logo_info["mime_type"],
                            len(logo_bytes),
                        )
                        if html_img_src != f"cid:{logo_info['cid']}":
                            logger.warning(
                                "[EMAIL_SIGNATURE_DEBUG] ⚠️ MISMATCH CID: html_img_src=%s != cid:company_logo",
                                html_img_src,
                            )
            else:
                # Fallback: URL absolue (fichier non trouvé ou erreur)
                logo_url_absolute = _make_logo_url_absolute(company.logo_url)
                if logo_url_absolute:
                    if EMAIL_SIGNATURE_DEBUG:
                        logger.info(
                            "[EMAIL_SIGNATURE_DEBUG] inject_signature: logo fallback URL absolue (fichier non trouvé ou erreur) - %s",
                            logo_url_absolute,
                        )
                    signature_html = signature_html.replace(
                        "cid:company_logo", logo_url_absolute
                    )

    # Séparateur propre (—)
    separator = "<br><br>—<br>"

    # Injecter AVANT </body>
    if "</body>" in html_content:
        html_content = html_content.replace(
            "</body>", f"{separator}{signature_html}</body>"
        )
    elif "</html>" in html_content:
        html_content = html_content.replace(
            "</html>", f"{separator}{signature_html}</html>"
        )
    else:
        html_content = f"{html_content}{separator}{signature_html}"

    if EMAIL_SIGNATURE_DEBUG:
        logger.info(
            (
                "[EMAIL_SIGNATURE_DEBUG] inject_signature: signature injectée - "
                "mode=%s, logo_info=%s, wrapper_width=520px"
            ),
            signature_mode,
            "inline" if logo_info else "url",
        )

    return (html_content, logo_info)
