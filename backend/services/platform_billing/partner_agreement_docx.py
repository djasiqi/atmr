"""Composition DOCX du contrat-cadre partenaire LIRIE (python-docx)."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from models.enums import LegalForm

GENERATOR_VERSION = "1.4"
TEMPLATE_VERSION = "lirie-partner-v1.4"
TEMPLATE_RELATIVE = Path("templates/contracts/lirie_partenariat_base_v1.docx")

DEFAULT_OPERATOR_CONTRACTUAL_EMAIL = "info@lirie.ch"

_CANCEL_LABELS = {
    "exclude": (
        "les courses annulées, y compris celles donnant lieu à des frais "
        "d'annulation, sont exclues de la base de commission"
    ),
    "on_cancellation_fees": (
        "lorsqu'un frais d'annulation est effectivement facturé au client final, "
        "la commission est calculée sur le montant HT de ce seul frais ; "
        "sinon la course annulée est exclue"
    ),
    "on_billed_amount": (
        "la commission est calculée sur le montant HT facturé au client final, "
        "y compris en cas d'annulation facturée"
    ),
}

_LIABILITY_FLOOR_CHF = "10'000"
_BRAND_ALIASES = frozenset({"lirie", "lirie.ch", "lirie sa", "lirie sàrl"})


def template_path() -> Path:
    # backend/services/platform_billing -> backend/
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / TEMPLATE_RELATIVE


def ensure_base_template() -> Path:
    """Crée le document de base (styles) s'il est absent."""
    path = template_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    doc.add_heading("LIRIE — modèle de base contrat partenaire", level=1)
    doc.add_paragraph(
        "Document de styles uniquement. Le contenu juridique est composé par code."
    )
    doc.save(str(path))
    return path


def template_sha256() -> str:
    path = ensure_base_template()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(rate: str | None) -> str:
    if rate is None or rate == "":
        return "—"
    try:
        n = float(str(rate).replace(",", "."))
        return f"{(n * 100):.2f}".rstrip("0").rstrip(".") + " %"
    except ValueError:
        return str(rate)


def _fmt_ide(uid_ide: str | None) -> str:
    value = (uid_ide or "").strip()
    return value if value else "non attribué"


def _normalize_street(value: str) -> str:
    """Corrige « Ernest- Pictet » → « Ernest-Pictet »."""
    text = (value or "").strip()
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text)


def _fmt_address(party: dict[str, Any]) -> str:
    addr = _normalize_street(party.get("street_name") or "")
    if party.get("building_number"):
        addr = f"{addr} {party['building_number']}".strip()
    postal = (party.get("postal_code") or "").strip()
    city = (party.get("city") or "").strip()
    country = (party.get("country_code") or "CH").strip()
    country_label = "Suisse" if country.upper() == "CH" else country
    return f"{addr}, {postal} {city}, {country_label}".strip(", ")


def _is_sole_proprietor(party: dict[str, Any]) -> bool:
    return (party.get("legal_form") or "") == LegalForm.SOLE_PROPRIETORSHIP.value


def _operator_natural_person_name(party: dict[str, Any]) -> str:
    """Personne physique de l'Exploitant (pas l'enseigne LIRIE)."""
    signatory = (party.get("signatory_name") or "").strip()
    legal = (party.get("legal_name") or "").strip()
    if _is_sole_proprietor(party):
        if signatory:
            return signatory
        if legal and legal.lower() not in _BRAND_ALIASES:
            return legal
        return legal or "—"
    return legal or signatory or "—"


def _operator_display_name(party: dict[str, Any]) -> str:
    """Identification juridique de l'Exploitant (indépendant sous enseigne LIRIE)."""
    name = _operator_natural_person_name(party)
    if _is_sole_proprietor(party):
        return f"{name}, exerçant en qualité d'indépendant sous l'enseigne LIRIE"
    label = party.get("legal_form_label") or party.get("legal_form") or ""
    if label:
        return f"{name} ({label}), exploitant la plateforme LIRIE"
    return f"{name}, exploitant la plateforme LIRIE"


def _partner_legal_name(party: dict[str, Any]) -> str:
    """Raison sociale complète (ex. Emmenez-moi Sàrl)."""
    name = (party.get("legal_name") or "").strip() or "—"
    form = party.get("legal_form") or ""
    label = (party.get("legal_form_label") or "").strip()
    if form == LegalForm.SARL.value and label:
        if "sàrl" not in name.lower() and "sarl" not in name.lower():
            return f"{name} {label}"
    if form == LegalForm.SA.value and label:
        if not re.search(r"\bSA\b", name, flags=re.IGNORECASE):
            return f"{name} {label}"
    return name


def _contractual_email(party: dict[str, Any], *, is_operator: bool) -> str:
    email = (party.get("contractual_email") or "").strip()
    if email:
        return email
    if is_operator:
        return DEFAULT_OPERATOR_CONTRACTUAL_EMAIL
    return "à compléter"


def _party_block(*, role: str, party: dict[str, Any], is_operator: bool) -> list[str]:
    if is_operator:
        identity_line = _operator_display_name(party)
        domicile_label = "Domicilié à"
        form_line = (
            "Statut : indépendant"
            if _is_sole_proprietor(party)
            else f"Forme juridique : {party.get('legal_form_label') or '—'}"
        )
    else:
        identity_line = _partner_legal_name(party)
        domicile_label = "Siège"
        form_line = f"Forme juridique : {party.get('legal_form_label') or '—'}"

    lines = [
        role,
        identity_line,
        form_line,
        f"{domicile_label} : {_fmt_address(party)}",
        f"IDE : {_fmt_ide(party.get('uid_ide'))}",
        f"Courriel contractuel : {_contractual_email(party, is_operator=is_operator)}",
    ]

    if is_operator and _is_sole_proprietor(party):
        # L'indépendant agit personnellement — pas de représentation.
        return lines

    signatory = (party.get("signatory_name") or "").strip() or "—"
    title = (party.get("signatory_title") or "").strip()
    represented = f"Représenté(e) par : {signatory}"
    if title:
        represented = f"{represented}, {title}"
    lines.append(represented)
    return lines


def _add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def _add_para(doc: Document, text: str, *, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold


def _add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def _add_signature_block(doc: Document, *, label: str, signatory: str | None) -> None:
    _add_para(doc, label, bold=True)
    _add_para(doc, (signatory or "").strip() or "________________")
    _add_para(doc, "Lieu : _______________________________")
    _add_para(doc, "Date : _______________________________")
    _add_para(doc, "Signature : __________________________")
    _add_para(doc, "")


def build_partner_agreement_docx_bytes(
    *,
    reference: str,
    parties: dict[str, Any],
    commercial: dict[str, Any],
    agreement_effective_from: str,
) -> bytes:
    """Compose le contrat-cadre et retourne les octets DOCX."""
    ensure_base_template()
    doc = Document(str(template_path()))

    # Nettoyer le corps du modèle de base
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("sectPr"):
            continue
        body.remove(child)

    operator = parties.get("operator") or {}
    partner = parties.get("partner") or {}
    mode = commercial.get("subscription_pricing_mode") or "volume"
    free_months = commercial.get("free_license_max_months")
    commission_rate = commercial.get("commission_rate")
    cancel_policy = commercial.get("commission_cancellation_policy") or "exclude"
    payment_days = commercial.get("payment_terms_days") or 30
    dispute_days = commercial.get("statement_dispute_days") or 10
    commission_enabled = bool(commercial.get("lirie_commission_enabled", True))
    own_enabled = bool(commercial.get("own_portfolio_billing_enabled", True))

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run(
        "CONTRAT CADRE DE PARTENARIAT\n"
        "& LICENCE D'UTILISATION DE LA PLATEFORME LIRIE"
    )
    r.bold = True
    r.font.size = Pt(14)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("Plateforme digitale LIRIE\nwww.lirie.ch")

    ref_p = doc.add_paragraph()
    ref_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ref_p.add_run(f"Réf. : {reference}").bold = True

    _add_para(doc, f"Entrée en vigueur : {agreement_effective_from}")
    _add_para(
        doc,
        "Le contrat entre en vigueur à la date d'effet indiquée ci-dessus, "
        "y compris de manière rétroactive si la signature intervient ultérieurement, "
        "indépendamment de la date de signature.",
    )
    _add_para(doc, f"Version modèle : {TEMPLATE_VERSION}")

    _add_heading(doc, "ENTRE", level=1)
    for line in _party_block(
        role="L'Exploitant", party=operator, is_operator=True
    ):
        _add_para(doc, line)
    _add_para(doc, "Ci-après désigné : « l'Exploitant »", bold=True)

    _add_heading(doc, "ET", level=1)
    for line in _party_block(
        role="Le Partenaire", party=partner, is_operator=False
    ):
        _add_para(doc, line)
    _add_para(doc, "Ci-après désigné : « le Partenaire »", bold=True)
    _add_para(
        doc,
        "L'Exploitant et le Partenaire étant ensemble désignés « les Parties ».",
    )

    _add_heading(doc, "PRÉAMBULE", level=1)
    _add_para(
        doc,
        "L'Exploitant développe et exploite une plateforme digitale de gestion "
        "et de coordination de transports, dénommée « LIRIE ».",
    )
    _add_para(
        doc,
        "Le Partenaire exerce une activité professionnelle de transport de personnes, "
        "notamment à mobilité réduite.",
    )
    _add_para(doc, "Les Parties souhaitent définir les conditions dans lesquelles :")
    _add_bullets(
        doc,
        [
            "le Partenaire pourra utiliser la plateforme LIRIE ;",
            "les courses du portefeuille propre et les Courses Marketplace LIRIE "
            "seront distinguées ;",
            "les modalités financières et responsabilités seront encadrées.",
        ],
    )
    _add_para(doc, "Il est convenu ce qui suit.")

    _add_heading(doc, "ARTICLE 1 – OBJET", level=1)
    _add_para(doc, "Le présent contrat a pour objet :")
    _add_bullets(
        doc,
        [
            "l'octroi d'un droit d'utilisation professionnel de la plateforme LIRIE ;",
            "l'organisation des prestations de transport issues de la plateforme ;",
            "la fixation des conditions financières applicables ;",
            "la définition des responsabilités respectives des Parties.",
        ],
    )

    _add_heading(doc, "ARTICLE 2 – INDÉPENDANCE DES PARTIES", level=1)
    _add_para(
        doc,
        "Les Parties agissent en qualité d'entités indépendantes. Le présent contrat "
        "ne constitue ni un contrat de travail, ni une société simple au sens des "
        "art. 530 ss CO, ni une relation d'agence ou de représentation, ni un mandat exclusif.",
    )

    _add_heading(doc, "ARTICLE 3 – STATUT DE LIRIE", level=1)
    _add_para(
        doc,
        "L'Exploitant intervient comme fournisseur de plateforme digitale et, pour les "
        "Courses Marketplace LIRIE, comme intermédiaire de mise en relation. Il n'est "
        "ni transporteur, ni employeur des chauffeurs, ni partie au contrat de transport "
        "conclu entre le Partenaire et le client final ou le payeur désigné. "
        "L'Exploitant ne garantit pas de volume de demandes ni la solvabilité des clients.",
    )

    _add_heading(doc, "ARTICLE 4 – LICENCE D'UTILISATION", level=1)
    _add_para(
        doc,
        "L'Exploitant accorde au Partenaire une licence non exclusive, non cessible, "
        "strictement professionnelle et limitée à la durée du présent contrat.",
    )

    _add_heading(doc, "ARTICLE 5 – PORTEFEUILLE PROPRE ET MARKETPLACE", level=1)
    _add_para(
        doc,
        "Est une « Course du portefeuille propre » toute course créée directement par "
        "le Partenaire dans son espace LIRIE pour un client ou une relation commerciale "
        "qu'il gère indépendamment du réseau d'acquisition LIRIE "
        "(origine commerciale OWN_PORTFOLIO).",
    )
    _add_para(
        doc,
        "Est une « Course Marketplace LIRIE » toute demande créée par une institution, "
        "un client privé ou un autre utilisateur distinct du Partenaire, transmise au "
        "Partenaire par le système de mise en relation LIRIE et expressément acceptée "
        "par celui-ci (origine commerciale LIRIE_MARKETPLACE). Son exécution, son "
        "annulation ou sa facturation n'altère pas son origine commerciale ; ces "
        "événements déterminent uniquement son traitement financier conformément à "
        "l'article 6.",
    )
    if own_enabled and mode == "free":
        months_txt = str(free_months) if free_months else "60"
        _add_para(
            doc,
            f"La licence relative au portefeuille propre est gratuite pendant au maximum "
            f"{months_txt} mois calendaires à compter de l'entrée en vigueur, tant que "
            "le contrat demeure en vigueur. Aucun abonnement ni frais fixe d'utilisation "
            "n'est dû pour cette licence durant cette période. La gratuité ne concerne "
            "pas les commissions sur les Courses Marketplace LIRIE.",
        )
        _add_para(
            doc,
            f"À l'expiration des {months_txt} mois de gratuité, les éventuelles "
            "conditions tarifaires applicables à la licence du portefeuille propre "
            "devront faire l'objet d'un avenant écrit. Aucun abonnement ne sera "
            "appliqué automatiquement en l'absence d'un tel avenant.",
        )
    elif own_enabled and mode == "fixed":
        _add_para(
            doc,
            "L'abonnement portefeuille propre est facturé selon le montant fixe convenu "
            "dans les conditions commerciales annexées au présent document.",
        )
    elif own_enabled:
        _add_para(
            doc,
            "L'abonnement portefeuille propre est facturé selon le volume mensuel "
            "conformément à la grille tarifaire LIRIE applicable.",
        )
    else:
        _add_para(
            doc,
            "La facturation de l'abonnement portefeuille propre n'est pas activée "
            "pour ce Partenaire.",
        )

    _add_heading(doc, "ARTICLE 6 – COMMISSION", level=1)
    if commission_enabled:
        _add_para(
            doc,
            f"Pour toute Course Marketplace LIRIE commissionnable, le Partenaire "
            f"verse à l'Exploitant une commission de {_pct(commission_rate)} du montant "
            "HT définitif facturé au titre de la prestation de transport, y compris les "
            "suppléments directement liés à la course, après déduction des remises, "
            "rabais, remboursements et notes de crédit. Sont exclus les pourboires, "
            "débours remboursés au prix coûtant, taxes publiques et montants de TVA.",
        )
        _add_para(
            doc,
            "Le Partenaire renseigne dans LIRIE le montant HT définitif de la "
            "prestation au plus tard dans les cinq (5) jours ouvrables suivant son "
            "exécution et conserve les justificatifs correspondants. À la demande "
            "motivée de l'Exploitant, il fournit les éléments strictement nécessaires "
            "à la vérification du relevé. Le montant ne peut être artificiellement "
            "réduit dans le but de diminuer la commission.",
        )
        _add_para(
            doc,
            f"Politique d'annulation : {_CANCEL_LABELS.get(cancel_policy, cancel_policy)}.",
        )
        _add_para(
            doc,
            f"Modalités : décompte mensuel ; facturation par l'Exploitant ; paiement "
            f"sous {payment_days} jours à compter de la date d'émission de la facture ; "
            f"intérêt moratoire de 5 % l'an (art. 104 CO) en cas de retard. Le "
            f"Partenaire dispose de {dispute_days} jours ouvrables à compter de la "
            "notification de la mise à disposition du relevé dans LIRIE pour le "
            "contester de manière motivée ; à défaut, le relevé est réputé accepté sous "
            "réserve d'erreur manifeste. La contestation d'une partie ne suspend pas le "
            "paiement de la partie non contestée. Le Partenaire est en demeure "
            "automatiquement à l'échéance indiquée sur la facture, sans rappel préalable.",
        )
    else:
        _add_para(
            doc,
            "Aucune commission Marketplace n'est due tant que le produit commission "
            "n'est pas activé dans les conditions commerciales.",
        )

    _add_heading(doc, "ARTICLE 6 BIS – DÉFAUT DE PAIEMENT", level=1)
    auto_dunning = bool(commercial.get("automated_dunning_enabled", True))
    rem_delay = int(commercial.get("reminder_delay_days_after_due") or 0)
    rem_grace = int(commercial.get("reminder_grace_days") or 10)
    full_days = int(commercial.get("full_suspend_days_after_due") or 30)
    full_count = int(commercial.get("full_suspend_overdue_invoice_count") or 2)
    term_days = int(commercial.get("termination_notice_days") or 10)
    block_offers = bool(commercial.get("partial_block_marketplace_offers", True))
    block_accept = bool(
        commercial.get("partial_block_marketplace_acceptance", True)
    )
    block_support = bool(commercial.get("partial_block_billable_support", True))
    block_config = bool(
        commercial.get("partial_block_billable_configuration", True)
    )

    _add_para(
        doc,
        "À défaut de paiement à l'échéance, le Partenaire est automatiquement en "
        "demeure, sans rappel préalable. Le montant impayé porte intérêt au taux "
        "de 5 % l'an à compter de son échéance.",
    )
    if not auto_dunning:
        _add_para(
            doc,
            "Les mesures automatisées de rappel et de suspension ne sont pas "
            "activées pour le présent contrat. L'Exploitant conserve ses droits "
            "de réclamer les montants échus, les intérêts moratoires, de "
            "suspendre les services après notification et d'engager les "
            "démarches de recouvrement prévues par le droit applicable.",
        )
    else:
        _add_para(
            doc,
            f"L'Exploitant adresse au Partenaire une notification "
            f"(délai de {rem_delay} jour(s) après l'échéance) lui accordant un "
            f"délai supplémentaire de {rem_grace} jours calendaires pour régler "
            "l'intégralité des montants échus ou contester la facture de manière "
            "motivée.",
        )
        _add_para(
            doc,
            "Si le paiement n'est pas intervenu à l'expiration de ce délai, "
            "l'Exploitant peut, après en avoir informé le Partenaire :",
        )
        bullets: list[str] = []
        if block_offers or block_accept:
            bullets.append(
                "suspendre la transmission et/ou l'acceptation de nouvelles "
                "Courses Marketplace LIRIE ;"
            )
        if block_support or block_config:
            bullets.append(
                "suspendre l'accès aux fonctionnalités donnant lieu à "
                "facturation et refuser les prestations de support ou de "
                "configuration facturables ;"
            )
        bullets.append(
            "engager une procédure de recouvrement ou de poursuite."
        )
        _add_bullets(doc, bullets)
        _add_para(
            doc,
            "La suspension ne doit pas empêcher l'exécution et le suivi des "
            "courses déjà engagées, ni l'accès temporaire aux informations "
            "nécessaires à leur réalisation, à la sécurité des passagers, au "
            "paiement des factures ou à l'export des données du Partenaire.",
        )
        _add_para(
            doc,
            f"Lorsque le retard dépasse {full_days} jours, lorsque "
            f"{full_count} factures échues et impayées demeurent ouvertes ou "
            "lorsque le Partenaire ne respecte pas un accord de paiement écrit, "
            "l'Exploitant peut restreindre l'ensemble des opérations "
            "commerciales nouvelles et résilier le contrat pour faute grave, "
            f"après une dernière mise en demeure restée sans effet pendant "
            f"{term_days} jours.",
        )

    _add_para(
        doc,
        "La suspension ou la résiliation ne libère pas le Partenaire du paiement "
        "des factures, commissions, intérêts moratoires et autres montants déjà "
        "dus.",
    )
    _add_para(
        doc,
        "Les frais officiels de poursuite ainsi que tout dommage supplémentaire "
        "nécessaire et démontré résultant du retard peuvent être réclamés au "
        "Partenaire conformément au droit applicable. Aucun frais forfaitaire de "
        "recouvrement non justifié n'est automatiquement dû.",
    )
    _add_para(
        doc,
        "L'Exploitant peut, sans y être obligé, accepter un échéancier écrit. Un "
        "tel accord ne constitue ni une remise de dette ni une renonciation aux "
        "intérêts, aux garanties ou aux autres droits de l'Exploitant, sauf "
        "disposition écrite contraire.",
    )

    _add_heading(doc, "ARTICLE 7 – INTERDICTION DE CONTOURNEMENT", level=1)
    _add_para(
        doc,
        "Le Partenaire s'interdit de solliciter activement un client qui lui a été "
        "présenté pour la première fois par l'intermédiaire du réseau LIRIE dans le "
        "but de soustraire à la plateforme des demandes qui auraient normalement dû "
        "y être enregistrées. Cette interdiction ne s'applique pas aux relations "
        "commerciales préexistantes démontrables, aux appels d'offres publics ou "
        "ouverts, ni aux demandes reçues indépendamment de toute utilisation des "
        "données LIRIE. Elle demeure applicable pendant la durée du contrat et "
        "pendant douze (12) mois après sa résiliation. Toute violation constitue "
        "un manquement grave et peut entraîner le paiement des commissions éludées "
        "ainsi que des dommages-intérêts complémentaires prouvés.",
    )

    _add_heading(doc, "ARTICLE 8 – RESPONSABILITÉS", level=1)
    _add_para(
        doc,
        "Le Partenaire demeure seul responsable de l'exécution des transports, de son "
        "personnel, de ses véhicules, des retards, accidents, dommages, du prix, de "
        "la facturation client et du respect des lois suisses applicables.",
    )
    _add_para(
        doc,
        "L'Exploitant demeure responsable de l'exploitation de la plateforme dans son "
        "périmètre, de la sécurité relevant de son contrôle, de la transmission "
        "conforme des informations reçues, ainsi que du calcul et de l'émission des "
        "relevés de commission. Sauf dol, faute grave ou responsabilité ne pouvant "
        "légalement être limitée, la responsabilité totale de l'Exploitant est limitée "
        f"au montant le plus élevé entre CHF {_LIABILITY_FLOOR_CHF} et les commissions "
        "effectivement payées par le Partenaire au cours des douze derniers mois.",
    )

    _add_heading(doc, "ARTICLE 9 – ASSURANCES", level=1)
    _add_para(
        doc,
        "Le Partenaire garantit disposer en permanence d'une assurance RC "
        "professionnelle, d'assurances véhicules adaptées, d'une couverture "
        "passagers et des autorisations administratives nécessaires. Il fournit "
        "à première demande, et au minimum une fois par année, une attestation "
        "d'assurance à jour, et informe immédiatement l'Exploitant de toute "
        "suspension, réduction ou résiliation de couverture.",
    )

    _add_heading(doc, "ARTICLE 10 – DONNÉES PERSONNELLES (SOCLE LPD)", level=1)
    _add_para(
        doc,
        "Les Parties traitent des données personnelles, y compris le cas échéant des "
        "données sensibles liées à la santé, dans le respect de la LPD. Chaque Partie "
        "agit selon son rôle : l'institution ou le client peut être responsable du "
        "traitement pour ses finalités ; LIRIE traite les données pour la "
        "coordination, la sécurité et la facturation plateforme ; le Partenaire "
        "traite les données nécessaires à l'exécution du transport.",
    )
    _add_bullets(
        doc,
        [
            "traitement limité aux finalités d'exécution du service et instructions "
            "contractuelles ;",
            "confidentialité et accès réservés aux personnes autorisées ;",
            "mesures de sécurité techniques et organisationnelles appropriées ;",
            "chaque Partie notifie l'autre Partie sans délai injustifié et, dans la "
            "mesure du possible, dans les 24 heures suivant la découverte d'un "
            "incident pertinent ; la Partie responsable du traitement procède, le "
            "cas échéant, aux annonces légales requises ;",
            "le recours à des sous-traitants techniques ultérieurs est autorisé sous "
            "réserve de garanties équivalentes ; l'Exploitant informe le Partenaire "
            "de tout changement significatif afin de lui permettre de formuler une "
            "objection motivée ;",
            "restitution ou suppression des données à la fin du contrat, sous "
            "réserve des obligations légales de conservation.",
        ],
    )
    _add_para(
        doc,
        "Le présent article constitue un socle minimal. Une annexe détaillée de "
        "protection des données pourra le compléter ultérieurement.",
    )

    _add_heading(doc, "ARTICLE 11 – PROPRIÉTÉ INTELLECTUELLE", level=1)
    _add_para(
        doc,
        "La plateforme LIRIE (code, architecture, algorithmes, interfaces, marque, "
        "documentation) demeure la propriété exclusive de l'Exploitant. Les données "
        "du portefeuille propre du Partenaire restent sous son contrôle. Le Partenaire "
        "accorde à l'Exploitant le droit limité de traiter les données introduites "
        "dans la plateforme dans la seule mesure nécessaire à la fourniture, à la "
        "sécurisation, à la facturation et à l'amélioration du service. "
        "L'amélioration générale du service repose, autant que possible, sur des "
        "données anonymisées ou agrégées.",
    )

    _add_heading(doc, "ARTICLE 12 – ÉVOLUTION DE LA PLATEFORME", level=1)
    _add_para(
        doc,
        "L'Exploitant peut faire évoluer les fonctionnalités, l'architecture et "
        "l'interface de LIRIE. Il veille à ne pas supprimer, sans motif légitime ni "
        "préavis raisonnable, les fonctionnalités essentielles nécessaires à l'usage "
        "convenu. Toute modification du taux de commission, de la gratuité ou des "
        "obligations principales nécessite un avenant écrit.",
    )

    _add_heading(doc, "ARTICLE 13 – CESSION", level=1)
    _add_para(
        doc,
        "Le Partenaire accepte que le présent contrat puisse être cédé, transféré "
        "ou transmis à toute société ultérieurement constituée pour exploiter LIRIE, "
        "à toute société du groupe ou à toute entité issue d'une fusion ou "
        "restructuration. La cession sera notifiée au Partenaire et prendra effet "
        "à condition que le cessionnaire reprenne l'ensemble des droits et "
        "obligations découlant du présent contrat. À compter de la date d'effet "
        "notifiée de la cession, le cessionnaire se substitue à l'Exploitant pour "
        "les obligations futures. Les obligations nées avant cette date demeurent "
        "régies par le présent contrat.",
    )

    _add_heading(doc, "ARTICLE 14 – DURÉE ET RÉSILIATION", level=1)
    _add_para(
        doc,
        "Le contrat entre en vigueur à sa date d'effet pour une durée initiale de "
        "cinq (5) ans. À l'issue de cette durée initiale, le contrat se renouvelle "
        "tacitement pour une durée indéterminée. Il peut alors être résilié par "
        "chaque Partie moyennant un préavis écrit de trois (3) mois pour la fin "
        "d'un mois. Pendant la durée initiale, chaque Partie peut également le "
        "résilier de manière ordinaire moyennant le même préavis.",
    )
    _add_para(
        doc,
        "Une résiliation immédiate est possible en cas de faute grave, notamment "
        "contournement, fraude, absence d'assurance, atteinte grave aux données, "
        "ou comportement objectivement susceptible de causer un dommage grave et "
        "démontrable à la réputation de LIRIE. Sauf urgence, fraude ou manquement "
        "irréparable, la résiliation immédiate ne peut intervenir qu'après une mise "
        "en demeure restée sans effet pendant dix (10) jours.",
    )
    _add_para(
        doc,
        "Les courses déjà acceptées restent à exécuter sauf instruction contraire "
        "justifiée ; les commissions acquises restent dues ; les accès sont révoqués ; "
        "les clauses de confidentialité, données, propriété intellectuelle, paiement "
        "et non-contournement survivent.",
    )

    _add_heading(doc, "ARTICLE 15 – FORCE MAJEURE", level=1)
    _add_para(
        doc,
        "Aucune Partie n'est responsable en cas d'événement imprévisible et "
        "indépendant de sa volonté. La Partie empêchée informe l'autre sans délai "
        "et prend les mesures raisonnables pour limiter les effets. Si "
        "l'empêchement se prolonge au-delà de soixante (60) jours, chaque Partie "
        "peut résilier le contrat sans indemnité.",
    )

    _add_heading(doc, "ARTICLE 16 – DROIT APPLICABLE ET FOR", level=1)
    _add_para(
        doc,
        "Le présent contrat est soumis au droit matériel suisse, à l'exclusion de "
        "ses règles de conflit de lois. Sous réserve des fors impératifs, les "
        "tribunaux ordinaires du canton de Genève sont exclusivement compétents.",
    )

    _add_heading(doc, "ARTICLE 17 – DISPOSITIONS FINALES", level=1)
    _add_para(
        doc,
        "Si une clause est invalide, le reste du contrat demeure valable. Le fait "
        "de ne pas faire valoir immédiatement un droit ne constitue pas une "
        "renonciation. Le présent contrat remplace les discussions antérieures "
        "portant sur le même objet.",
    )
    _add_para(
        doc,
        "Les notifications de résiliation, de mise en demeure ou de modification "
        "contractuelle sont effectuées par courrier recommandé ou par courrier "
        "électronique avec confirmation de réception aux adresses désignées par "
        "les Parties, notamment aux courriels contractuels indiqués ci-dessus. "
        "Les autres notifications contractuelles sont valablement adressées aux "
        "coordonnées indiquées par les Parties.",
    )

    _add_heading(doc, "SIGNATURES", level=1)
    _add_para(doc, "Fait à Genève")
    _add_para(doc, f"Référence : {reference}")
    _add_para(doc, "")
    operator_signatory = (
        _operator_natural_person_name(operator)
        if _is_sole_proprietor(operator)
        else (operator.get("signatory_name") or _operator_natural_person_name(operator))
    )
    _add_signature_block(
        doc,
        label="Pour l'Exploitant :",
        signatory=operator_signatory,
    )
    _add_signature_block(
        doc,
        label="Pour le Partenaire :",
        signatory=partner.get("signatory_name"),
    )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
