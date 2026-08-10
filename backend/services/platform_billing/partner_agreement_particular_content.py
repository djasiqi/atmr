"""Source unique du contenu du contrat particulier LIRIE (3 pages)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from models.enums import LegalForm
from services.platform_billing.partner_agreement_versions import (
    DPA_VERSION,
    GENERAL_TERMS_VERSION,
    PARTICULAR_VERSION,
    PENALTY_CURRENCY,
    PENALTY_MINIMUM_CHF,
    PENALTY_MULTIPLIER,
    RETENTION_POLICY_VERSION,
    SUBPROCESSORS_VERSION,
)

_FR_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
DEFAULT_OPERATOR_EMAIL = "info@lirie.ch"
PARTNER_POWER_ATTESTATION = (
    "Le signataire atteste disposer du pouvoir nécessaire pour engager le Partenaire."
)


@dataclass(frozen=True)
class PartyColumn:
    header: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class CommercialRow:
    label: str
    value: str


@dataclass(frozen=True)
class ClauseBlock:
    title: str
    body: str
    emphasize: bool = False


@dataclass(frozen=True)
class RoleRow:
    treatment: str
    role: str


@dataclass(frozen=True)
class IncorporatedDoc:
    label: str
    version: str
    sha256: str


@dataclass(frozen=True)
class SignatureBlock:
    side: str
    name: str
    title: str
    co_signatory_name: str | None = None
    co_signatory_title: str | None = None
    power_attestation: str | None = None


@dataclass(frozen=True)
class ParticularAgreementContent:
    """Modèle structuré — seule source juridique pour PDF et DOCX."""

    title: str
    subtitle: str
    reference: str
    effective_date_fr: str
    particular_version: str
    pack_note_title: str
    pack_note: str
    parties: tuple[PartyColumn, PartyColumn]
    commercial_terms: tuple[CommercialRow, ...]
    key_principles_title: str
    key_principles: tuple[str, ...]
    clauses_intro: str
    clauses: tuple[ClauseBlock, ...]
    data_protection_roles: tuple[RoleRow, ...]
    data_protection_summary: str
    gps_summary: str
    providers_summary: str
    incorporated_documents: tuple[IncorporatedDoc, ...]
    acceptance_clause: str
    signature_intro: str
    signatures: tuple[SignatureBlock, ...]
    special_conditions: tuple[str, ...] = field(default_factory=tuple)

    def essential_text_blobs(self) -> list[str]:
        """Fragments textuels pour tests d'alignement PDF/DOCX."""
        blobs: list[str] = [
            self.title,
            self.reference,
            self.effective_date_fr,
            self.particular_version,
            self.pack_note,
            self.data_protection_summary,
            self.gps_summary,
            self.providers_summary,
            self.acceptance_clause,
            self.signature_intro,
            self.clauses_intro,
            self.key_principles_title,
        ]
        for col in self.parties:
            blobs.extend(col.lines)
        for row in self.commercial_terms:
            blobs.append(row.label)
            blobs.append(row.value)
        blobs.extend(self.key_principles)
        for clause in self.clauses:
            blobs.append(clause.title)
            blobs.append(clause.body)
        for role in self.data_protection_roles:
            blobs.append(role.treatment)
            blobs.append(role.role)
        for doc in self.incorporated_documents:
            blobs.append(doc.version)
            blobs.append(doc.sha256)
        for sig in self.signatures:
            blobs.append(sig.side)
            blobs.append(sig.name)
            if sig.power_attestation:
                blobs.append(sig.power_attestation)
        return [b for b in blobs if b]


def _fmt_date_fr(iso_or_str: str | None) -> str:
    text = (str(iso_or_str) if iso_or_str else "").strip()
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return text or "—"
    year, month, day = (int(match.group(i)) for i in (1, 2, 3))
    if not 1 <= month <= 12:
        return text
    day_txt = "1er" if day == 1 else str(day)
    return f"{day_txt} {_FR_MONTHS[month - 1]} {year}"


def _fmt_chf(value: Any) -> str:
    try:
        dec = Decimal(str(value).replace(",", ".")).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    if dec == dec.to_integral_value():
        return f"{int(dec):,}".replace(",", "'") + ".–"
    int_part = int(abs(dec))
    cents = int((abs(dec) - int_part) * 100)
    return f"{int_part:,}".replace(",", "'") + f".{cents:02d}"


def _pct(rate: Any) -> str:
    if rate is None or rate == "":
        return "—"
    try:
        n = float(str(rate).replace(",", "."))
        return f"{(n * 100):.2f}".rstrip("0").rstrip(".") + " %"
    except ValueError:
        return str(rate)


def _normalize_street(value: str) -> str:
    text = re.sub(r"\s*-\s*", "-", (value or "").strip())
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


def _fmt_ide(uid: str | None) -> str | None:
    text = (uid or "").strip()
    return text or None


def _title_as_entered(title: str) -> str:
    """Conserve le titre / pouvoir tel que saisi en admin (aucune réécriture)."""
    return (title or "").strip()


def _email(party: dict[str, Any], *, operator: bool) -> str:
    email = (party.get("contractual_email") or "").strip()
    if email:
        return email
    return DEFAULT_OPERATOR_EMAIL if operator else "à compléter"


def _is_sole(party: dict[str, Any]) -> bool:
    return (party.get("legal_form") or "") == LegalForm.SOLE_PROPRIETORSHIP.value


def _title_token(token: str) -> str:
    if "-" in token:
        return "-".join(_title_token(part) for part in token.split("-"))
    if not token:
        return token
    return token[:1].upper() + token[1:].lower()


def _format_person_name(name: str) -> str:
    """Prénom en casse titre, nom de famille en majuscules (usage contractuel)."""
    text = (name or "").strip()
    if not text:
        return "—"
    parts = [p for p in re.split(r"\s+", text) if p]
    if len(parts) == 1:
        return _title_token(parts[0])
    firsts = [_title_token(p) for p in parts[:-1]]
    last = parts[-1].upper()
    return " ".join([*firsts, last])


def _operator_name(party: dict[str, Any]) -> str:
    signatory = _format_person_name(party.get("signatory_name") or "")
    legal = _format_person_name(party.get("legal_name") or "")
    if _is_sole(party):
        return signatory if signatory != "—" else (legal or "—")
    return legal if legal != "—" else (signatory or "—")


def _partner_name(party: dict[str, Any]) -> str:
    name = (party.get("legal_name") or "").strip() or "—"
    label = (party.get("legal_form_label") or "").strip()
    form = party.get("legal_form") or ""
    if (
        form == LegalForm.SARL.value
        and label
        and "sàrl" not in name.lower()
        and "sarl" not in name.lower()
    ):
        return f"{name} {label}"
    return name


def _fmt_days_table(
    value: int, *, calendar: bool = False, working: bool = False
) -> str:
    unit = "jour" if abs(value) <= 1 else "jours"
    suffix = " calendaires" if calendar else (" ouvrables" if working else "")
    return f"{value} {unit}{suffix}"


def _party_column(
    *,
    header: str,
    party: dict[str, Any],
    operator: bool,
) -> PartyColumn:
    ide = _fmt_ide(party.get("uid_ide"))
    if operator:
        name = _operator_name(party)
        status = (
            "Indépendant exploitant la plateforme sous l'enseigne LIRIE"
            if _is_sole(party)
            else f"Forme juridique : {party.get('legal_form_label') or '—'}"
        )
        lines = [
            name,
            status,
            _fmt_address(party),
            f"Courriel contractuel : {_email(party, operator=True)}",
        ]
        if ide:
            lines.append(f"IDE : {ide}")
    else:
        name = _partner_name(party)
        signatory = _format_person_name(party.get("signatory_name") or "") or "—"
        title = _title_as_entered(party.get("signatory_title") or "")
        rep = f"Représenté par : {signatory}"
        if title:
            rep = f"{rep}, {title}"
        lines = [
            name,
            f"Forme juridique : {party.get('legal_form_label') or '—'}",
            _fmt_address(party),
        ]
        if ide:
            lines.append(f"IDE : {ide}")
        lines.append(f"Courriel contractuel : {_email(party, operator=False)}")
        lines.append(rep)
    return PartyColumn(header=header, lines=tuple(lines))


def _commercial_rows(commercial: dict[str, Any]) -> tuple[CommercialRow, ...]:
    commission_enabled = bool(commercial.get("lirie_commission_enabled", True))
    rate = _pct(commercial.get("commission_rate"))
    free_months = commercial.get("free_license_max_months")
    if free_months:
        portfolio = (
            f"Gratuit pendant {int(free_months)} mois ; aucun abonnement "
            "ne sera appliqué automatiquement à l'issue"
        )
    elif commercial.get("own_portfolio_billing_enabled", True):
        portfolio = "Selon grille / mode tarifaire figé au présent contrat"
    else:
        portfolio = "Non activé"
    support_rate = commercial.get("support_hourly_rate_default")
    support = (
        f"CHF {_fmt_chf(support_rate)} par heure, hors TVA, sur devis accepté "
        "ou demande expresse documentée du Partenaire"
        if support_rate not in (None, "")
        else "Selon conditions figées / devis"
    )
    payment = int(commercial.get("payment_terms_days") or 30)
    dispute = int(commercial.get("statement_dispute_days") or 10)
    rows = [
        CommercialRow(
            "Commission — courses transmises",
            (
                f"{rate} du montant HT définitif facturé au client"
                if commission_enabled
                else "Désactivée"
            ),
        ),
        CommercialRow("Licence — portefeuille propre", portfolio),
        CommercialRow("Support spécifique", support),
        CommercialRow("Relevé", "Mensuel – disponible dans LIRIE"),
        CommercialRow(
            "Paiement",
            f"{_fmt_days_table(payment)} à compter de la facture",
        ),
        CommercialRow(
            "Contestation du relevé",
            _fmt_days_table(dispute, working=True),
        ),
        CommercialRow(
            "Préavis de résiliation",
            _fmt_days_table(30, calendar=True),
        ),
        CommercialRow(
            "Non-contournement",
            "Pendant le contrat et 6 mois après sa fin",
        ),
    ]
    return tuple(rows)


def _penalty_phrase(commercial: dict[str, Any]) -> str:
    penalty = commercial.get("penalty") or {}
    multiplier = penalty.get("multiplier", PENALTY_MULTIPLIER)
    minimum = penalty.get("minimum", PENALTY_MINIMUM_CHF)
    currency = penalty.get("currency", PENALTY_CURRENCY)
    minimum_fmt = f"{int(minimum):,}".replace(",", "'")
    return (
        f"montant le plus élevé entre {multiplier}× les commissions éludées "
        f"et {currency} {minimum_fmt}.–"
    )


def build_particular_agreement_content(
    *,
    reference: str,
    parties: dict[str, Any],
    commercial: dict[str, Any],
    agreement_effective_from: str,
    general_terms_sha256: str,
    dpa_sha256: str,
    general_terms_version: str = GENERAL_TERMS_VERSION,
    dpa_version: str = DPA_VERSION,
) -> ParticularAgreementContent:
    operator = parties.get("operator") or {}
    partner = parties.get("partner") or {}
    authority = parties.get("signatory_authority_verification") or {}
    special = (commercial.get("contract_special_conditions") or "").strip()
    special_lines = tuple(line.strip() for line in special.splitlines() if line.strip())

    clauses = (
        ClauseBlock(
            "4.1  Licence et comptes",
            "LIRIE concède au Partenaire une licence professionnelle, non exclusive, "
            "non cessible et limitée à l'usage de la Plateforme pour son activité. Les "
            "comptes sont nominatifs ; le partage d'identifiants est interdit. Toute "
            "compromission ou perte d'accès doit être signalée sans délai. Les détails "
            "figurent dans les Conditions générales.",
        ),
        ClauseBlock(
            "4.2  Obligations du Partenaire",
            "Le Partenaire exécute personnellement ou par son personnel qualifié les "
            "transports acceptés, maintient assurances et autorisations, saisit des "
            "données exactes et complètes, déclare les montants HT définitifs ainsi que "
            "les avoirs et corrections, et paie les sommes dues aux échéances. Il "
            "clarifie l'identité du Client contractuel avant toute acceptation définitive.",
        ),
        ClauseBlock(
            "4.3  Transmission et responsabilité opérationnelle",
            "LIRIE fournit la Plateforme par obligation de moyens et n'est pas "
            "transporteur. Le Partenaire vérifie régulièrement son tableau de bord. Les "
            "journaux techniques constituent une présomption des opérations système, "
            "sans garantir une lecture humaine systématique. Le Partenaire demeure seul "
            "responsable de l'exécution du transport et de la relation avec le client. "
            "L'assistance ordinaire liée aux fonctionnalités existantes est incluse ; "
            "les interventions spécifiques sont régies par les Conditions générales et "
            "les conditions particulières ci-dessus.",
        ),
        ClauseBlock(
            "4.4  Non-contournement",
            "Les relations commerciales préexistantes indépendantes de LIRIE sont "
            "préservées. Toute suite, renouvellement ou répétition d'une demande "
            "initialement transmise ou gérée par LIRIE doit être enregistrée dans la "
            "Plateforme pendant le contrat et six mois après. En cas de violation "
            f"intentionnelle ou sciemment dissimulée : commissions éludées dues, plus "
            f"peine conventionnelle égale au {_penalty_phrase(commercial)}, sous "
            "réserve du pouvoir du juge.",
            emphasize=True,
        ),
        ClauseBlock(
            "4.5  Responsabilité et indemnisation",
            "Sous réserve du dol, de la faute grave et des responsabilités qui ne "
            "peuvent être exclues par la loi, LIRIE ne répond que des dommages directs, "
            "prouvés et raisonnablement prévisibles. Sont exclus les dommages "
            "indirects, le manque à gagner, la perte de chance, de chiffre d'affaires "
            "ou de clientèle. Sa responsabilité cumulée est plafonnée aux commissions "
            "et abonnements HT effectivement payés par le Partenaire durant les douze "
            "mois précédant le fait générateur. Les obligations de paiement, de "
            "confidentialité, de protection des données, d'indemnisation et de "
            "non-contournement du Partenaire demeurent régies par le présent contrat "
            "et les Conditions générales.",
            emphasize=True,
        ),
        ClauseBlock(
            "4.6  Confidentialité et propriété intellectuelle",
            "Chaque Partie protège les informations confidentielles de l'autre. La "
            "Plateforme (code, architecture, interfaces, marque) demeure la propriété "
            "exclusive de LIRIE. L'usage du nom ou du logo de l'autre Partie requiert un "
            "accord écrit. LIRIE peut produire des statistiques irréversiblement "
            "anonymisées.",
        ),
        ClauseBlock(
            "4.7  Suspension, durée et résiliation",
            "Le contrat est conclu pour une durée indéterminée. Chaque Partie peut y "
            "mettre fin moyennant un préavis écrit de trente (30) jours calendaires. "
            "LIRIE peut appliquer une suspension progressive en cas d'impayé et une "
            "suspension immédiate en cas de fraude ou d'atteinte grave à la sécurité. "
            "Après la fin du contrat, un accès en lecture seule est maintenu trente "
            "(30) jours pour l'export des données.",
            emphasize=True,
        ),
        ClauseBlock(
            "4.8  Droit applicable et dispositions finales",
            "Droit matériel suisse ; tribunaux ordinaires du canton de Genève. Force "
            "majeure, cession à une société ultérieurement constituée pour exploiter "
            "LIRIE (reprise des droits et obligations futurs et, dans la mesure légale "
            "et acceptée, déjà nés), invalidité partielle, absence de renonciation, "
            "notifications et signature électronique : selon les Conditions générales. "
            "La version française prévaut. En cas de contradiction, le présent Contrat "
            "particulier prévaut sur les Conditions générales pour les conditions "
            "commerciales et opérationnelles propres au Partenaire. L'Accord de "
            "traitement des données prévaut pour toute question relative à la "
            "protection des données. Les Conditions générales s'appliquent pour le "
            "surplus.",
        ),
    )

    roles = (
        RoleRow(
            "Portefeuille propre",
            "Partenaire responsable ; LIRIE sous-traitant",
        ),
        RoleRow(
            "Sécurité et comptes LIRIE",
            "LIRIE responsable distinct",
        ),
        RoleRow(
            "Transport et facturation du client",
            "Partenaire responsable",
        ),
        RoleRow(
            "Géolocalisation des chauffeurs",
            "Partenaire responsable ; LIRIE sous-traitant technique et "
            "responsable distinct pour les finalités propres précisées ci-dessous",
        ),
    )

    incorporated = (
        IncorporatedDoc(
            "Conditions générales partenaires LIRIE",
            general_terms_version,
            general_terms_sha256,
        ),
        IncorporatedDoc(
            "Accord de traitement des données LIRIE",
            dpa_version,
            dpa_sha256,
        ),
    )

    partner_signatory = _format_person_name(partner.get("signatory_name") or "") or "—"
    partner_title = _title_as_entered(partner.get("signatory_title") or "")
    co_name = None
    co_title = None
    if authority.get("signature_mode") == "collective" or authority.get(
        "co_signatory_required"
    ):
        co_name = _format_person_name(authority.get("co_signatory_name") or "") or None
        co_title = (
            _title_as_entered(authority.get("co_signatory_function") or "") or None
        )

    signatures = (
        SignatureBlock(
            side="Pour l'Exploitant – LIRIE",
            name=_operator_name(operator),
            title="Exploitant indépendant",
        ),
        SignatureBlock(
            side="Pour le Partenaire",
            name=partner_signatory,
            title=partner_title or "—",
            co_signatory_name=co_name,
            co_signatory_title=co_title,
            power_attestation=PARTNER_POWER_ATTESTATION,
        ),
    )

    return ParticularAgreementContent(
        title="CONTRAT PARTICULIER DE PARTENARIAT LIRIE",
        subtitle="Plateforme digitale LIRIE · www.lirie.ch",
        reference=reference,
        effective_date_fr=_fmt_date_fr(agreement_effective_from),
        particular_version=PARTICULAR_VERSION,
        pack_note_title="Documents contractuels et entrée en vigueur",
        pack_note=(
            "Le présent Contrat particulier, les Conditions générales partenaires "
            f"({general_terms_version}) et l'Accord de traitement des données "
            f"({dpa_version}, conservation {RETENTION_POLICY_VERSION}, "
            f"prestataires {SUBPROCESSORS_VERSION}) forment un ensemble contractuel "
            "indivisible. Les conditions financières prennent effet à la date "
            "d'effet commerciale. Les autres dispositions prennent effet à la date "
            "de la dernière signature, sauf mention contraire. L'Accord de "
            "traitement des données s'applique également aux traitements effectués "
            "dans le cadre de la relation depuis la date d'effet commerciale."
        ),
        parties=(
            _party_column(
                header="LIRIE – Exploitant",
                party=operator,
                operator=True,
            ),
            _party_column(header="Partenaire", party=partner, operator=False),
        ),
        commercial_terms=_commercial_rows(commercial),
        key_principles_title="Article 3 — Formation et traitement des courses",
        key_principles=(
            "LIRIE fournit une plateforme numérique et un intermédiaire technique ; "
            "elle n'est pas transporteur et ne conclut pas le contrat de transport.",
            "Le contrat de transport est formé à l'acceptation définitive du "
            "Partenaire dans la Plateforme.",
            "Toute modification substantielle de la demande requiert une nouvelle "
            "acceptation avant d'être opposable.",
            "La commission due à LIRIE est calculée sur le résultat financier "
            "définitif (montant HT après avoirs et corrections déclarés).",
            "La seule désignation d'un Payeur ou d'un Destinataire de facture ne "
            "crée aucune obligation de paiement ; un engagement séparé, exprès ou "
            "autrement démontrable, est requis.",
        ),
        clauses_intro=(
            "Les dispositions ci-après résument les engagements essentiels. "
            "Les Conditions générales partenaires précisent les modalités détaillées."
        ),
        clauses=clauses,
        data_protection_roles=roles,
        data_protection_summary=(
            "LIRIE traite les données confiées en qualité de sous-traitant "
            "conformément aux instructions du Partenaire, applique des mesures "
            "de sécurité appropriées, impose la confidentialité à ses "
            "collaborateurs, assiste le Partenaire dans l'exercice des droits "
            "des personnes et l'informe des incidents pertinents. Les traitements "
            "propres de LIRIE — sécurité de la Plateforme, facturation propre et "
            "défense de ses droits — sont réalisés sous sa responsabilité distincte."
        ),
        gps_summary=(
            "Le Partenaire informe préalablement ses chauffeurs des modalités et "
            "finalités de la géolocalisation. Celle-ci est limitée à l'organisation, "
            "la sécurité, la preuve d'exécution et aux finalités de l'Accord de "
            "traitement des données. Toute surveillance comportementale est exclue."
        ),
        providers_summary=(
            "Les prestataires techniques actifs (rôles, données, régions, "
            f"garanties) figurent dans l'Accord de traitement des données "
            f"{dpa_version}."
        ),
        incorporated_documents=incorporated,
        acceptance_clause=(
            "Le Partenaire reconnaît avoir reçu, lu et accepté, avant la signature, "
            "les documents identifiés ci-dessus. Ces documents font partie "
            "intégrante du présent contrat. Les empreintes numériques (SHA-256) "
            "des documents remis figurent dans le bordereau de remise et sont "
            "conservées par LIRIE à des fins de preuve et d'intégrité. Le "
            "bordereau de remise complète la preuve de composition du dossier "
            "sans se substituer à la présente acceptation."
        ),
        signature_intro=(
            "Le présent contrat peut être signé manuscritement ou électroniquement, "
            "en un ou plusieurs exemplaires ou copies ayant la même valeur juridique."
        ),
        signatures=signatures,
        special_conditions=special_lines,
    )
