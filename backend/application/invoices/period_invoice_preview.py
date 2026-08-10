"""Prévisualisation facture période — alignée sur l'éligibilité et les montants de génération."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import aliased

from application.invoices.institution_patient_resolution import (
    resolve_missing_institution_patient_ids,
)
from application.invoices.invoice_line_description import (
    booking_source_type_for_preview,
    build_invoice_line_description,
    build_invoice_line_description_clinic_monthly,
    resolve_patient_name_for_invoice,
)
from application.invoices.round_trip_billing_lock import (
    filter_bookings_open_for_new_invoice_line,
    round_trip_component_id_sets,
)
from application.invoices.round_trip_booking_pairs import (
    normalize_address_for_round_trip_comparison,
)
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from models import Booking, ClientStay, Company
from models.enums import BookingStatus
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from services.billing.clinic_s2_eligibility import clinic_s2_billed_to_company_predicate

# Taille exacte d'un A/R : 2 segments (aller + retour). Les composantes de taille
# > 2 sont des chaînes de trajets distincts qui doivent rester facturées en lignes
# individuelles, pas regroupées en un faux A/R.
_ROUND_TRIP_COMPONENT_SIZE = 2


def _is_strict_round_trip_pair(bookings_pair: list[Any]) -> bool:
    """Vrai A/R : 2 segments A→B et B→A (adresses inversées) ou liés explicitement.

    Refuse les chaînes A→B + B→C ou les paires « hub » qui partagent une seule
    extrémité — chacun doit rester facturé séparément.
    """
    if len(bookings_pair) != 2:
        return False
    a, b = bookings_pair
    # Lien explicite parent / retour : toujours considéré comme A/R commercial.
    pid_a = getattr(a, "parent_booking_id", None)
    pid_b = getattr(b, "parent_booking_id", None)
    try:
        if pid_a is not None and int(pid_a) == int(getattr(b, "id", -1)):
            return True
        if pid_b is not None and int(pid_b) == int(getattr(a, "id", -1)):
            return True
    except (TypeError, ValueError):
        pass
    # Sinon : inversion stricte des adresses normalisées (A→B et B→A).
    a_pick = normalize_address_for_round_trip_comparison(
        getattr(a, "pickup_location", "") or ""
    )
    a_drop = normalize_address_for_round_trip_comparison(
        getattr(a, "dropoff_location", "") or ""
    )
    b_pick = normalize_address_for_round_trip_comparison(
        getattr(b, "pickup_location", "") or ""
    )
    b_drop = normalize_address_for_round_trip_comparison(
        getattr(b, "dropoff_location", "") or ""
    )
    if not (a_pick and a_drop and b_pick and b_drop):
        return False
    return a_pick == b_drop and a_drop == b_pick


def _preview_base_amount_ht(booking: Any, billing_settings_dto: Any) -> Decimal:
    """HT ligne pour preview : montant facturable canonique."""
    from application.invoices.billable_amount import calculate_billable_booking_amount

    return calculate_billable_booking_amount(
        booking, billing_settings=billing_settings_dto
    ).amount_ht


def _scheduled_iso(booking: Any) -> str | None:
    st = getattr(booking, "scheduled_time", None)
    if st is None:
        return None
    if hasattr(st, "isoformat"):
        return st.isoformat()
    return str(st)


def _round_trip_leg_by_booking_id(bookings: list[Any]) -> dict[int, bool]:
    """True si le booking est l'aller ou le retour d'un A/R avec l'autre segment dans la même liste."""
    if not bookings:
        return {}
    ids = {int(b.id) for b in bookings}
    parents_with_return_in_list: set[int] = set()
    for b in bookings:
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None and int(pid) in ids:
            parents_with_return_in_list.add(int(pid))
    out: dict[int, bool] = {}
    for b in bookings:
        bid = int(b.id)
        is_ret = bool(getattr(b, "is_return", False))
        pid = getattr(b, "parent_booking_id", None)
        linked_to_parent = pid is not None and int(pid) in ids
        has_child_return = bid in parents_with_return_in_list
        out[bid] = is_ret or linked_to_parent or has_child_return
    return out


def _consolidate_period_preview_round_trip_rows(
    preview_lines: list[PeriodPreviewLine],
    bookings: list[Any],
    _billing_settings_dto: Any,
) -> list[PeriodPreviewLine]:
    """Une ligne par A/R (HT cumulé, description du segment principal)."""
    preview_ht_by_bid = {
        pl.booking_id: Decimal(str(pl.amount_ht)).quantize(Decimal("0.01"))
        for pl in preview_lines
    }

    def _amount_for_booking(b: Any) -> Decimal:
        return preview_ht_by_bid.get(int(b.id), Decimal("0"))

    comps = [
        c
        for c in round_trip_component_id_sets(
            bookings, amount_ht_fn=_amount_for_booking
        )
        if len(c) == _ROUND_TRIP_COMPONENT_SIZE
    ]
    if not comps:
        return preview_lines
    bookings_by_id = {int(b.id): b for b in bookings}
    by_id = {pl.booking_id: pl for pl in preview_lines}
    hidden_ids: set[int] = set()
    merged_rows: dict[int, PeriodPreviewLine] = {}
    for comp in comps:
        pls = [by_id[bid] for bid in comp if bid in by_id]
        if len(pls) != _ROUND_TRIP_COMPONENT_SIZE:
            continue
        comp_bookings = [bookings_by_id[bid] for bid in comp if bid in bookings_by_id]
        if not _is_strict_round_trip_pair(comp_bookings):
            # 2 trajets sur la meme journee mais qui ne forment pas un A/R
            # strict (ex. A->B puis B->C en chaine) : facturer separement.
            continue
        primary_pl = min(
            pls,
            key=lambda pl: ((pl.scheduled_at or ""), pl.booking_id),
        )
        pri = primary_pl.booking_id
        sum_ht = round(sum(float(p.amount_ht) for p in pls), 2)
        sum_origin = round(sum(float(p.origin_amount_ht) for p in pls), 2)
        _sched_candidates = [p.scheduled_at for p in pls if p.scheduled_at]
        sched = min(_sched_candidates) if _sched_candidates else None
        sched_end = max(_sched_candidates) if _sched_candidates else None
        if sched is not None and sched_end is not None and sched == sched_end:
            sched_end = None
        secondary_pl = next((p for p in pls if p.booking_id != pri), None)
        merged_rows[pri] = PeriodPreviewLine(
            booking_id=pri,
            scheduled_at=sched,
            amount_ht=sum_ht,
            origin_amount_ht=sum_origin,
            description=primary_pl.description,
            source_type=primary_pl.source_type,
            is_locked=any(p.is_locked for p in pls),
            already_invoiced=any(p.already_invoiced for p in pls),
            is_round_trip_leg=True,
            scheduled_at_end=sched_end,
            patient_name=primary_pl.patient_name,
            round_trip_partner_booking_id=(
                int(secondary_pl.booking_id) if secondary_pl is not None else None
            ),
            round_trip_primary_amount_ht=float(primary_pl.amount_ht),
            round_trip_partner_amount_ht=(
                float(secondary_pl.amount_ht) if secondary_pl is not None else None
            ),
            round_trip_partner_description=(
                secondary_pl.description if secondary_pl is not None else None
            ),
            round_trip_partner_scheduled_at=(
                secondary_pl.scheduled_at if secondary_pl is not None else None
            ),
            round_trip_primary_scheduled_at=primary_pl.scheduled_at,
        )
        for p in pls:
            if p.booking_id != pri:
                hidden_ids.add(p.booking_id)
    out: list[PeriodPreviewLine] = []
    emitted: set[int] = set()
    for pl in preview_lines:
        if pl.booking_id in hidden_ids:
            continue
        if pl.booking_id in merged_rows:
            if pl.booking_id not in emitted:
                out.append(merged_rows[pl.booking_id])
                emitted.add(pl.booking_id)
            continue
        out.append(pl)
    return out


@dataclass(frozen=True, slots=True)
class PeriodPreviewLine:
    booking_id: int
    scheduled_at: str | None
    amount_ht: float
    origin_amount_ht: float
    description: str
    source_type: str
    is_locked: bool
    already_invoiced: bool
    is_round_trip_leg: bool = False
    scheduled_at_end: str | None = None
    patient_name: str | None = None
    round_trip_partner_booking_id: int | None = None
    round_trip_primary_amount_ht: float | None = None
    round_trip_partner_amount_ht: float | None = None
    round_trip_partner_description: str | None = None
    round_trip_partner_scheduled_at: str | None = None
    round_trip_primary_scheduled_at: str | None = None


@dataclass(frozen=True, slots=True)
class PeriodPreviewResult:
    mode: str
    transports_count: int
    estimated_total: float
    currency: str
    warnings: list[str]
    preview_lines: tuple[PeriodPreviewLine, ...] = field(default_factory=tuple)
    estimated_subtotal_ht: float = 0.0
    estimated_vat_total: float = 0.0
    estimated_total_with_vat: float = 0.0


def build_period_invoice_preview(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    client_id: int | None = None,
    clinic_company_id: int | None = None,
    institution_patient_id: int | None = None,
    billing_party_id: int | None = None,
    include_line_details: bool = True,
) -> PeriodPreviewResult:
    """Aperçu read-only : exactement un de client_id (patient direct) ou clinic_company_id (S2).

    Filtres optionnels patient institutionnel : ``institution_patient_id`` / ``billing_party_id``.

    ``include_line_details=False`` : agrégats seuls (registre opportunités) — pas de
    résolution nominative par ligne (évite le N+1 clients).
    """
    if bool(client_id) == bool(clinic_company_id):
        raise ValueError(
            "Fournir exactement un des paramètres: client_id ou clinic_company_id"
        )

    warnings: list[str] = []
    settings_repo = CompanyBillingSettingsRepository()
    billing_settings_dto = settings_repo.find_or_create(company_id)
    calc = InvoiceCalculator()

    vat_rate_setting = billing_settings_dto.vat_rate
    vat_applicable = (
        bool(billing_settings_dto.vat_applicable) and vat_rate_setting is not None
    )
    default_vat_rate = Decimal("0")
    if vat_applicable:
        try:
            default_vat_rate = Decimal(str(vat_rate_setting)).quantize(Decimal("0.01"))
            if default_vat_rate <= 0:
                vat_applicable = False
        except Exception:
            vat_applicable = False
            default_vat_rate = Decimal("0")

    preview_lines_out: list[PeriodPreviewLine] = []

    if client_id is not None:
        crepo = ClientRepository()
        if not crepo.find_model_by_id_and_company(int(client_id), company_id):
            raise ValueError("Client introuvable pour cette entreprise")

        repo = BookingRepository()
        eligible_b = repo.find_models_eligible_for_billing_period_by_company_and_client(
            company_id,
            int(client_id),
            period_year,
            period_month,
            billed_to_type="patient",
        )
        bookings = filter_bookings_open_for_new_invoice_line(
            eligible_b,
            amount_ht_fn=lambda b: _preview_base_amount_ht(b, billing_settings_dto),
        )
        if institution_patient_id is not None:
            # Le patient prime : un même patient peut porter plusieurs
            # BillingParty historiques, on ne filtre donc pas sur le payeur.
            resolve_missing_institution_patient_ids(bookings)
            bookings = [
                b
                for b in bookings
                if getattr(b, "institution_patient_id", None)
                == int(institution_patient_id)
            ]
        elif billing_party_id is not None:
            bookings = [
                b
                for b in bookings
                if getattr(b, "billing_party_id", None) == int(billing_party_id)
            ]
        client = crepo.find_model_by_id_with_user(int(client_id), company_id)
        patient_name = (
            resolve_patient_name_for_invoice(client, bookings)
            if include_line_details
            else None
        )
        rt_map = _round_trip_leg_by_booking_id(bookings)
        # Une seule requête pour tous les clients des lignes (évite N+1).
        clients_by_id: dict[int, Any] = {}
        if include_line_details:
            clients_by_id = crepo.find_models_by_ids_and_company_with_user(
                {int(b.client_id) for b in bookings if b.client_id is not None},
                company_id,
            )

        gross = Decimal("0.00")
        vat_sum = Decimal("0.00")
        tw_sum = Decimal("0.00")

        for b in bookings:
            amt = _preview_base_amount_ht(b, billing_settings_dto)
            gross += amt
            if amt == 0:
                warnings.append(
                    f"Transport #{getattr(b, 'id', '?')} sans montant (un estimé peut s'appliquer à la génération)"
                )

            va, tw = calc.calculate_vat(
                amt, default_vat_rate if vat_applicable else Decimal("0")
            )
            vat_sum += va
            tw_sum += tw

            if not include_line_details:
                continue

            desc = build_invoice_line_description(
                b,
                patient_name=patient_name,
                bill_to_client_id=None,
                clinic_company_id=None,
                billing_party_id=None,
                booking_for_cancellation=b,
                description_builder=None,
            )
            locked = bool(getattr(b, "invoice_line_id", None))
            b_client = (
                clients_by_id.get(int(b.client_id)) if b.client_id is not None else None
            )
            row_patient = resolve_patient_name_for_invoice(b_client, [b])
            if not row_patient:
                row_patient = (getattr(b, "customer_name", None) or "").strip() or None
            preview_lines_out.append(
                PeriodPreviewLine(
                    booking_id=int(b.id),
                    scheduled_at=_scheduled_iso(b),
                    amount_ht=float(amt),
                    origin_amount_ht=float(amt),
                    description=desc,
                    source_type=booking_source_type_for_preview(b),
                    is_locked=locked,
                    already_invoiced=bool(getattr(b, "invoice_line_id", None)),
                    is_round_trip_leg=bool(rt_map.get(int(b.id), False)),
                    patient_name=row_patient,
                )
            )

        if include_line_details:
            preview_lines_out = _consolidate_period_preview_round_trip_rows(
                preview_lines_out, bookings, billing_settings_dto
            )

        if len(bookings) == 0:
            warnings.insert(
                0,
                (
                    "Aucun transport à facturer pour ce patient sur cette période "
                    + "(vérifiez le mois, le payeur facturé « patient » et les courses déjà facturées)."
                ),
            )

        total = round_to_5_cents(gross)
        total_tw = round_to_5_cents(tw_sum)
        total_vat = round_to_5_cents(vat_sum)

        return PeriodPreviewResult(
            mode="standard",
            transports_count=len(bookings),
            estimated_total=float(total),
            currency="CHF",
            warnings=warnings[:25],
            preview_lines=tuple(preview_lines_out),
            estimated_subtotal_ht=float(total),
            estimated_vat_total=float(total_vat),
            estimated_total_with_vat=float(total_tw),
        )

    # --- S2 : même filtre que preview clinique / génération ---
    if clinic_company_id is None:
        raise ValueError("clinic_company_id requis pour l'aperçu clinique")
    ccid = int(clinic_company_id)
    if not Company.query.filter_by(id=ccid).first():
        raise ValueError("Clinique (entreprise) introuvable")

    PERIOD_MONTH_THRESHOLD = 12
    start_date = datetime(period_year, period_month, 1)
    if period_month == PERIOD_MONTH_THRESHOLD:
        end_date = datetime(period_year + 1, 1, 1)
    else:
        end_date = datetime(period_year, period_month + 1, 1)

    target_statuses = [
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
    ]
    from sqlalchemy import exists

    stay_overlaps_booking = exists().where(
        ClientStay.client_id == Booking.client_id,
        ClientStay.company_id == ccid,
        ClientStay.status == "active",
        ClientStay.start_date <= Booking.scheduled_time,
        or_(
            ClientStay.end_date.is_(None),
            ClientStay.end_date >= Booking.scheduled_time,
        ),
    )
    canceled_eligible = (
        (Booking.status == BookingStatus.CANCELED.value)
        & (Booking.amount > 0)
        & (
            (Booking.is_cancellation_billable == True)  # noqa: E712
            | (
                Booking.billing_override_reason.isnot(None)
                & (Booking.billing_override_reason != "")
            )
        )
        & stay_overlaps_booking
        & (Booking.is_return == False)  # noqa: E712
    )
    eligible_query = Booking.query.filter(
        Booking.company_id == company_id,
        clinic_s2_billed_to_company_predicate(ccid, company_id),
        Booking.billed_to_type == "clinic",
        or_(
            Booking.status.in_(target_statuses),
            canceled_eligible,
        ),
        Booking.invoice_line_id.is_(None),
        Booking.scheduled_time >= start_date,
        Booking.scheduled_time < end_date,
    )
    ParentB = aliased(Booking)
    eligible_query = eligible_query.outerjoin(
        ParentB, ParentB.id == Booking.parent_booking_id
    ).filter(
        or_(
            Booking.is_return == False,  # noqa: E712
            ParentB.id.is_(None),
            ParentB.status != BookingStatus.CANCELED.value,
        )
    )
    eligible_bookings = eligible_query.order_by(Booking.scheduled_time.asc()).all()
    rt_map_s2 = _round_trip_leg_by_booking_id(eligible_bookings)
    crepo = ClientRepository()
    # Batch : une requête client (+ user) pour toute la période clinique (Sentry N+1).
    clients_by_id_s2: dict[int, Any] = {}
    if include_line_details:
        clients_by_id_s2 = crepo.find_models_by_ids_and_company_with_user(
            {int(b.client_id) for b in eligible_bookings if b.client_id is not None},
            company_id,
        )

    gross_s2 = Decimal("0.00")
    vat_sum_s2 = Decimal("0.00")
    tw_sum_s2 = Decimal("0.00")

    for b in eligible_bookings:
        amt = _preview_base_amount_ht(b, billing_settings_dto)
        gross_s2 += amt
        if amt == 0:
            warnings.append(
                f"Transport #{getattr(b, 'id', '?')} sans montant (un estimé peut s'appliquer à la génération)"
            )

        va, tw = calc.calculate_vat(
            amt, default_vat_rate if vat_applicable else Decimal("0")
        )
        vat_sum_s2 += va
        tw_sum_s2 += tw

        if not include_line_details:
            continue

        desc = build_invoice_line_description_clinic_monthly(
            b, description_builder=None
        )

        locked = bool(getattr(b, "invoice_line_id", None))
        b_client = (
            clients_by_id_s2.get(int(b.client_id)) if b.client_id is not None else None
        )
        from application.invoices.invoice_line_description import (
            resolve_s2_clinic_line_patient_name,
        )

        row_patient = resolve_s2_clinic_line_patient_name(b_client, b)
        preview_lines_out.append(
            PeriodPreviewLine(
                booking_id=int(b.id),
                scheduled_at=_scheduled_iso(b),
                amount_ht=float(amt),
                origin_amount_ht=float(amt),
                description=desc,
                source_type=booking_source_type_for_preview(b),
                is_locked=locked,
                already_invoiced=bool(getattr(b, "invoice_line_id", None)),
                is_round_trip_leg=bool(rt_map_s2.get(int(b.id), False)),
                patient_name=row_patient,
            )
        )

    if include_line_details:
        preview_lines_out = _consolidate_period_preview_round_trip_rows(
            preview_lines_out, eligible_bookings, billing_settings_dto
        )

    if len(eligible_bookings) == 0:
        warnings.insert(
            0,
            (
                "Aucun transport clinique à facturer sur cette période "
                + "(transports `billed_to_type=clinic` pour cette clinique, non encore facturés)."
            ),
        )

    total = round_to_5_cents(gross_s2)
    total_tw = round_to_5_cents(tw_sum_s2)
    total_vat = round_to_5_cents(vat_sum_s2)

    return PeriodPreviewResult(
        mode="clinic_monthly",
        transports_count=len(eligible_bookings),
        estimated_total=float(total),
        currency="CHF",
        warnings=warnings[:25],
        preview_lines=tuple(preview_lines_out),
        estimated_subtotal_ht=float(total),
        estimated_vat_total=float(total_vat),
        estimated_total_with_vat=float(total_tw),
    )


def preview_line_to_dict(pl: PeriodPreviewLine) -> dict[str, Any]:
    booking_ids = [int(pl.booking_id)]
    if pl.round_trip_partner_booking_id is not None:
        partner = int(pl.round_trip_partner_booking_id)
        if partner not in booking_ids:
            booking_ids.append(partner)
    unit_type = "round_trip" if pl.is_round_trip_leg else "single"
    d: dict[str, Any] = {
        "booking_id": pl.booking_id,
        "primary_booking_id": pl.booking_id,
        "booking_ids": booking_ids,
        "unit_type": unit_type,
        "segments_count": len(booking_ids),
        "preview_row_id": (
            f"unit:round_trip:{booking_ids[0]}:{booking_ids[1]}"
            if len(booking_ids) == 2
            else f"unit:single:{pl.booking_id}"
        ),
        "scheduled_at": pl.scheduled_at,
        "amount_ht": pl.amount_ht,
        "origin_amount_ht": pl.origin_amount_ht,
        "description": pl.description,
        "source_type": pl.source_type,
        "is_locked": pl.is_locked,
        "already_invoiced": pl.already_invoiced,
        "is_round_trip_leg": pl.is_round_trip_leg,
    }
    if pl.patient_name:
        d["patient_name"] = pl.patient_name
    if pl.round_trip_partner_booking_id is not None:
        d["round_trip_partner_booking_id"] = pl.round_trip_partner_booking_id
    if pl.round_trip_primary_amount_ht is not None:
        d["round_trip_primary_amount_ht"] = pl.round_trip_primary_amount_ht
    if pl.round_trip_partner_amount_ht is not None:
        d["round_trip_partner_amount_ht"] = pl.round_trip_partner_amount_ht
    if pl.round_trip_partner_description:
        d["round_trip_partner_description"] = pl.round_trip_partner_description
    if pl.round_trip_partner_scheduled_at:
        d["round_trip_partner_scheduled_at"] = pl.round_trip_partner_scheduled_at
    if pl.round_trip_primary_scheduled_at:
        d["round_trip_primary_scheduled_at"] = pl.round_trip_primary_scheduled_at
    if pl.scheduled_at_end:
        d["scheduled_at_end"] = pl.scheduled_at_end
    return d


def preview_to_dict(p: PeriodPreviewResult) -> dict[str, Any]:
    return {
        "mode": p.mode,
        "transports_count": p.transports_count,
        "estimated_total": p.estimated_total,
        "currency": p.currency,
        "warnings": p.warnings,
        "preview_lines": [preview_line_to_dict(x) for x in p.preview_lines],
        "estimated_subtotal_ht": p.estimated_subtotal_ht,
        "estimated_vat_total": p.estimated_vat_total,
        "estimated_total_with_vat": p.estimated_total_with_vat,
    }
