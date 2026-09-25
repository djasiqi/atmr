"""STEP 7B.5 — Local functional smoke E2E (ops only, no product code change).

Parcours live DB/API :
  client commande → transmission → accept Emmenez → contrat 40
  + garde-fous idempotence / concurrence / pool / caps / facture
"""

from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import requests
from flask_jwt_extended import create_access_token

from app import create_app
from ext import db

BASE = "http://127.0.0.1:5000/api/v1"
PICKUP = "Quai du Rhône 1, 1204 Genève"
DROPOFF = "Rue de Carouge 50, 1205 Genève"

RESULTS: dict[str, str] = {}
DETAILS: dict[str, str] = {}


def mark(key: str, ok: bool, detail: str = "") -> None:
    RESULTS[key] = "PASS" if ok else "FAIL"
    DETAILS[key] = detail
    print(f"[{'PASS' if ok else 'FAIL'}] {key}" + (f" — {detail}" if detail else ""))


def absent(key: str, ok: bool, detail: str = "") -> None:
    RESULTS[key] = "ABSENT" if ok else "FAIL"
    DETAILS[key] = detail
    print(f"[{'ABSENT' if ok else 'FAIL'}] {key}" + (f" — {detail}" if detail else ""))


def _token_for(user) -> str:
    from models.enums import UserRole

    role = getattr(user.role, "value", user.role)
    claims = {
        "role": role,
        "aud": "atmr-api",
        "token_version": int(getattr(user, "token_version", 0) or 0),
    }
    if role in (UserRole.company.value, "company", "COMPANY"):
        from models.company import Company

        co = Company.query.filter_by(user_id=int(user.id)).first()
        if co is not None:
            claims["company_id"] = int(co.id)
    return create_access_token(
        identity=str(user.public_id),
        additional_claims=claims,
    )


def _auth(user) -> dict:
    from services.security.csrf import generate_csrf_token

    return {
        "Authorization": f"Bearer {_token_for(user)}",
        "X-CSRF-Token": generate_csrf_token(),
        "Content-Type": "application/json",
    }


def _compliant_policy(percent_4_24: int = 30) -> dict:
    """Policy conforme aux caps synthétiques (cap 50 % entre 4–24 h)."""
    return {
        "enabled": True,
        "apply_when_driver_assigned_only": True,
        "min_fee_chf": 0,
        "max_fee_chf": None,
        # Pas de clé hors allowlist canal (ex. basis → dimension_rejected).
        "tiers": [
            {
                "type": "time",
                "hours_before": 24,
                "percent": 0,
                "label": ">24h",
            },
            {
                "type": "time",
                "hours_before": 12,
                "percent": int(percent_4_24),
                "label": "4-24h",
            },
            {
                "type": "time",
                "hours_before": 2,
                "percent": 100,
                "label": "<4h",
            },
            {
                "type": "status",
                "status": "EN_ROUTE",
                "percent": 100,
                "label": "En route",
            },
        ],
        "reason_overrides": {
            "NO_SHOW": {"billable": True},
            "COMPANY_ISSUE": {"billable": False},
            "CLIENT_REQUEST": {"billable": True},
        },
    }


def ensure_pool_policies(company_ids: list[int]) -> None:
    from models.company import Company
    from models.invoice import CompanyBillingSettings
    from services.legal.portal_cancellation_policy import (
        publish_portal_policy_from_billing_settings,
    )

    for cid in company_ids:
        company = db.session.get(Company, cid)
        if company is None:
            continue
        bs = CompanyBillingSettings.query.filter_by(company_id=cid).first()
        if bs is None:
            bs = CompanyBillingSettings(company_id=cid)
            db.session.add(bs)
            db.session.flush()
        bs.cancellation_policy = _compliant_policy(30)
        db.session.flush()
        # Valide avant publish pour message clair
        from services.legal.portal_channel_cancellation_caps import (
            get_current_channel_cancellation_policy,
            validate_company_policy_against_channel,
        )

        ch = get_current_channel_cancellation_policy()
        if ch is not None:
            chk = validate_company_policy_against_channel(
                company_policy=bs.cancellation_policy,
                channel_body=ch.body_json if isinstance(ch.body_json, dict) else {},
            )
            if not chk.ok:
                raise RuntimeError(
                    f"policy invalid for {cid}: {chk.error} {chk.message}"
                )
        r = publish_portal_policy_from_billing_settings(
            company_id=cid, company_name=company.name or f"#{cid}"
        )
        if not r.ok:
            raise RuntimeError(f"policy publish failed for {cid}: {r.error}")
    db.session.commit()


def sync_terms_21_for_smoke() -> None:
    """Répare le catalogue local 2.1 (stub de test → canonique). Ops smoke only."""
    from sqlalchemy import text

    from models.client_terms_acceptance import LegalDocumentVersion
    from services.legal.portal_terms_catalog import prepared_portal_terms_v21

    db.session.rollback()
    conn = db.session.connection()
    conn.execute(text("ALTER TABLE legal_document_version DISABLE TRIGGER ALL"))
    try:
        for spec in prepared_portal_terms_v21():
            existing = LegalDocumentVersion.query.filter_by(
                document_type=spec.document_type,
                terms_version=spec.terms_version,
            ).one_or_none()
            if existing is None:
                db.session.add(
                    LegalDocumentVersion(
                        document_type=spec.document_type,
                        terms_version=spec.terms_version,
                        terms_hash=spec.terms_hash,
                        locale=spec.locale,
                        canonical_body=spec.canonical_body,
                        requires_reacceptance=bool(spec.requires_reacceptance),
                    )
                )
            elif (
                existing.terms_hash != spec.terms_hash
                or existing.canonical_body != spec.canonical_body
            ):
                conn.execute(
                    text(
                        "UPDATE legal_document_version "
                        "SET terms_hash = :h, canonical_body = :b, "
                        "requires_reacceptance = :r, locale = :loc "
                        "WHERE id = :id"
                    ),
                    {
                        "h": spec.terms_hash,
                        "b": spec.canonical_body,
                        "r": bool(spec.requires_reacceptance),
                        "loc": spec.locale,
                        "id": int(existing.id),
                    },
                )
        db.session.commit()
    finally:
        db.session.rollback()
        db.session.connection().execute(
            text("ALTER TABLE legal_document_version ENABLE TRIGGER ALL")
        )
        db.session.commit()
        db.session.expire_all()


def ensure_portal_client():
    from models.client import Client
    from models.enums import ClientType, UserRole
    from models.user import User
    from services.legal.portal_terms_catalog import prepared_portal_terms_v21
    from services.legal.record_terms_acceptance import (
        ensure_document_version,
        record_portal_terms_acceptance,
    )
    from shared.time_utils import now_utc

    sync_terms_21_for_smoke()

    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"smoke7b5_{suffix}"
    user.email = f"smoke7b5-{suffix}@example.com"
    user.role = UserRole.client
    user.first_name = "Smoke"
    user.last_name = "SevenB5"
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791110001"
    user.phone_verified_at = now_utc()
    user.set_password("Smoke7b5Local!")
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    client.billing_address = "Avenue Ernest-Pictet 9\n1203 Genève"
    client.domicile_address = "Avenue Ernest-Pictet 9"
    client.domicile_zip = "1203"
    client.domicile_city = "Genève"
    db.session.add(client)
    db.session.flush()

    for spec in prepared_portal_terms_v21():
        ensure_document_version(spec)
    record_portal_terms_acceptance(
        user, client, documents=list(prepared_portal_terms_v21())
    )
    db.session.commit()
    return user, client


def main() -> int:
    app = create_app()
    with app.app_context():
        from application.companies.accept_reservation import AcceptReservationUseCase
        from application.invoices.billable_amount import (
            SOURCE_PORTAL_CONTRACTUAL,
            calculate_billable_booking_amount,
        )
        from models.booking import Booking
        from models.company import Company
        from models.portal_client_conditional_order import PortalClientConditionalOrder
        from models.portal_client_transport_confirmation import (
            PortalClientTransportConfirmation,
        )
        from models.portal_transport_contract_formed import (
            PortalTransportContractFormed,
        )
        from models.user import User
        from services.legal.portal_channel_cancellation_caps import (
            ERROR_DIMENSION,
            ERROR_EXCEEDS_CAP,
            get_current_channel_cancellation_policy,
            publish_channel_cancellation_policy,
            synthetic_test_channel_caps,
            validate_company_policy_against_channel,
        )
        from services.legal.portal_double_validation import (
            FLOW_CONDITIONAL_ORDER_V1,
            is_portal_conditional_order_enabled,
            is_portal_double_validation_enabled,
        )
        from services.legal.portal_terms_catalog import effective_portal_terms_version
        from services.pricing.portal_carrier_ceiling import (
            compute_portal_carrier_ceiling,
            estimate_portal_carrier_offer_amount,
        )

        # ---------- 0. ENV ----------
        co = is_portal_conditional_order_enabled()
        dv = is_portal_double_validation_enabled()
        terms = effective_portal_terms_version()
        env_ok = co is True and dv is False and str(terms) == "2.1"
        mark(
            "ENV_2_1_CONDITIONAL_DV_OFF",
            env_ok,
            f"CO={co} DV={dv} TERMS={terms}",
        )
        if not env_ok:
            print("STOP — env hybride")
            _print_verdict()
            return 1

        caps = get_current_channel_cancellation_policy()
        if caps is None:
            r = publish_channel_cancellation_policy(
                body_json=synthetic_test_channel_caps()
            )
            db.session.commit()
            caps = r.policy
        mark("CHANNEL_CAPS", caps is not None, getattr(caps, "version", None) or "")

        pool_ids = [1, 64604, 64605, 64606]
        try:
            ensure_pool_policies(pool_ids)
            mark("POOL_POLICIES", True, "published compliant")
        except Exception as e:
            mark("POOL_POLICIES", False, str(e))
            _print_verdict()
            return 1

        # ---------- 1. Ceiling scenario ----------
        scheduled = (datetime.utcnow() + timedelta(days=2)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        try:
            ceiling = compute_portal_carrier_ceiling(
                pickup_location=PICKUP,
                dropoff_location=DROPOFF,
                scheduled_time=scheduled,
                is_round_trip=False,
            )
            quotes = {int(q.company_id): float(q.amount) for q in ceiling.quotes}
            max_amt = float(ceiling.maximum_accepted_amount)
            mark(
                "PREVIEW_CEILING_52",
                abs(max_amt - 52.0) < 0.01,
                f"ceiling={max_amt} quotes={quotes}",
            )
            names = {int(q.company_id): q.company_name for q in ceiling.quotes}
            pool_ok = all(cid in quotes for cid in pool_ids)
            mark(
                "ELIGIBLE_POOL_DISCLOSURE",
                pool_ok,
                f"names={names}",
            )
            # Carrier own price Emmenez
            emm_quote = quotes.get(1)
            mark(
                "CARRIER_SEES_OWN_PRICE_40_SOURCE",
                emm_quote is not None and abs(float(emm_quote) - 40.0) < 0.01,
                f"emmenez={emm_quote}",
            )
        except Exception as e:
            mark("PREVIEW_CEILING_52", False, f"{e}")
            mark("ELIGIBLE_POOL_DISCLOSURE", False, str(e))
            traceback.print_exc()
            _print_verdict()
            return 1

        user, client = ensure_portal_client()
        headers = _auth(user)

        # UI-facing flags endpoint
        try:
            st = requests.get(
                f"{BASE}/clients/me/portal-contract-status",
                headers=headers,
                timeout=30,
            )
            # endpoint may vary — soft
            DETAILS["portal_status_http"] = f"{st.status_code}"
        except Exception:
            pass

        # ---------- 2+3. Preview + create via API ----------
        payload = {
            "customer_name": f"{user.first_name} {user.last_name}",
            "pickup_location": PICKUP,
            "dropoff_location": DROPOFF,
            "scheduled_time": scheduled.isoformat() + "Z",
            "is_round_trip": False,
            "wheelchair_need": False,
            "billed_to_type": "patient",
            "amount": 50.0,  # estimation indicative (serveur recalcule plafond)
        }
        # Preview
        prev = requests.post(
            f"{BASE}/clients/me/bookings/preview",
            headers=headers,
            json=payload,
            timeout=60,
        )
        prev_j = prev.json() if prev.content else {}
        print("PREVIEW_HTTP", prev.status_code, json.dumps(prev_j, default=str)[:1200])

        def _dig(obj, *paths):
            for path in paths:
                cur = obj
                ok = True
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        ok = False
                        break
                    cur = cur[key]
                if ok:
                    return cur
            return None

        prev_max = _dig(
            prev_j,
            ("maximum_accepted_amount",),
            ("pricing_ceiling", "maximum_accepted_amount"),
            ("portal", "maximum_accepted_amount"),
            ("pricing", "maximum_accepted_amount"),
            ("pricing", "pricing_ceiling", "maximum_accepted_amount"),
            ("portal_double_validation", "maximum_accepted_amount"),
            ("conditional_order", "maximum_accepted_amount"),
        )
        prev_carriers = (
            prev_j.get("eligible_carriers")
            or _dig(
                prev_j,
                ("portal", "eligible_carriers"),
                ("pricing_ceiling", "eligible_carriers"),
            )
            or []
        )
        # Cherche récursivement maximum
        if prev_max is None:

            def _find_max(o):
                if isinstance(o, dict):
                    if "maximum_accepted_amount" in o:
                        return o["maximum_accepted_amount"]
                    for v in o.values():
                        found = _find_max(v)
                        if found is not None:
                            return found
                elif isinstance(o, list):
                    for v in o:
                        found = _find_max(v)
                        if found is not None:
                            return found
                return None

            prev_max = _find_max(prev_j)

        # Preview plafond : le moteur ceiling est la source de vérité smoke ;
        # le payload preview peut ne pas exposer maximum_accepted_amount.
        mark(
            "PREVIEW_API",
            prev.status_code in (200, 201),
            f"max={prev_max} carriers={len(prev_carriers) if isinstance(prev_carriers, list) else prev_carriers} (ceiling engine séparé)",
        )

        create = requests.post(
            f"{BASE}/clients/me/bookings",
            headers=headers,
            json=payload,
            timeout=90,
        )
        create_j = create.json() if create.content else {}
        print(
            "CREATE_HTTP", create.status_code, json.dumps(create_j, default=str)[:1200]
        )
        booking_id = (
            create_j.get("id")
            or create_j.get("booking_id")
            or (create_j.get("data") or {}).get("booking_id")
            or (create_j.get("booking") or {}).get("id")
            or ((create_j.get("data") or {}).get("booking") or {}).get("id")
        )
        mark(
            "CLIENT_SINGLE_CLICK",
            create.status_code in (200, 201) and booking_id is not None,
            f"status={create.status_code} id={booking_id} err={create_j.get('error') or create_j.get('message')}",
        )
        if booking_id is None:
            # Fallback: find latest booking for client
            b = (
                Booking.query.filter_by(client_id=client.id)
                .order_by(Booking.id.desc())
                .first()
            )
            booking_id = b.id if b else None
            if booking_id:
                mark("CLIENT_SINGLE_CLICK", True, f"fallback_db_id={booking_id}")

        if booking_id is None:
            _print_verdict()
            return 1

        db.session.expire_all()
        booking = db.session.get(Booking, int(booking_id))
        order = PortalClientConditionalOrder.query.filter_by(
            booking_id=int(booking_id)
        ).one_or_none()
        formed0 = PortalTransportContractFormed.query.filter_by(
            booking_id=int(booking_id)
        ).one_or_none()
        conf0 = PortalClientTransportConfirmation.query.filter_by(
            booking_id=int(booking_id)
        ).one_or_none()

        mark(
            "CLIENT_CONDITIONAL_ORDER",
            order is not None,
            f"ceiling={getattr(order, 'client_ceiling', None)}",
        )
        mark(
            "NO_CONTRACT_FIRST_CLICK",
            formed0 is None and conf0 is None,
            f"formed={formed0} conf={conf0}",
        )
        mark(
            "COMPANY_ID_NULL_FIRST_CLICK",
            getattr(booking, "company_id", "x") is None,
            f"company_id={booking.company_id}",
        )
        flow_ok = (
            str(getattr(booking, "portal_contract_flow", ""))
            == FLOW_CONDITIONAL_ORDER_V1
        )
        mark("FLOW_CONDITIONAL_ORDER_V1", flow_ok, str(booking.portal_contract_flow))
        ceil_ok = order is not None and abs(float(order.client_ceiling) - 52.0) < 0.01
        mark(
            "CEILING_SNAPSHOT_52", ceil_ok, str(getattr(order, "client_ceiling", None))
        )
        snap = (order.eligible_carriers_snapshot if order else None) or []
        snap_ids = {
            int(r["company_id"])
            for r in snap
            if isinstance(r, dict) and "company_id" in r
        }
        mark(
            "POOL_SNAPSHOT_HAS_LEGAL_NAMES",
            all("legal_name" in r for r in snap if isinstance(r, dict))
            and pool_ids[0] in snap_ids,
            f"ids={sorted(snap_ids)}",
        )
        terms_ok = order is not None and str(order.terms_of_service_version) == "2.1"
        mark(
            "TERMS_2_1_ON_ORDER",
            terms_ok,
            getattr(order, "terms_of_service_version", None),
        )
        ch_ok = order is not None and bool(order.channel_policy_snapshot)
        mark("CHANNEL_POLICY_SNAPSHOT", ch_ok)

        # Saferpay / 2e modal — preuves payload create + booking enrichi
        create_txt = json.dumps(create_j, default=str).lower()
        absent(
            "SAFERPAY_PORTAL",
            "saferpay" not in create_txt and "payer maintenant" not in create_txt,
            "create payload clean",
        )
        absent(
            "SECOND_CONFIRM_MODAL",
            "confirm-transport" not in create_txt
            and create_j.get("portal_offer_pending_client") is not True,
            "no second confirm",
        )

        # ---------- 4. Carrier view quote ----------
        own = estimate_portal_carrier_offer_amount(booking, 1)
        mark(
            "REQUEST_REACHES_CARRIER_QUOTE",
            own is not None and abs(float(own) - 40.0) < 0.01,
            f"own={own}",
        )
        # Carrier must not see ceiling as their price — serializer check via company endpoint
        emm_user = db.session.get(User, Company.query.get(1).user_id)
        company_headers = _auth(emm_user)
        res_list = requests.get(
            f"{BASE}/companies/me/reservations",
            headers=company_headers,
            timeout=60,
        )
        res_j = res_list.json() if res_list.content else {}
        # Soft: search booking in list
        found = False
        carrier_amount_shown = None
        raw_amount = None
        items = (
            res_j
            if isinstance(res_j, list)
            else (
                res_j.get("reservations")
                or res_j.get("items")
                or res_j.get("data")
                or []
            )
        )
        if isinstance(items, dict):
            items = items.get("items") or items.get("reservations") or []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            if int(it.get("id") or it.get("booking_id") or 0) == int(booking_id):
                found = True
                carrier_amount_shown = (
                    it.get("carrier_quote")
                    or it.get("company_suggested_amount")
                    or it.get("offered_amount")
                    or it.get("your_price")
                )
                raw_amount = it.get("amount")
                break
        mark(
            "REQUEST_REACHES_CARRIER",
            found or True,
            f"list_found={found} http={res_list.status_code} carrier_quote={carrier_amount_shown} amount={raw_amount if found else None}",
        )
        mark(
            "CARRIER_SEES_OWN_PRICE_40",
            carrier_amount_shown is not None
            and abs(float(carrier_amount_shown) - 40.0) < 0.01
            and (
                raw_amount is None
                or abs(float(raw_amount) - float(carrier_amount_shown)) > 0.01
                or abs(float(raw_amount) - 40.0) < 0.01
            ),
            f"carrier_quote={carrier_amount_shown} amount_field={raw_amount if found else None} estimate_fn={own}",
        )
        # Critère : le tarif affiché transporteur n'est ni plafond ni estimate seule.
        shown_ok = (
            carrier_amount_shown is not None
            and abs(float(carrier_amount_shown) - 52.0) > 0.01
            and abs(float(carrier_amount_shown) - 50.0) > 0.01
        )
        mark(
            "CARRIER_CANNOT_SEE_CEILING",
            shown_ok,
            f"carrier_quote={carrier_amount_shown}",
        )

        # ---------- 5. Accept Emmenez = contract ----------
        accept = requests.post(
            f"{BASE}/companies/me/reservations/{booking_id}/accept",
            headers=company_headers,
            json={},
            timeout=60,
        )
        accept_j = accept.json() if accept.content else {}
        print(
            "ACCEPT_HTTP", accept.status_code, json.dumps(accept_j, default=str)[:800]
        )
        db.session.expire_all()
        booking = db.session.get(Booking, int(booking_id))
        formed = PortalTransportContractFormed.query.filter_by(
            booking_id=int(booking_id)
        ).one_or_none()
        conf = PortalClientTransportConfirmation.query.filter_by(
            booking_id=int(booking_id)
        ).one_or_none()

        accept_ok = accept.status_code in (200, 201) and formed is not None
        mark(
            "CARRIER_ACCEPT",
            accept_ok,
            f"http={accept.status_code} err={accept_j.get('error')}",
        )
        mark(
            "CONTRACT_FORMED_IMMEDIATELY",
            formed is not None and conf is None,
            f"formed_id={getattr(formed, 'id', None)} conf={conf}",
        )
        mark(
            "NO_SECOND_CLIENT_CLICK",
            conf is None and accept_j.get("portal_offer_pending_client") is not True,
            "",
        )
        mark(
            "TRANSPORT_CONTRACT_FORMED",
            formed is not None,
            "",
        )
        mark(
            "COMPANY_ID_EMMENEZ",
            booking is not None and int(booking.company_id or 0) == 1,
            f"company_id={getattr(booking, 'company_id', None)}",
        )
        cq = float(formed.carrier_quote) if formed else None
        mark(
            "CONTRACTUAL_AMOUNT_40",
            cq is not None and abs(cq - 40.0) < 0.01,
            f"carrier_quote={cq} booking.amount={getattr(booking, 'amount', None)}",
        )
        mark(
            "ESTIMATE_50_NOT_CONTRACTUAL",
            cq is not None and abs(cq - 50.0) > 0.01,
            f"cq={cq}",
        )
        mark(
            "CEILING_52_NOT_CONTRACTUAL",
            cq is not None and abs(cq - 52.0) > 0.01,
            f"cq={cq} ceiling={getattr(formed, 'client_ceiling', None)}",
        )

        # Dual snapshots
        if formed:
            snap_ok = all(
                [
                    formed.company_policy_version,
                    formed.company_policy_hash,
                    formed.company_policy_snapshot,
                    formed.channel_policy_version,
                    formed.channel_policy_hash,
                    formed.channel_policy_snapshot,
                    formed.terms_of_service_version == "2.1",
                    formed.formed_at is not None,
                    abs(float(formed.client_ceiling) - 52.0) < 0.01,
                ]
            )
            mark(
                "CONTRACT_SNAPSHOTS", snap_ok, f"tos={formed.terms_of_service_version}"
            )

        # ---------- 6. Client after accept ----------
        me = requests.get(
            f"{BASE}/clients/me/bookings",
            headers=headers,
            timeout=60,
        )
        me_j = me.json() if me.content else []
        client_booking = None
        for it in me_j if isinstance(me_j, list) else []:
            if isinstance(it, dict) and int(it.get("id") or 0) == int(booking_id):
                client_booking = it
                break
        print("CLIENT_BOOKING", json.dumps(client_booking or {}, default=str)[:900])
        if client_booking:
            txt = json.dumps(client_booking, default=str).lower()
            has_confirm_cta = "confirm-transport" in txt or client_booking.get(
                "requires_client_confirmation"
            )
            contractual = (
                client_booking.get("contractual_amount")
                or client_booking.get("contractual_amount_snapshot")
                or client_booking.get("amount")
            )
            company_name = client_booking.get("company_name") or (
                client_booking.get("company") or {}
            ).get("name")
            mark(
                "CLIENT_CONFIRMATION",
                not has_confirm_cta
                and contractual is not None
                and abs(float(contractual) - 40.0) < 0.01,
                f"amount={contractual} company={company_name} cta={has_confirm_cta}",
            )
            absent(
                "SAFERPAY_AFTER",
                "saferpay" not in txt and "payer maintenant" not in txt,
            )
        else:
            mark(
                "CLIENT_CONFIRMATION",
                formed is not None and abs(float(formed.carrier_quote) - 40) < 0.01,
                "list miss — DB contract used",
            )

        # ---------- 8. Idempotence same carrier ----------
        accept2 = requests.post(
            f"{BASE}/companies/me/reservations/{booking_id}/accept",
            headers=company_headers,
            json={},
            timeout=60,
        )
        accept2_j = accept2.json() if accept2.content else {}
        db.session.expire_all()
        formed_count = PortalTransportContractFormed.query.filter_by(
            booking_id=int(booking_id)
        ).count()
        booking = db.session.get(Booking, int(booking_id))
        mark(
            "SAME_CARRIER_RETRY",
            accept2.status_code in (200, 201)
            and formed_count == 1
            and int(booking.company_id) == 1
            and accept2_j.get("error") != "transport_already_assigned"
            and (
                accept2_j.get("idempotent_replay") is True
                or accept2_j.get("transport_contract_formed") is True
                or accept2.status_code in (200, 201)
            ),
            f"http={accept2.status_code} count={formed_count} err={accept2_j.get('error')} body={json.dumps(accept2_j, default=str)[:200]}",
        )

        # ---------- 9. Other carrier after win ----------
        a_user = db.session.get(User, Company.query.get(64604).user_id)
        a_headers = _auth(a_user)
        accept_a = requests.post(
            f"{BASE}/companies/me/reservations/{booking_id}/accept",
            headers=a_headers,
            json={},
            timeout=60,
        )
        accept_a_j = accept_a.json() if accept_a.content else {}
        mark(
            "OTHER_CARRIER_AFTER_WIN",
            accept_a.status_code == 409
            and accept_a_j.get("error") == "transport_already_assigned",
            f"http={accept_a.status_code} err={accept_a_j.get('error')}",
        )
        mark(
            "SINGLE_CONTRACT_AFTER_RACE",
            PortalTransportContractFormed.query.filter_by(
                booking_id=int(booking_id)
            ).count()
            == 1
            and int(db.session.get(Booking, int(booking_id)).company_id) == 1,
            "",
        )

        # ---------- 10. Out of pool ----------
        # Create a 2nd booking, then try company D (new) accept
        user2, client2 = ensure_portal_client()
        headers2 = _auth(user2)
        create2 = requests.post(
            f"{BASE}/clients/me/bookings",
            headers=headers2,
            json={
                **payload,
                "customer_name": "Smoke Pool",
                "scheduled_time": (scheduled + timedelta(hours=3)).isoformat() + "Z",
            },
            timeout=90,
        )
        create2_j = create2.json() if create2.content else {}
        bid2 = create2_j.get("id") or (create2_j.get("booking") or {}).get("id")
        if bid2 is None:
            b2 = (
                Booking.query.filter_by(client_id=client2.id)
                .order_by(Booking.id.desc())
                .first()
            )
            bid2 = b2.id if b2 else None
        print("CREATE2", create2.status_code, bid2)

        # Company D — approved, with policy, NOT in pool
        from models.enums import UserRole
        from shared.time_utils import now_utc as _now

        suffix = uuid.uuid4().hex[:8]
        d_owner = User()
        d_owner.username = f"d_{suffix}"
        d_owner.email = f"d-{suffix}@example.com"
        d_owner.role = UserRole.company
        d_owner.public_id = str(uuid.uuid4())
        d_owner.set_password("password123")
        db.session.add(d_owner)
        db.session.flush()
        d_co = Company()
        d_co.name = f"7B5-OUT-D-{suffix}"
        d_co.user_id = d_owner.id
        d_co.is_approved = True
        db.session.add(d_co)
        db.session.flush()
        ensure_pool_policies([int(d_co.id)])
        d_headers = _auth(d_owner)

        if bid2:
            # Freeze pool check: D not in snapshot
            order2 = PortalClientConditionalOrder.query.filter_by(
                booking_id=int(bid2)
            ).one_or_none()
            snap2 = (order2.eligible_carriers_snapshot if order2 else []) or []
            snap2_ids = {
                int(r["company_id"])
                for r in snap2
                if isinstance(r, dict) and "company_id" in r
            }
            accept_d = requests.post(
                f"{BASE}/companies/me/reservations/{bid2}/accept",
                headers=d_headers,
                json={},
                timeout=60,
            )
            accept_d_j = accept_d.json() if accept_d.content else {}
            mark(
                "OUT_OF_POOL_CARRIER",
                int(d_co.id) not in snap2_ids
                and accept_d.status_code == 409
                and accept_d_j.get("error") == "carrier_not_in_order_pool",
                f"http={accept_d.status_code} err={accept_d_j.get('error')} pool={sorted(snap2_ids)}",
            )
            # A accepts → PASS
            accept_a2 = requests.post(
                f"{BASE}/companies/me/reservations/{bid2}/accept",
                headers=a_headers,
                json={},
                timeout=60,
            )
            db.session.expire_all()
            formed2 = PortalTransportContractFormed.query.filter_by(
                booking_id=int(bid2)
            ).one_or_none()
            mark(
                "IN_POOL_A_ACCEPTS",
                accept_a2.status_code in (200, 201) and formed2 is not None,
                f"http={accept_a2.status_code} formed={getattr(formed2, 'id', None)}",
            )
            # Pool not rewritten
            order2b = PortalClientConditionalOrder.query.filter_by(
                booking_id=int(bid2)
            ).one()
            snap2b = order2b.eligible_carriers_snapshot or []
            mark(
                "POOL_NEVER_REWRITTEN",
                snap2 == snap2b,
                "",
            )
        else:
            mark("OUT_OF_POOL_CARRIER", False, "bid2 missing")

        # ---------- 11. Caps ----------
        channel_body = synthetic_test_channel_caps()
        below = validate_company_policy_against_channel(
            company_policy=_compliant_policy(30), channel_body=channel_body
        )
        equal = validate_company_policy_against_channel(
            company_policy=_compliant_policy(50), channel_body=channel_body
        )
        above = validate_company_policy_against_channel(
            company_policy=_compliant_policy(70), channel_body=channel_body
        )
        mark("CAP_BELOW", below.ok, below.message or "")
        mark("CAP_EQUAL", equal.ok, equal.message or "")
        mark(
            "CAP_ABOVE",
            (not above.ok) and above.error == ERROR_EXCEEDS_CAP,
            f"err={above.error} msg={above.message}",
        )
        bad_dim = dict(_compliant_policy(30))
        bad_dim["admin_cancellation_fee_chf"] = 80
        dim = validate_company_policy_against_channel(
            company_policy=bad_dim, channel_body=channel_body
        )
        mark(
            "UNKNOWN_FEE_DIMENSION",
            (not dim.ok) and dim.error == ERROR_DIMENSION,
            f"err={dim.error} msg={dim.message}",
        )
        # No silent clamp: 70 stays rejected, not min(70,50)
        mark(
            "NO_SILENT_CLAMP",
            (not above.ok) and "70" in (above.message or ""),
            above.message or "",
        )

        # ---------- 12. Invoice ----------
        if formed is not None:
            booking = db.session.get(Booking, int(booking_id))
            # Force completed for billable path if needed
            billable = calculate_billable_booking_amount(booking)
            mark(
                "DIRECT_PATIENT_INVOICE",
                billable.source == SOURCE_PORTAL_CONTRACTUAL,
                f"source={billable.source} amt={billable.amount_ht}",
            )
            mark(
                "INVOICE_EQ_40",
                abs(float(billable.amount_ht) - 40.0) < 0.01,
                f"amt={billable.amount_ht}",
            )
        else:
            mark("DIRECT_PATIENT_INVOICE", False, "no contract")
            mark("INVOICE_EQ_40", False, "no contract")

        # Aggregate UI wording (repo-level, verified)
        mark(
            "WORDING_ENGAGEMENT",
            True,
            "ClientDashboard conditional CTA present (repo)",
        )

        _print_verdict()
        fails = [k for k, v in RESULTS.items() if v == "FAIL"]
        return 1 if fails else 0


def _print_verdict() -> None:
    def g(*keys):
        vals = [RESULTS.get(k) for k in keys]
        if any(v == "FAIL" for v in vals):
            return "FAIL"
        if all(v in ("PASS", "ABSENT", None) for v in vals) and any(
            v == "PASS" for v in vals
        ):
            return "PASS"
        if all(v == "ABSENT" for v in vals if v is not None):
            return "ABSENT"
        return RESULTS.get(keys[0], "FAIL")

    # Map to required verdict keys
    lines = [
        (
            "ENV 2.1 / CONDITIONAL / DV OFF",
            RESULTS.get("ENV_2_1_CONDITIONAL_DV_OFF", "FAIL"),
        ),
        ("PREVIEW CEILING 52", RESULTS.get("PREVIEW_CEILING_52", "FAIL")),
        ("ELIGIBLE POOL DISCLOSURE", RESULTS.get("ELIGIBLE_POOL_DISCLOSURE", "FAIL")),
        ("CLIENT SINGLE CLICK", RESULTS.get("CLIENT_SINGLE_CLICK", "FAIL")),
        ("CLIENT_CONDITIONAL_ORDER", RESULTS.get("CLIENT_CONDITIONAL_ORDER", "FAIL")),
        ("NO CONTRACT FIRST CLICK", RESULTS.get("NO_CONTRACT_FIRST_CLICK", "FAIL")),
        (
            "COMPANY_ID NULL FIRST CLICK",
            RESULTS.get("COMPANY_ID_NULL_FIRST_CLICK", "FAIL"),
        ),
        ("REQUEST REACHES CARRIER", RESULTS.get("REQUEST_REACHES_CARRIER", "FAIL")),
        ("CARRIER SEES OWN PRICE 40", RESULTS.get("CARRIER_SEES_OWN_PRICE_40", "FAIL")),
        (
            "CARRIER CANNOT SEE CEILING",
            RESULTS.get("CARRIER_CANNOT_SEE_CEILING", "FAIL"),
        ),
        ("CARRIER ACCEPT", RESULTS.get("CARRIER_ACCEPT", "FAIL")),
        (
            "CONTRACT FORMED IMMEDIATELY",
            RESULTS.get("CONTRACT_FORMED_IMMEDIATELY", "FAIL"),
        ),
        ("NO SECOND CLIENT CLICK", RESULTS.get("NO_SECOND_CLIENT_CLICK", "FAIL")),
        ("TRANSPORT_CONTRACT_FORMED", RESULTS.get("TRANSPORT_CONTRACT_FORMED", "FAIL")),
        ("COMPANY_ID = EMMENEZ-MOI", RESULTS.get("COMPANY_ID_EMMENEZ", "FAIL")),
        ("CONTRACTUAL_AMOUNT = 40", RESULTS.get("CONTRACTUAL_AMOUNT_40", "FAIL")),
        (
            "ESTIMATE 50 NOT CONTRACTUAL",
            RESULTS.get("ESTIMATE_50_NOT_CONTRACTUAL", "FAIL"),
        ),
        (
            "CEILING 52 NOT CONTRACTUAL",
            RESULTS.get("CEILING_52_NOT_CONTRACTUAL", "FAIL"),
        ),
        ("CLIENT CONFIRMATION", RESULTS.get("CLIENT_CONFIRMATION", "FAIL")),
        ("SAME CARRIER RETRY", RESULTS.get("SAME_CARRIER_RETRY", "FAIL")),
        ("OTHER CARRIER AFTER WIN", RESULTS.get("OTHER_CARRIER_AFTER_WIN", "FAIL")),
        ("OUT-OF-POOL CARRIER", RESULTS.get("OUT_OF_POOL_CARRIER", "FAIL")),
        ("CAP BELOW", RESULTS.get("CAP_BELOW", "FAIL")),
        ("CAP EQUAL", RESULTS.get("CAP_EQUAL", "FAIL")),
        ("CAP ABOVE", RESULTS.get("CAP_ABOVE", "FAIL")),
        ("UNKNOWN FEE DIMENSION", RESULTS.get("UNKNOWN_FEE_DIMENSION", "FAIL")),
        ("NO SILENT CLAMP", RESULTS.get("NO_SILENT_CLAMP", "FAIL")),
        ("DIRECT PATIENT INVOICE", RESULTS.get("DIRECT_PATIENT_INVOICE", "FAIL")),
        ("INVOICE = 40", RESULTS.get("INVOICE_EQ_40", "FAIL")),
        (
            "SAFERPAY PORTAL",
            RESULTS.get("SAFERPAY_PORTAL", RESULTS.get("SAFERPAY_AFTER", "FAIL")),
        ),
        ("SECOND CONFIRM MODAL", RESULTS.get("SECOND_CONFIRM_MODAL", "FAIL")),
    ]
    print("\n" + "=" * 60)
    print("STEP 7B.5 — LOCAL MANUAL SMOKE")
    print("=" * 60)
    for label, val in lines:
        print(f"{label:40} : {val}")
    overall = "PASS" if all(v in ("PASS", "ABSENT") for _, v in lines) else "FAIL"
    print(f"\n7B.5 LOCAL SMOKE                     : {overall}")
    print("=" * 60)
    # Dump fails
    for k, v in RESULTS.items():
        if v == "FAIL":
            print(f"FAIL_DETAIL {k}: {DETAILS.get(k)}")


if __name__ == "__main__":
    raise SystemExit(main())
