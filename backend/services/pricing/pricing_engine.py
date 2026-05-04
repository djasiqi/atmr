from __future__ import annotations

# ruff: noqa: ARG001, SIM102, SIM114
import json
import re
from collections import OrderedDict
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from models import PricingProfileVersion
from services.pricing.zone_set_resolver import estimate_zones_traversed, resolve_zone_id

_LAST_MINUTE_LEAD_TIME_MAX_MINUTES = 120


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value or 0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _normalize_breakdown(
    model: str, base_amount: Decimal, base_rule: str
) -> OrderedDict[str, Any]:
    return OrderedDict(
        [
            ("model", model),
            (
                "base",
                OrderedDict([("amount", f"{base_amount:.2f}"), ("rule", base_rule)]),
            ),
            ("extras", []),
            ("minimum", OrderedDict([("applied", False), ("amount", "0.00")])),
            ("total", f"{base_amount:.2f}"),
        ]
    )


def _apply_minimum(
    total: Decimal, minimum: Decimal, breakdown: OrderedDict[str, Any]
) -> Decimal:
    if minimum > 0 and total < minimum:
        breakdown["minimum"] = OrderedDict(
            [("applied", True), ("amount", f"{minimum:.2f}")]
        )
        return minimum
    return total


def _is_after_time(pickup_local_time: str | None, threshold: str | None) -> bool:
    if not pickup_local_time or not threshold:
        return False
    try:
        lhs = datetime.strptime(pickup_local_time, "%H:%M").time()
        rhs = datetime.strptime(threshold, "%H:%M").time()
    except ValueError:
        return False
    return lhs >= rhs


def compute_price(
    booking: dict[str, Any],
    pricing_profile_version: PricingProfileVersion,
    context: dict[str, Any],
) -> tuple[Decimal, dict[str, Any]]:
    rules = pricing_profile_version.rules_json or {}
    model = rules.get("model", pricing_profile_version.pricing_profile.model_type.value)

    if model == "flat":
        amount, breakdown = _compute_flat(rules, context)
    elif model == "zone_count":
        amount, breakdown = _compute_zone_count_v1(rules, context)
    elif model == "hybrid_stack":
        amount, breakdown = _compute_hybrid_stack_v1(rules, context)
    elif model == "zone":
        if _has_zone_matrix_rules(rules):
            amount, breakdown = _compute_zone_matrix(rules, context)
        else:
            amount, breakdown = _compute_zone(rules, context)
    elif model == "zone_matrix":
        amount, breakdown = _compute_zone_matrix(rules, context)
    elif model == "zone_matrix_v1":
        amount, breakdown = _compute_zone_matrix(rules, context)
    elif model == "zone_v1":
        amount, breakdown = _compute_zone(rules, context)
    elif model == "distance":
        amount, breakdown = _compute_distance(rules, context)
    else:
        raise ValueError(f"Unsupported pricing model: {model}")

    # Deterministic JSON for audit persistence.
    stable_json = json.loads(json.dumps(breakdown, sort_keys=True))
    return amount, stable_json


def _compute_flat(
    rules: dict[str, Any], context: dict[str, Any]
) -> tuple[Decimal, OrderedDict[str, Any]]:
    components = (
        rules.get("components") if isinstance(rules.get("components"), dict) else {}
    )
    base_comp = components.get("base") if isinstance(components, dict) else {}
    if rules.get("base_fee") is not None:
        base = _to_decimal(rules.get("base_fee", 0))
        rule_name = "base_fee"
    elif isinstance(base_comp, dict) and base_comp.get("enabled"):
        base = _to_decimal(base_comp.get("amount", 0))
        rule_name = "base_component"
    else:
        base = _to_decimal(0)
        rule_name = "base_fee"
    if context.get("is_weekend"):
        for item in rules.get("time_rules", []):
            if item.get("when") == "weekend" and item.get("base_fee") is not None:
                base = _to_decimal(item["base_fee"])
                rule_name = "weekend_base_fee"
                break

    breakdown = _normalize_breakdown("flat", base, rule_name)
    total = base
    for extra in rules.get("surcharges", []):
        etype = extra.get("type")
        if etype == "after_time" and _is_after_time(
            context.get("pickup_local_time"), extra.get("after")
        ):
            amount = _to_decimal(extra.get("amount", 0))
        elif etype == "last_minute" and context.get(
            "minutes_until_pickup", 999999
        ) < int(extra.get("under_minutes", 0)):
            amount = _to_decimal(extra.get("amount", 0))
        else:
            continue
        breakdown["extras"].append(
            OrderedDict(
                [
                    ("type", etype),
                    ("qty", 1),
                    ("unit", f"{amount:.2f}"),
                    ("amount", f"{amount:.2f}"),
                ]
            )
        )
        total += amount

    minimum = _to_decimal(
        (rules.get("caps") or {}).get("minimum", 0)
        if isinstance(rules.get("caps"), dict)
        else rules.get("minimum", 0)
    )
    total = _apply_minimum(total, minimum, breakdown)
    breakdown["total"] = f"{total:.2f}"
    return total, breakdown


def _compute_zone(
    rules: dict[str, Any], context: dict[str, Any]
) -> tuple[Decimal, OrderedDict[str, Any]]:
    pricing = rules.get("pricing", {})
    day_key = "weekend" if context.get("is_weekend") else "weekday"
    trip_key = "round_trip" if context.get("is_round_trip") else "one_way"
    base = _to_decimal(pricing.get(day_key, {}).get(trip_key, 0))
    breakdown = _normalize_breakdown("zone", base, f"{day_key}_{trip_key}")
    total = base

    zones_count = int(context.get("zones_count", 1) or 1)
    for extra in rules.get("extras", []):
        total = _append_zone_extra(
            extra=extra,
            context=context,
            zones_count=zones_count,
            breakdown=breakdown,
            total=total,
        )

    minimum = _to_decimal(rules.get("minimum", 0))
    total = _apply_minimum(total, minimum, breakdown)
    breakdown["total"] = f"{total:.2f}"
    return total, breakdown


def _append_zone_extra(
    *,
    extra: dict[str, Any],
    context: dict[str, Any],
    zones_count: int,
    breakdown: OrderedDict[str, Any],
    total: Decimal,
) -> Decimal:
    etype = extra.get("type")
    if etype == "after_time_per_zone" and _is_after_time(
        context.get("pickup_local_time"), extra.get("after")
    ):
        unit = _to_decimal(extra.get("amount", 0))
        amount = unit * max(zones_count, 1)
        breakdown["extras"].append(
            OrderedDict(
                [
                    ("type", etype),
                    ("qty", max(zones_count, 1)),
                    ("unit", f"{unit:.2f}"),
                    ("amount", f"{amount:.2f}"),
                ]
            )
        )
        return total + amount
    if etype == "last_minute" and context.get("minutes_until_pickup", 999999) < int(
        extra.get("under_minutes", 0)
    ):
        amount = _to_decimal(extra.get("amount", 0))
        breakdown["extras"].append(
            OrderedDict(
                [
                    ("type", etype),
                    ("qty", 1),
                    ("unit", f"{amount:.2f}"),
                    ("amount", f"{amount:.2f}"),
                ]
            )
        )
        return total + amount
    return total


def _has_zone_matrix_rules(rules: dict[str, Any]) -> bool:
    return bool(rules.get("zones")) and isinstance(rules.get("matrix"), dict)


def _zone_token_from_context(prefix: str, context: dict[str, Any]) -> str | None:
    token = context.get(f"{prefix}_admin_token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    geo_unit_id = context.get(f"{prefix}_geo_unit_id")
    if isinstance(geo_unit_id, int) and geo_unit_id > 0:
        return f"commune:{geo_unit_id}"
    if isinstance(geo_unit_id, str) and geo_unit_id.isdigit():
        return f"commune:{geo_unit_id}"
    return None


def _zone_from_definitions(
    *,
    zones: list[dict[str, Any]],
    token: str | None,
    token_to_zone_id: dict[str, str],
) -> dict[str, Any] | None:
    if not token:
        return None
    zone_id = token_to_zone_id.get(token)
    if not zone_id:
        return None
    for zone in zones:
        if str(zone.get("id") or "") == zone_id:
            return zone
    return None


def _normalize_zone_matrix_matrix(raw: Any) -> dict[str, dict[str, Decimal]]:
    normalized: dict[str, dict[str, Decimal]] = {}
    if not isinstance(raw, dict):
        return normalized
    for src, destinations in raw.items():
        src_key = str(src)
        if not isinstance(destinations, dict):
            continue
        normalized[src_key] = {}
        for dst, value in destinations.items():
            normalized[src_key][str(dst)] = _to_decimal(value)
    return normalized


def _compute_zone_matrix(
    rules: dict[str, Any], context: dict[str, Any]
) -> tuple[Decimal, OrderedDict[str, Any]]:
    zones_raw = rules.get("zones")
    zones = zones_raw if isinstance(zones_raw, list) else []
    token_to_zone_id: dict[str, str] = {}
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        zone_id = str(zone.get("id") or "").strip()
        if not zone_id:
            continue
        tokens = zone.get("tokens")
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            token_value = str(token or "").strip()
            if token_value:
                token_to_zone_id[token_value] = zone_id

    matrix = _normalize_zone_matrix_matrix(rules.get("matrix"))
    matrix_symmetry = bool(rules.get("matrix_symmetry", False))
    default_same_zone_price = rules.get("default_same_zone_price")
    default_same_zone = (
        _to_decimal(default_same_zone_price)
        if default_same_zone_price is not None
        else None
    )

    pickup_token = _zone_token_from_context("pickup", context)
    dropoff_token = _zone_token_from_context("dropoff", context)
    from_zone = _zone_from_definitions(
        zones=zones, token=pickup_token, token_to_zone_id=token_to_zone_id
    )
    to_zone = _zone_from_definitions(
        zones=zones, token=dropoff_token, token_to_zone_id=token_to_zone_id
    )

    warnings: list[str] = []
    if not from_zone or not to_zone:
        warnings.append("unassigned_commune_fallback")
        fallback_amount, fallback_breakdown = _compute_zone(rules, context)
        fallback_breakdown["model"] = "zone_matrix"
        fallback_breakdown["warnings"] = warnings
        fallback_breakdown["from_zone"] = None
        fallback_breakdown["to_zone"] = None
        return fallback_amount, fallback_breakdown

    from_id = str(from_zone.get("id"))
    to_id = str(to_zone.get("id"))
    base = matrix.get(from_id, {}).get(to_id)
    if base is None and matrix_symmetry:
        base = matrix.get(to_id, {}).get(from_id)
    if base is None and default_same_zone is not None and from_id == to_id:
        base = default_same_zone
    if base is None:
        warnings.append("missing_matrix_key_fallback")
        fallback_amount, fallback_breakdown = _compute_zone(rules, context)
        fallback_breakdown["model"] = "zone_matrix"
        fallback_breakdown["warnings"] = warnings
        fallback_breakdown["from_zone"] = OrderedDict(
            [
                ("id", from_id),
                ("code", from_zone.get("code")),
                ("label", from_zone.get("label")),
            ]
        )
        fallback_breakdown["to_zone"] = OrderedDict(
            [
                ("id", to_id),
                ("code", to_zone.get("code")),
                ("label", to_zone.get("label")),
            ]
        )
        return fallback_amount, fallback_breakdown

    breakdown = _normalize_breakdown("zone_matrix", base, f"{from_id}->{to_id}")
    breakdown["from_zone"] = OrderedDict(
        [
            ("id", from_id),
            ("code", from_zone.get("code")),
            ("label", from_zone.get("label")),
        ]
    )
    breakdown["to_zone"] = OrderedDict(
        [("id", to_id), ("code", to_zone.get("code")), ("label", to_zone.get("label"))]
    )
    breakdown["warnings"] = warnings
    total = base

    zones_count = 1 if from_id == to_id else 2
    for extra in rules.get("extras", []):
        total = _append_zone_extra(
            extra=extra,
            context=context,
            zones_count=int(context.get("zones_count") or zones_count),
            breakdown=breakdown,
            total=total,
        )

    minimum = _to_decimal(rules.get("minimum", 0))
    total = _apply_minimum(total, minimum, breakdown)
    breakdown["total"] = f"{total:.2f}"
    return total, breakdown


def _validate_zone_token(token: str) -> bool:
    return bool(re.match(r"^(commune|district|canton):[A-Za-z0-9_-]+$", token))


def validate_zone_matrix_rules(rules: dict[str, Any]) -> dict[str, Any]:
    """Valide et normalise le format zone_matrix V1."""
    normalized = dict(rules or {})
    zones_raw = normalized.get("zones")
    zones = zones_raw if isinstance(zones_raw, list) else []
    zone_ids: set[str] = set()
    used_tokens: set[str] = set()

    norm_zones: list[dict[str, Any]] = []
    for zone in zones:
        if not isinstance(zone, dict):
            raise ValueError("Zone invalide: objet attendu.")
        zone_id = str(zone.get("id") or "").strip()
        if not zone_id:
            raise ValueError("Zone invalide: id requis.")
        if zone_id in zone_ids:
            raise ValueError("Zone invalide: id dupliqué.")
        if not re.match(r"^[A-Za-z0-9_-]{1,40}$", zone_id):
            raise ValueError("Zone invalide: id non conforme.")
        zone_ids.add(zone_id)

        tokens_raw = zone.get("tokens")
        tokens = tokens_raw if isinstance(tokens_raw, list) else []
        norm_tokens: list[str] = []
        for token in tokens:
            token_value = str(token or "").strip()
            if not token_value:
                continue
            if not _validate_zone_token(token_value):
                raise ValueError("Zone invalide: token non canonique.")
            if token_value in used_tokens:
                raise ValueError("Zone invalide: token présent dans plusieurs zones.")
            used_tokens.add(token_value)
            norm_tokens.append(token_value)

        norm_zones.append(
            {
                "id": zone_id,
                "code": str(zone.get("code") or "").strip() or None,
                "label": str(zone.get("label") or zone_id).strip(),
                "tokens": norm_tokens,
            }
        )

    matrix_symmetry = bool(normalized.get("matrix_symmetry", False))
    default_same_zone_price = normalized.get("default_same_zone_price")
    if default_same_zone_price is not None and _to_decimal(
        default_same_zone_price
    ) < Decimal("0"):
        raise ValueError("default_same_zone_price doit être >= 0.")

    raw_matrix = normalized.get("matrix")
    matrix = raw_matrix if isinstance(raw_matrix, dict) else {}
    norm_matrix: dict[str, dict[str, float]] = {}
    for src, destinations in matrix.items():
        src_id = str(src)
        if src_id not in zone_ids:
            raise ValueError("Matrice invalide: zone source inconnue.")
        if not isinstance(destinations, dict):
            raise ValueError("Matrice invalide: map de destinations attendue.")
        norm_matrix[src_id] = {}
        for dst, value in destinations.items():
            dst_id = str(dst)
            if dst_id not in zone_ids:
                raise ValueError("Matrice invalide: zone destination inconnue.")
            amount = _to_decimal(value)
            if amount < Decimal("0"):
                raise ValueError("Matrice invalide: montant négatif interdit.")
            norm_matrix[src_id][dst_id] = float(amount)

    if matrix_symmetry:
        ids = sorted(zone_ids)
        for src_id in ids:
            norm_matrix.setdefault(src_id, {})
            for dst_id in ids:
                if src_id == dst_id:
                    continue
                lhs = norm_matrix.get(src_id, {}).get(dst_id)
                rhs = norm_matrix.get(dst_id, {}).get(src_id)
                if lhs is None and rhs is not None:
                    norm_matrix[src_id][dst_id] = rhs
                elif rhs is None and lhs is not None:
                    norm_matrix.setdefault(dst_id, {})[src_id] = lhs

    if default_same_zone_price is not None:
        diag = float(_to_decimal(default_same_zone_price))
        for zone_id in zone_ids:
            norm_matrix.setdefault(zone_id, {})
            norm_matrix[zone_id].setdefault(zone_id, diag)

    normalized["model"] = "zone_matrix"
    normalized["zones"] = norm_zones
    normalized["matrix"] = norm_matrix
    normalized["matrix_symmetry"] = matrix_symmetry
    normalized["default_same_zone_price"] = (
        float(_to_decimal(default_same_zone_price))
        if default_same_zone_price is not None
        else None
    )
    extras = normalized.get("extras")
    normalized["extras"] = extras if isinstance(extras, list) else []
    return normalized


def build_zone_matrix_summary(rules: dict[str, Any]) -> dict[str, Any]:
    zones = rules.get("zones")
    matrix = rules.get("matrix")
    if not isinstance(zones, list) or not isinstance(matrix, dict):
        return {"zones_count": 0, "transitions_count": 0}
    zone_count = len([z for z in zones if isinstance(z, dict) and z.get("id")])
    transitions = 0
    for _, v in matrix.items():
        if isinstance(v, dict):
            transitions += len(v)
    return {"zones_count": zone_count, "transitions_count": transitions}


def _compute_distance(
    rules: dict[str, Any], context: dict[str, Any]
) -> tuple[Decimal, OrderedDict[str, Any]]:
    if isinstance(rules.get("components"), dict):
        return _compute_distance_v1(rules, context)
    base_fee = _to_decimal(rules.get("base_fee", 0))
    per_km = _to_decimal(rules.get("per_km", 0))
    distance_km = Decimal(str(context.get("distance_km", 0) or 0))
    base = (base_fee + (per_km * distance_km)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    breakdown = _normalize_breakdown("distance", base, "base_plus_km")
    total = base

    if context.get("is_weekend"):
        for item in rules.get("time_rules", []):
            if item.get("when") == "weekend" and item.get("multiplier"):
                multiplier = Decimal(str(item.get("multiplier")))
                weekend_delta = (base * (multiplier - Decimal("1"))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                if weekend_delta > 0:
                    breakdown["extras"].append(
                        OrderedDict(
                            [
                                ("type", "weekend_multiplier"),
                                ("qty", 1),
                                ("unit", f"{weekend_delta:.2f}"),
                                ("amount", f"{weekend_delta:.2f}"),
                            ]
                        )
                    )
                    total += weekend_delta
                break

    for extra in rules.get("extras", []):
        if extra.get("type") == "after_time" and _is_after_time(
            context.get("pickup_local_time"), extra.get("after")
        ):
            amount = _to_decimal(extra.get("amount", 0))
            breakdown["extras"].append(
                OrderedDict(
                    [
                        ("type", "after_time"),
                        ("qty", 1),
                        ("unit", f"{amount:.2f}"),
                        ("amount", f"{amount:.2f}"),
                    ]
                )
            )
            total += amount

    minimum = _to_decimal(rules.get("minimum", 0))
    total = _apply_minimum(total, minimum, breakdown)
    breakdown["total"] = f"{total:.2f}"
    return total, breakdown


def _compute_zones_count_from_rules(
    rules: dict[str, Any], context: dict[str, Any]
) -> int:
    zone_set_id = str(rules.get("zone_set_id") or "").strip()
    pickup_token = str(context.get("pickup_admin_token") or "").strip()
    dropoff_token = str(context.get("dropoff_admin_token") or "").strip()
    context_count = int(context.get("zones_count", 1) or 1)
    traversed_estimate = estimate_zones_traversed(
        zone_set_key=zone_set_id,
        pickup_token=pickup_token,
        dropoff_token=dropoff_token,
        pickup_lat=context.get("pickup_lat"),
        pickup_lng=context.get("pickup_lng"),
        dropoff_lat=context.get("dropoff_lat"),
        dropoff_lng=context.get("dropoff_lng"),
        route_points=context.get("route_points"),
        route_geometry=context.get("route_geometry"),
    )
    if traversed_estimate:
        return max(traversed_estimate, context_count)
    if zone_set_id and pickup_token and dropoff_token:
        pickup_zone_id = resolve_zone_id(pickup_token, zone_set_id)
        dropoff_zone_id = resolve_zone_id(dropoff_token, zone_set_id)
        if pickup_zone_id and dropoff_zone_id:
            resolved = 1 if pickup_zone_id == dropoff_zone_id else 2
            return max(resolved, context_count)
    fallback = context_count
    return max(1, fallback)


def _compute_extras_v1(
    rules: dict[str, Any], context: dict[str, Any], subtotal: Decimal
) -> tuple[Decimal, list[OrderedDict[str, Any]]]:
    extras_rules = rules.get("extras") if isinstance(rules.get("extras"), dict) else {}
    entries: list[OrderedDict[str, Any]] = []
    total = subtotal

    weekend = extras_rules.get("weekend") if isinstance(extras_rules, dict) else None
    if (
        isinstance(weekend, dict)
        and weekend.get("enabled")
        and weekend.get("type") == "multiplier"
    ):
        multiplier = Decimal(str(weekend.get("value") or 1))
        if context.get("is_weekend") and multiplier > Decimal("1"):
            delta = (subtotal * (multiplier - Decimal("1"))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if delta > 0:
                total += delta
                entries.append(
                    OrderedDict(
                        [
                            ("type", "weekend_multiplier"),
                            ("qty", 1),
                            ("unit", f"{delta:.2f}"),
                            ("amount", f"{delta:.2f}"),
                        ]
                    )
                )

    after_20h = (
        extras_rules.get("after_20h") if isinstance(extras_rules, dict) else None
    )
    if (
        isinstance(after_20h, dict)
        and after_20h.get("enabled")
        and after_20h.get("type") == "fixed"
    ):
        if _is_after_time(context.get("pickup_local_time"), "20:00"):
            amount = _to_decimal(after_20h.get("value", 0))
            if amount > 0:
                total += amount
                entries.append(
                    OrderedDict(
                        [
                            ("type", "after_20h"),
                            ("qty", 1),
                            ("unit", f"{amount:.2f}"),
                            ("amount", f"{amount:.2f}"),
                        ]
                    )
                )

    last_minute = (
        extras_rules.get("last_minute") if isinstance(extras_rules, dict) else None
    )
    if (
        isinstance(last_minute, dict)
        and last_minute.get("enabled")
        and last_minute.get("type") == "fixed"
    ):
        if (
            int(context.get("minutes_until_pickup", 999999) or 999999)
            < _LAST_MINUTE_LEAD_TIME_MAX_MINUTES
        ):
            amount = _to_decimal(last_minute.get("value", 0))
            if amount > 0:
                total += amount
                entries.append(
                    OrderedDict(
                        [
                            ("type", "last_minute"),
                            ("qty", 1),
                            ("unit", f"{amount:.2f}"),
                            ("amount", f"{amount:.2f}"),
                        ]
                    )
                )

    return total, entries


def _compute_zone_count_v1(
    rules: dict[str, Any], context: dict[str, Any]
) -> tuple[Decimal, OrderedDict[str, Any]]:
    components = (
        rules.get("components") if isinstance(rules.get("components"), dict) else {}
    )
    base_comp = components.get("base") if isinstance(components, dict) else {}
    zone_comp = components.get("zone_count") if isinstance(components, dict) else {}
    base_amount = (
        _to_decimal(base_comp.get("amount", 0))
        if isinstance(base_comp, dict) and base_comp.get("enabled")
        else Decimal("0.00")
    )
    zones_count = _compute_zones_count_from_rules(rules, context)
    unit = (
        _to_decimal(zone_comp.get("unit_price", 0))
        if isinstance(zone_comp, dict) and zone_comp.get("enabled")
        else Decimal("0.00")
    )
    included_zones = (
        max(1, int(zone_comp.get("included_zones") or 1))
        if isinstance(zone_comp, dict)
        else 1
    )
    chargeable_zones = max(max(zones_count, 1) - included_zones, 0)
    zone_amount = (unit * Decimal(chargeable_zones)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    subtotal = base_amount + zone_amount
    breakdown = _normalize_breakdown("zone_count", subtotal, "base_plus_zone_count")
    breakdown["zones_traversees"] = max(zones_count, 1)
    breakdown["zones_incluses"] = included_zones
    breakdown["zones_facturables"] = chargeable_zones
    breakdown["supplement_zone"] = f"{unit:.2f}"
    breakdown["supplement_total"] = f"{zone_amount:.2f}"
    if zone_amount > 0:
        breakdown["extras"].append(
            OrderedDict(
                [
                    ("type", "zone_count"),
                    ("zones_total", max(zones_count, 1)),
                    ("zones_included", included_zones),
                    ("qty", chargeable_zones),
                    ("unit", f"{unit:.2f}"),
                    ("amount", f"{zone_amount:.2f}"),
                ]
            )
        )

    total, extras = _compute_extras_v1(rules, context, subtotal)
    for extra in extras:
        breakdown["extras"].append(extra)
    minimum = _to_decimal(
        (rules.get("caps") or {}).get("minimum", 0)
        if isinstance(rules.get("caps"), dict)
        else rules.get("minimum", 0)
    )
    total = _apply_minimum(total, minimum, breakdown)
    breakdown["total"] = f"{total:.2f}"
    return total, breakdown


def _compute_distance_v1(
    rules: dict[str, Any], context: dict[str, Any]
) -> tuple[Decimal, OrderedDict[str, Any]]:
    components = (
        rules.get("components") if isinstance(rules.get("components"), dict) else {}
    )
    distance_comp = components.get("distance") if isinstance(components, dict) else {}
    # Règle métier distance: pas de prix de base, uniquement km * prix_km (+ minimum éventuel).
    base_amount = Decimal("0.00")
    per_km = (
        _to_decimal(distance_comp.get("per_km", 0))
        if isinstance(distance_comp, dict) and distance_comp.get("enabled")
        else Decimal("0.00")
    )
    included_km = Decimal(
        str(
            distance_comp.get("included_km", 0)
            if isinstance(distance_comp, dict)
            else 0
        )
    )
    distance_km = Decimal(str(context.get("distance_km", 0) or 0))
    billable_km = max(distance_km - included_km, Decimal("0"))
    distance_amount = (billable_km * per_km).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    subtotal = base_amount + distance_amount
    breakdown = _normalize_breakdown("distance", subtotal, "km_only")
    if distance_amount > 0:
        breakdown["extras"].append(
            OrderedDict(
                [
                    ("type", "distance_km"),
                    ("qty", f"{billable_km:.2f}"),
                    ("unit", f"{per_km:.2f}"),
                    ("amount", f"{distance_amount:.2f}"),
                ]
            )
        )
    total, extras = _compute_extras_v1(rules, context, subtotal)
    for extra in extras:
        breakdown["extras"].append(extra)
    minimum = _to_decimal(
        (rules.get("caps") or {}).get("minimum", 0)
        if isinstance(rules.get("caps"), dict)
        else rules.get("minimum", 0)
    )
    total = _apply_minimum(total, minimum, breakdown)
    breakdown["total"] = f"{total:.2f}"
    return total, breakdown


def _compute_hybrid_stack_v1(
    rules: dict[str, Any], context: dict[str, Any]
) -> tuple[Decimal, OrderedDict[str, Any]]:
    components = (
        rules.get("components") if isinstance(rules.get("components"), dict) else {}
    )
    base_comp = components.get("base") if isinstance(components, dict) else {}
    zone_comp = components.get("zone_count") if isinstance(components, dict) else {}
    distance_comp = components.get("distance") if isinstance(components, dict) else {}
    base_amount = (
        _to_decimal(base_comp.get("amount", 0))
        if isinstance(base_comp, dict) and base_comp.get("enabled")
        else Decimal("0.00")
    )
    zones_count = _compute_zones_count_from_rules(rules, context)
    zone_unit = (
        _to_decimal(zone_comp.get("unit_price", 0))
        if isinstance(zone_comp, dict) and zone_comp.get("enabled")
        else Decimal("0.00")
    )
    included_zones = (
        max(1, int(zone_comp.get("included_zones") or 1))
        if isinstance(zone_comp, dict)
        else 1
    )
    chargeable_zones = max(max(zones_count, 1) - included_zones, 0)
    zone_amount = (zone_unit * Decimal(chargeable_zones)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    per_km = (
        _to_decimal(distance_comp.get("per_km", 0))
        if isinstance(distance_comp, dict) and distance_comp.get("enabled")
        else Decimal("0.00")
    )
    included_km = Decimal(
        str(
            distance_comp.get("included_km", 0)
            if isinstance(distance_comp, dict)
            else 0
        )
    )
    distance_km = Decimal(str(context.get("distance_km", 0) or 0))
    billable_km = max(distance_km - included_km, Decimal("0"))
    distance_amount = (billable_km * per_km).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    subtotal = base_amount + zone_amount + distance_amount
    breakdown = _normalize_breakdown(
        "hybrid_stack", subtotal, "base_plus_zones_plus_km"
    )
    breakdown["zones_traversees"] = max(zones_count, 1)
    breakdown["zones_incluses"] = included_zones
    breakdown["zones_facturables"] = chargeable_zones
    breakdown["supplement_zone"] = f"{zone_unit:.2f}"
    breakdown["supplement_total"] = f"{zone_amount:.2f}"
    if zone_amount > 0:
        breakdown["extras"].append(
            OrderedDict(
                [
                    ("type", "zone_count"),
                    ("zones_total", max(zones_count, 1)),
                    ("zones_included", included_zones),
                    ("qty", chargeable_zones),
                    ("unit", f"{zone_unit:.2f}"),
                    ("amount", f"{zone_amount:.2f}"),
                ]
            )
        )
    if distance_amount > 0:
        breakdown["extras"].append(
            OrderedDict(
                [
                    ("type", "distance_km"),
                    ("qty", f"{billable_km:.2f}"),
                    ("unit", f"{per_km:.2f}"),
                    ("amount", f"{distance_amount:.2f}"),
                ]
            )
        )
    total, extras = _compute_extras_v1(rules, context, subtotal)
    for extra in extras:
        breakdown["extras"].append(extra)
    minimum = _to_decimal(
        (rules.get("caps") or {}).get("minimum", 0)
        if isinstance(rules.get("caps"), dict)
        else rules.get("minimum", 0)
    )
    total = _apply_minimum(total, minimum, breakdown)
    breakdown["total"] = f"{total:.2f}"
    return total, breakdown


def validate_company_pricing_rules(rules: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(rules or {})
    model = str(normalized.get("model") or "").strip().lower()
    if model == "zone_matrix":
        zone_set_id = str(normalized.get("zone_set_id") or "").strip() or None
        if not zone_set_id:
            raise ValueError("zone_set_id requis pour modèle zone_matrix.")
        validated = validate_zone_matrix_rules(normalized)
        validated["v"] = int(normalized.get("v") or 1)
        validated["currency"] = (
            str(normalized.get("currency") or "CHF").strip().upper() or "CHF"
        )
        validated["zone_set_id"] = zone_set_id
        minimum = validated.get("minimum")
        if minimum is not None and _to_decimal(minimum) < 0:
            raise ValueError("minimum doit être >= 0.")
        validated["minimum"] = float(_to_decimal(minimum or 0))
        return validated
    if model not in {"flat", "zone_count", "distance", "hybrid_stack"}:
        raise ValueError("model pricing non supporté.")
    normalized["model"] = model
    normalized["v"] = int(normalized.get("v") or 1)
    normalized["currency"] = (
        str(normalized.get("currency") or "CHF").strip().upper() or "CHF"
    )
    components = normalized.get("components")
    if not isinstance(components, dict):
        raise ValueError("components requis.")
    raw_base = components.get("base")
    base = raw_base if isinstance(raw_base, dict) else {"enabled": False, "amount": 0}
    raw_zone_count = components.get("zone_count")
    zone_count = (
        raw_zone_count
        if isinstance(raw_zone_count, dict)
        else {"enabled": False, "unit_price": 0, "max_units": 10}
    )
    raw_distance = components.get("distance")
    distance = (
        raw_distance
        if isinstance(raw_distance, dict)
        else {"enabled": False, "per_km": 0, "included_km": 0}
    )

    base_amount = _to_decimal(base.get("amount", 0))
    if base_amount < 0:
        raise ValueError("base.amount doit être >= 0.")
    zone_unit = _to_decimal(zone_count.get("unit_price", 0))
    if zone_unit < 0:
        raise ValueError("zone_count.unit_price doit être >= 0.")
    distance_unit = _to_decimal(distance.get("per_km", 0))
    if distance_unit < 0:
        raise ValueError("distance.per_km doit être >= 0.")
    included_km = Decimal(str(distance.get("included_km", 0) or 0))
    if included_km < 0:
        raise ValueError("distance.included_km doit être >= 0.")

    zone_set_id = str(normalized.get("zone_set_id") or "").strip() or None
    zone_enabled = bool(zone_count.get("enabled"))
    if model in {"zone_count", "hybrid_stack"} and zone_enabled and not zone_set_id:
        raise ValueError("zone_set_id requis pour modèle zone_count/hybrid_stack.")
    normalized["zone_set_id"] = zone_set_id

    base_enabled = bool(base.get("enabled"))
    base_amount_value = float(base_amount)
    if model == "distance":
        # Verrouille le contrat: modèle distance = sans base.
        base_enabled = False
        base_amount_value = 0.0

    normalized["components"] = {
        "base": {"enabled": base_enabled, "amount": base_amount_value},
        "zone_count": {
            "enabled": zone_enabled,
            "unit_price": float(zone_unit),
            "strategy": str(
                zone_count.get("strategy") or "pickup_dropoff_diff_or_same"
            ),
            "included_zones": max(1, int(zone_count.get("included_zones") or 1)),
            "max_units": max(1, int(zone_count.get("max_units") or 1)),
        },
        "distance": {
            "enabled": bool(distance.get("enabled")),
            "per_km": float(distance_unit),
            "included_km": float(included_km),
            "rounding": str(distance.get("rounding") or "ceil_0_1"),
        },
    }
    extras = normalized.get("extras")
    normalized["extras"] = extras if isinstance(extras, dict) else {}
    caps = normalized.get("caps")
    caps_map = caps if isinstance(caps, dict) else {}
    minimum = caps_map.get("minimum")
    if minimum is not None and _to_decimal(minimum) < 0:
        raise ValueError("caps.minimum doit être >= 0.")
    maximum = caps_map.get("maximum")
    if maximum is not None and _to_decimal(maximum) < 0:
        raise ValueError("caps.maximum doit être >= 0.")
    normalized["caps"] = {
        "minimum": float(_to_decimal(minimum)) if minimum is not None else None,
        "maximum": float(_to_decimal(maximum)) if maximum is not None else None,
    }
    return normalized
