"""Cas métier du temps de travail chauffeur (domaine pur)."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime

from application.bookings.record_booking_arrival import record_booking_arrival
from domain.work_time.clock import split_minutes_by_zurich_day
from domain.work_time.compensation import PolicyView
from domain.work_time.journeys import SegmentInput
from domain.work_time.periods import monthly_closure_refusal, zurich_period_bounds
from domain.work_time.report import (
    DurationDecisionView,
    apply_ledger_snapshot,
    build_work_time_report,
    closure_parity_errors,
)
from domain.work_time.transport_time import (
    AdjustmentView,
    TransportInput,
    business_policy_date,
    compute_transport_work_time,
)
from services.work_time.report_service import load_cutover

CUTOVER = datetime(2026, 10, 1, tzinfo=UTC)
# 01.10.2026 00:00 Europe/Zurich (CEST, UTC+2).
ZURICH_CUTOVER = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
PERIOD = ("2026-09-01", "2026-10-31")


def _policy(**overrides) -> PolicyView:
    payload = {
        "policy_id": 1,
        "driver_id": None,
        "effective_from": date(2020, 1, 1),
        "effective_until": None,
        "one_way_minutes": 30,
        "round_trip_minutes": 60,
        "intermediate_stop_minutes": 20,
    }
    payload.update(overrides)
    if "transport_flat_minutes" not in payload:
        payload["transport_flat_minutes"] = int(payload["one_way_minutes"])
    return PolicyView(**payload)


def _osrm(**overrides) -> dict:
    """Trace d'un vrai retour OSRM voiture, sans calcul géodésique."""
    payload = {
        "route_provider": "osrm",
        "routing_profile": "driving",
        "routing_status": "ok",
        "route_duration_seconds": 12 * 60,
        "route_distance_m": 4200,
        "route_calculated_at": "2026-09-30T08:00:00Z",
    }
    payload.update(overrides)
    return payload


def _segment(**overrides) -> SegmentInput:
    payload = {
        "booking_id": 1,
        "driver_id": 7,
        "status": "COMPLETED",
        "pickup_location": "Rue A",
        "dropoff_location": "Rue B",
        "arrived_at": datetime(2026, 9, 15, 8, 0, tzinfo=UTC),
        "completed_at": datetime(2026, 9, 15, 8, 40, tzinfo=UTC),
        "scheduled_time": datetime(2026, 9, 15, 10, 0),
    }
    payload.update(overrides)
    return SegmentInput(**payload)


def _report(segments, **kwargs):
    return build_work_time_report(
        segments=segments,
        adjustments=kwargs.get("adjustments", {}),
        manuals=kwargs.get("manuals", []),
        policies=kwargs.get("policies", [_policy()]),
        cutover=kwargs.get("cutover", CUTOVER),
        period_from=kwargs.get("period_from", PERIOD[0]),
        period_to=kwargs.get("period_to", PERIOD[1]),
        duration_decisions=kwargs.get("duration_decisions", {}),
        route_margin_minutes=kwargs.get("route_margin_minutes", 5),
    )


def _entries(report):
    rows = []
    for days in report["days_by_driver"].values():
        for day in days:
            rows.extend(day["entries"])
    return rows


def _line(report, key_part: str):
    return next(
        line
        for line in report["compensation_lines"]
        if key_part in str(line.get("line_key") or "")
        or key_part in str(line.get("journey_key") or "")
    )


def test_record_booking_arrival_est_idempotent():
    booking = type("B", (), {"arrived_at": None})()
    moment = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)
    assert record_booking_arrival(booking, now=moment) is True
    assert booking.arrived_at == moment
    later = datetime(2026, 10, 2, 9, 5, tzinfo=UTC)
    assert record_booking_arrival(booking, now=later) is False
    assert booking.arrived_at == moment


def test_aller_simple_remunere_trente_minutes():
    report = _report([_segment(booking_id=10)])
    line = _line(report, "bk:10")
    assert line["line_key"] == "bk:10"
    assert line["journey_class"] == "one_way"
    assert line["journey_status"] == "complete"
    assert line["rule_type"] == "transport_flat"
    assert line["compensation_status"] == "calculated"
    assert line["compensated_minutes"] == 30
    assert report["kpis"]["completed_segments_count"] == 1
    assert report["kpis"]["compensated_journeys_count"] == 1
    assert report["kpis"]["total_worked_minutes"] == 40


def test_aller_retour_incomplet_nest_pas_un_aller_simple():
    outbound = _segment(
        booking_id=1,
        is_round_trip=True,
        pickup_location="Domicile",
        dropoff_location="Hopital",
    )
    inbound = _segment(
        booking_id=2,
        status="ASSIGNED",
        is_return=True,
        parent_booking_id=1,
        pickup_location="Hopital",
        dropoff_location="Domicile",
        arrived_at=None,
        completed_at=None,
    )
    report = _report([outbound, inbound])
    line = _line(report, "bk:1")
    assert line["line_key"] == "bk:1"
    assert line["compensation_status"] == "calculated"
    assert line["compensated_minutes"] == 30
    assert report["kpis"]["transport_count"] == 1
    assert report["kpis"]["compensated_minutes"] == 30
    assert all(item.get("line_key") != "bk:2" for item in report["compensation_lines"])
    open_rows = [row for row in _entries(report) if row["kind"] == "open"]
    assert len(open_rows) == 1
    assert open_rows[0]["booking_id"] == 2
    assert "transport_not_completed" in open_rows[0]["anomalies"]


def test_multi_etapes_partiel_reste_en_attente():
    segments = [
        _segment(
            booking_id=1,
            route_group_id="g1",
            route_sequence_number=1,
            pickup_location="A",
            dropoff_location="B",
        ),
        _segment(
            booking_id=2,
            route_group_id="g1",
            route_sequence_number=2,
            status="IN_PROGRESS",
            pickup_location="B",
            dropoff_location="C",
            arrived_at=None,
            completed_at=None,
        ),
        _segment(
            booking_id=3,
            route_group_id="g1",
            route_sequence_number=3,
            status="PENDING",
            pickup_location="C",
            dropoff_location="A",
            arrived_at=None,
            completed_at=None,
        ),
    ]
    report = _report(segments)
    paid = [
        line
        for line in report["compensation_lines"]
        if line["compensation_status"] == "calculated"
    ]
    assert [line["line_key"] for line in paid] == ["bk:1"]
    assert paid[0]["compensated_minutes"] == 30
    assert report["kpis"]["transport_count"] == 1
    open_ids = {row["booking_id"] for row in _entries(report) if row["kind"] == "open"}
    assert open_ids == {2, 3}


def test_multi_etapes_complet_paie_etape_intermediaire():
    def leg(booking_id, seq, pickup, dropoff):
        return _segment(
            booking_id=booking_id,
            route_group_id="g2",
            route_sequence_number=seq,
            pickup_location=pickup,
            dropoff_location=dropoff,
            arrived_at=datetime(2026, 9, 20, 8, seq, tzinfo=UTC),
            completed_at=datetime(2026, 9, 20, 8, seq + 20, tzinfo=UTC),
        )

    report = _report([leg(1, 1, "A", "B"), leg(2, 2, "B", "C"), leg(3, 3, "C", "A")])
    keys = sorted(line["line_key"] for line in report["compensation_lines"])
    assert keys == ["bk:1", "bk:2", "bk:3"]
    assert {line["journey_key"] for line in report["compensation_lines"]} == {"rg:g2"}
    assert all(
        line["compensated_minutes"] == 30 for line in report["compensation_lines"]
    )
    assert report["kpis"]["completed_segments_count"] == 3
    assert report["kpis"]["transport_count"] == 3
    assert report["kpis"]["compensated_minutes"] == 90


def test_retour_annule_redevient_un_aller_simple():
    outbound = _segment(booking_id=1, is_round_trip=True)
    cancelled = _segment(
        booking_id=2,
        status="CANCELED",
        is_return=True,
        parent_booking_id=1,
        arrived_at=None,
        completed_at=None,
    )
    line = _line(_report([outbound, cancelled]), "bk:1")
    assert line["journey_class"] == "one_way"
    assert line["compensation_status"] == "calculated"
    assert line["compensated_minutes"] == 30


def test_course_terminee_sans_heure_de_fin_reste_visible():
    for status in ("COMPLETED", "RETURN_COMPLETED"):
        report = _report(
            [
                _segment(
                    booking_id=50,
                    status=status,
                    arrived_at=None,
                    boarded_at=None,
                    completed_at=None,
                    scheduled_time=datetime(2026, 9, 12, 9, 30),
                )
            ]
        )
        rows = _entries(report)
        assert len(rows) == 1
        assert rows[0]["quality"] == "incomplete"
        assert rows[0]["worked_minutes"] is None
        assert "completion_time_missing" in rows[0]["anomalies"]
        assert report["kpis"]["incomplete_segments_count"] == 1
        assert report["kpis"]["completion_time_missing_count"] == 1
        assert report["kpis"]["total_worked_minutes"] == 0
        assert report["kpis"]["transport_count"] == 1
        assert report["kpis"]["real_transport_minutes"] == 0
        assert report["kpis"]["flat_transport_minutes"] == 30
        assert report["kpis"]["review_count_real"] == 1
        assert report["compensation_lines"][0]["compensated_minutes"] == 30
        day = report["days_by_driver"]["7"][0]
        assert day["totals"]["worked_minutes"] is None
        assert day["totals"]["real_minutes"] is None
        assert day["totals"]["flat_minutes"] == 30


def test_bascule_est_un_instant_utc_minuit_zurich():
    """Naïf = UTC. 00:00Z le 1er octobre n'est pas minuit à Genève."""
    assert load_cutover("2026-09-30T22:00:00Z") == ZURICH_CUTOVER
    assert load_cutover("2026-09-30T22:00:00+00:00") == ZURICH_CUTOVER
    assert load_cutover("2026-09-30T22:00:00") == ZURICH_CUTOVER
    assert load_cutover("2026-10-01T00:00:00Z") == datetime(2026, 10, 1, tzinfo=UTC)
    assert load_cutover("2026-10-01T00:00:00Z") != ZURICH_CUTOVER
    from config import Config

    assert load_cutover(Config.WORK_TIME_ARRIVED_AT_CUTOVER_AT) == ZURICH_CUTOVER


def test_bascule_zurich_23h59_00h00_00h01():
    """23:59 le 30.09 reste estimé. Minuit et 00:01 le 01.10 sont après la bascule."""
    before = datetime(2026, 9, 30, 21, 59, tzinfo=UTC)
    boarded = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)
    historical = compute_transport_work_time(
        TransportInput(
            booking_id=1,
            arrived_at=None,
            boarded_at=boarded,
            completed_at=before,
            scheduled_time=datetime(2026, 9, 30, 23, 0),
            status="COMPLETED",
        ),
        None,
        ZURICH_CUTOVER,
    )
    assert historical.quality == "estimated_historical"

    at_midnight = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
    one_minute = datetime(2026, 9, 30, 22, 1, tzinfo=UTC)
    still_in_old_window_if_utc_midnight = datetime(2026, 9, 30, 23, 30, tzinfo=UTC)
    for instant in (at_midnight, one_minute, still_in_old_window_if_utc_midnight):
        result = compute_transport_work_time(
            TransportInput(
                booking_id=2,
                arrived_at=None,
                boarded_at=datetime(2026, 9, 30, 20, 0, tzinfo=UTC),
                completed_at=instant,
                scheduled_time=datetime(2026, 10, 1, 0, 0),
                status="COMPLETED",
            ),
            None,
            ZURICH_CUTOVER,
        )
        assert result.quality == "incomplete", instant
        assert result.worked_minutes is None
        assert "arrival_not_recorded" in result.anomalies


def test_snapshot_survit_si_la_course_change_de_jour():
    trip = _segment(
        booking_id=1,
        arrived_at=datetime(2026, 9, 15, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 15, 8, 25, tzinfo=UTC),
    )
    frozen = _report([trip], policies=[_policy(policy_id=1, one_way_minutes=30)])
    snapshot = [
        {**line, "driver_id": 7, "finalized_at": "2026-10-02T08:00:00Z"}
        for line in frozen["compensation_lines"]
    ]
    moved = _segment(
        booking_id=1,
        arrived_at=datetime(2026, 9, 20, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 20, 8, 25, tzinfo=UTC),
    )
    later = _report([moved], policies=[_policy(policy_id=9, one_way_minutes=45)])
    apply_ledger_snapshot(later, snapshot)
    assert later["kpis"]["compensated_minutes"] == 30
    assert sum(int(row["compensated_minutes"] or 0) for row in _entries(later)) == 30
    assert any(
        "ledger_detached_from_live_entry" in row["anomalies"] for row in _entries(later)
    )


def test_anomalie_propose_le_trajet_calcule_plus_cinq_minutes():
    report = _report(
        [
            _segment(
                arrived_at=None,
                boarded_at=None,
                completed_at=datetime(2026, 9, 28, 10, 0, tzinfo=UTC),
                **_osrm(route_duration_seconds=12 * 60, route_distance_m=5100),
            )
        ]
    )
    row = _entries(report)[0]
    assert row["worked_minutes"] is None
    assert row["work_time_status"] == "pending_validation"
    assert row["route_minutes"] == 12
    assert row["margin_minutes"] == 5
    assert row["proposed_worked_minutes"] == 17
    assert row["worked_minutes_source"] == "route_estimate"
    assert "arrival_not_recorded" in row["anomalies"]
    assert report["kpis"]["total_worked_minutes"] == 0
    assert report["kpis"]["pending_validation_minutes"] == 17
    assert report["kpis"]["real_transport_minutes"] == 0
    assert report["kpis"]["flat_transport_minutes"] == 30
    assert report["kpis"]["compensated_minutes"] == 30


def test_coordonnees_seules_ne_fabriquent_pas_de_duree():
    """HUG → Pictet : sans réponse OSRM, aucune minute n'est inventée."""
    report = _report(
        [
            _segment(
                arrived_at=None,
                boarded_at=None,
                completed_at=datetime(2026, 9, 27, 22, 57, tzinfo=UTC),
                duration_seconds=6 * 60,
                pickup_location="Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
                dropoff_location="Avenue Ernest-Pictet 9, 1203, Genève",
                pickup_lat=46.19226,
                pickup_lon=6.14262,
                dropoff_lat=46.2117141,
                dropoff_lon=6.1262074,
                routing_status="unavailable",
            )
        ]
    )
    row = _entries(report)[0]
    assert row["work_time_status"] == "incomplete"
    assert row["proposed_worked_minutes"] is None
    assert row["routing_status"] == "unavailable"
    assert row["worked_minutes"] is None


def test_adresses_sans_gps_sont_geocodees_puis_osrm(monkeypatch):
    from services.work_time import report_service

    report_service._DRIVING_ROUTE_CACHE.clear()
    report_service._GEOCODE_CACHE.clear()
    geocodes: list[str] = []

    def geocode(address, **_kwargs):
        geocodes.append(address)
        if "Courbes" in address:
            return {"lat": 46.2764, "lon": 6.2181}
        if "Gentil" in address:
            return {"lat": 46.19226, "lon": 6.14262}
        return None

    monkeypatch.setattr("services.geolocation.maps.geocode_address", geocode)
    monkeypatch.setattr(
        "services.geolocation.osrm.route_info",
        lambda *_args, **_kwargs: {
            "duration": 900.0,
            "distance": 12000.0,
            "fallback": False,
        },
    )
    monkeypatch.setattr(
        "services.geolocation.historical_eta.get_historical_duration",
        lambda *_args, **_kwargs: None,
    )
    booking = type(
        "Booking",
        (),
        {
            "id": 45711,
            "pickup_lat": None,
            "pickup_lon": None,
            "dropoff_lat": None,
            "dropoff_lon": None,
            "pickup_location": "Chemin des Courbes 9, 1247, Anières",
            "dropoff_location": "Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
        },
    )()
    trace = report_service._driving_route_trace(booking)
    assert geocodes == [
        "Chemin des Courbes 9, 1247, Anières",
        "Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
    ]
    assert trace["routing_status"] == "ok"
    assert trace["route_provider"] == "osrm"
    assert trace["route_duration_seconds"] == int(900.0 * 1.55)


def test_geocodage_impossible_ne_fabrique_pas_de_duree(monkeypatch):
    from services.work_time import report_service

    report_service._DRIVING_ROUTE_CACHE.clear()
    report_service._GEOCODE_CACHE.clear()
    monkeypatch.setattr(
        "services.geolocation.maps.geocode_address", lambda *_args, **_kwargs: None
    )
    called = {"osrm": False}

    def route(*_args, **_kwargs):
        called["osrm"] = True
        return {"duration": 400, "distance": 3000, "fallback": False}

    monkeypatch.setattr("services.geolocation.osrm.route_info", route)
    booking = type(
        "Booking",
        (),
        {
            "id": 45711,
            "pickup_lat": None,
            "pickup_lon": None,
            "dropoff_lat": None,
            "dropoff_lon": None,
            "pickup_location": "Chemin des Courbes 9, 1247, Anières",
            "dropoff_location": "Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
        },
    )()
    trace = report_service._driving_route_trace(booking)
    assert called["osrm"] is False
    assert trace["routing_status"] == "unavailable"


def test_echec_ou_repli_osrm_ne_propose_aucune_duree(monkeypatch):
    from services.work_time import report_service

    report_service._DRIVING_ROUTE_CACHE.clear()

    def repli(*_args, **_kwargs):
        return {"duration": 400, "distance": 3000, "fallback": True}

    monkeypatch.setattr("services.geolocation.osrm.route_info", repli)
    booking = type(
        "Booking",
        (),
        {
            "id": 46791,
            "pickup_lat": 46.19226,
            "pickup_lon": 6.14262,
            "dropoff_lat": 46.2117141,
            "dropoff_lon": 6.1262074,
        },
    )()
    trace = report_service._driving_route_trace(booking)
    assert trace["routing_status"] == "unavailable"
    assert trace["route_duration_seconds"] is None
    assert trace["route_distance_m"] is None


def test_cache_conserve_la_duree_estimee_pas_le_temps_a_vide(monkeypatch):
    """354 s OSRM à vide ne sont pas le temps de trajet : on garde l'estimation réservation."""
    from services.work_time import report_service

    report_service._DRIVING_ROUTE_CACHE.clear()
    calls = {"n": 0}

    def route(*_args, **kwargs):
        calls["n"] += 1
        assert kwargs["profile"] == "driving"
        return {"duration": 354.3, "distance": 3286.4, "fallback": False}

    monkeypatch.setattr("services.geolocation.osrm.route_info", route)
    monkeypatch.setattr(
        "services.geolocation.historical_eta.get_historical_duration",
        lambda *_args, **_kwargs: None,
    )
    booking = type(
        "Booking",
        (),
        {
            "id": 46791,
            "pickup_lat": 46.19226,
            "pickup_lon": 6.14262,
            "dropoff_lat": 46.2117141,
            "dropoff_lon": 6.1262074,
        },
    )()
    first = report_service._driving_route_trace(booking)
    second = report_service._driving_route_trace(booking)
    assert calls["n"] == 1
    assert first == second
    assert first["route_provider"] == "osrm"
    assert first["routing_profile"] == "driving"
    assert first["route_duration_seconds"] == int(354.3 * 1.55)
    assert first["route_duration_seconds"] != round(354.3)
    assert first["route_distance_m"] == 3286
    assert first["routing_status"] == "ok"


def test_historique_reel_prime_sur_le_temps_osrm_a_vide(monkeypatch):
    from services.work_time import report_service

    report_service._DRIVING_ROUTE_CACHE.clear()
    monkeypatch.setattr(
        "services.geolocation.osrm.route_info",
        lambda *_args, **_kwargs: {
            "duration": 354.3,
            "distance": 3286.4,
            "fallback": False,
        },
    )
    monkeypatch.setattr(
        "services.geolocation.historical_eta.get_historical_duration",
        lambda *_args, **_kwargs: 12 * 60,
    )
    booking = type(
        "Booking",
        (),
        {
            "id": 46791,
            "pickup_lat": 46.19,
            "pickup_lon": 6.14,
            "dropoff_lat": 46.21,
            "dropoff_lon": 6.12,
        },
    )()
    trace = report_service._driving_route_trace(booking)
    assert trace["route_duration_seconds"] == 12 * 60


def test_duree_proposee_vient_de_la_reponse_osrm_voiture():
    """La marge s'ajoute une fois à la durée renvoyée par le routeur, ici 354 s."""
    report = _report(
        [
            _segment(
                arrived_at=None,
                boarded_at=None,
                completed_at=datetime(2026, 9, 27, 22, 57, tzinfo=UTC),
                pickup_lat=46.19226,
                pickup_lon=6.14262,
                dropoff_lat=46.2117141,
                dropoff_lon=6.1262074,
                **_osrm(route_duration_seconds=354, route_distance_m=3286),
            )
        ]
    )
    row = _entries(report)[0]
    assert row["route_duration_seconds"] == 354
    assert row["route_distance_m"] == 3286
    assert row["route_minutes"] == round(354 / 60)
    assert row["margin_minutes"] == 5
    assert row["proposed_worked_minutes"] == row["route_minutes"] + 5
    assert row["route_provider"] == "osrm"
    assert row["routing_profile"] == "driving"
    assert row["routing_status"] == "ok"


def test_proposition_separe_trajet_et_marge_sans_creer_d_horaires():
    """Terminé à 10:00, trajet 15 min + marge 5 min → 20 min proposées, pas d'arrivée inventée."""
    report = _report(
        [
            _segment(
                arrived_at=None,
                boarded_at=None,
                completed_at=datetime(2026, 9, 28, 8, 0, tzinfo=UTC),
                **_osrm(route_duration_seconds=15 * 60, route_distance_m=8000),
            )
        ]
    )
    row = _entries(report)[0]
    assert row["route_minutes"] == 15
    assert row["margin_minutes"] == 5
    assert row["proposed_worked_minutes"] == 20
    assert row["worked_minutes"] is None
    assert "proposed_arrived_at" not in row


def test_valider_la_proposition_retient_la_duree_et_le_forfait():
    report = _report(
        [
            _segment(
                arrived_at=None,
                boarded_at=None,
                completed_at=datetime(2026, 9, 28, 10, 0, tzinfo=UTC),
                duration_seconds=12 * 60,
            )
        ],
        duration_decisions={
            1: DurationDecisionView(
                validated_worked_minutes=17,
                source="validated_route_estimate",
                proposed_worked_minutes=17,
                route_minutes=12,
                margin_minutes=5,
                route_provider="osrm",
            )
        },
    )
    row = _entries(report)[0]
    assert row["work_time_status"] == "validated_estimate"
    assert row["worked_minutes"] == 17
    assert "arrival_not_recorded" not in row["anomalies"]
    assert "arrival_not_recorded" in row["quality_notes"]
    assert report["kpis"]["total_worked_minutes"] == 17
    assert report["kpis"]["pending_validation_minutes"] == 0
    assert report["kpis"]["compensated_minutes"] == 30


def test_rectifier_retient_la_duree_admin_sans_changer_le_forfait():
    report = _report(
        [
            _segment(
                arrived_at=None,
                boarded_at=None,
                completed_at=datetime(2026, 9, 28, 10, 0, tzinfo=UTC),
                duration_seconds=12 * 60,
            )
        ],
        duration_decisions={
            1: DurationDecisionView(
                validated_worked_minutes=18,
                source="admin_adjustment",
                proposed_worked_minutes=17,
                route_minutes=12,
                margin_minutes=5,
                reason="course plus longue",
            )
        },
    )
    row = _entries(report)[0]
    assert row["work_time_status"] == "adjusted"
    assert row["worked_minutes"] == 18
    assert report["kpis"]["compensated_minutes"] == 30


def test_temps_valide_ne_paie_pas_un_trajet_reparti():
    report = _report(
        [
            _segment(
                booking_id=1,
                driver_id=7,
                arrived_at=None,
                completed_at=datetime(2026, 9, 28, 10, 0, tzinfo=UTC),
                duration_seconds=4 * 60,
                pickup_location="HUG",
                dropoff_location="Pictet",
            ),
            _segment(
                booking_id=2,
                driver_id=8,
                is_return=True,
                parent_booking_id=1,
                arrived_at=datetime(2026, 9, 28, 11, 0, tzinfo=UTC),
                completed_at=datetime(2026, 9, 28, 11, 20, tzinfo=UTC),
                pickup_location="Pictet",
                dropoff_location="HUG",
            ),
        ],
        duration_decisions={
            1: DurationDecisionView(
                validated_worked_minutes=9,
                source="validated_route_estimate",
                proposed_worked_minutes=9,
                route_minutes=4,
                margin_minutes=5,
            )
        },
    )
    outbound = next(row for row in _entries(report) if row["booking_id"] == 1)
    assert outbound["worked_minutes"] == 9
    assert outbound["compensated_minutes"] == 30
    assert outbound["flat_minutes"] == 30
    assert "journey_split_across_drivers" not in outbound["anomalies"]
    assert report["kpis"]["compensated_minutes"] == 60
    assert report["kpis"]["flat_transport_minutes"] == 60
    assert report["kpis"]["review_count_flat"] == 0


def test_mode_temps_valide_remunere_les_minutes_retenues():
    report = _report(
        [
            _segment(
                arrived_at=None,
                boarded_at=None,
                completed_at=datetime(2026, 9, 28, 10, 0, tzinfo=UTC),
                duration_seconds=12 * 60,
            )
        ],
        policies=[_policy(mode="validated_work_time")],
        duration_decisions={
            1: DurationDecisionView(
                validated_worked_minutes=17,
                source="validated_route_estimate",
            )
        },
    )
    assert report["kpis"]["compensated_minutes"] == 17
    assert report["kpis"]["total_worked_minutes"] == 17


def test_historique_estime_avant_bascule_et_anomalie_apres():
    historical = compute_transport_work_time(
        TransportInput(
            booking_id=1,
            arrived_at=None,
            boarded_at=datetime(2026, 9, 1, 7, 0, tzinfo=UTC),
            completed_at=datetime(2026, 9, 1, 8, 0, tzinfo=UTC),
            scheduled_time=datetime(2026, 9, 1, 9, 0),
            status="COMPLETED",
        ),
        None,
        CUTOVER,
    )
    assert historical.quality == "estimated_historical"
    assert historical.worked_minutes == 60
    assert historical.worked_minutes_source == "historical_pickup_to_completed"

    after = compute_transport_work_time(
        TransportInput(
            booking_id=2,
            arrived_at=None,
            boarded_at=datetime(2026, 10, 2, 7, 0, tzinfo=UTC),
            completed_at=datetime(2026, 10, 2, 8, 0, tzinfo=UTC),
            scheduled_time=datetime(2026, 10, 2, 9, 0),
            status="COMPLETED",
        ),
        None,
        CUTOVER,
    )
    assert after.quality == "incomplete"
    assert after.worked_minutes is None
    assert "arrival_not_recorded" in after.anomalies


def test_course_a_cheval_sur_minuit_ventile_les_minutes():
    # 30.09 23:50 → 01.10 00:20, heure de Zurich (UTC+2).
    start = datetime(2026, 9, 30, 21, 50, tzinfo=UTC)
    end = datetime(2026, 9, 30, 22, 20, tzinfo=UTC)
    slices = dict(split_minutes_by_zurich_day(start, end))
    assert slices == {"2026-09-30": 10, "2026-10-01": 20}
    assert sum(slices.values()) == 30

    report = _report(
        [_segment(booking_id=8, arrived_at=start, completed_at=end)],
        period_from="2026-09-30",
        period_to="2026-10-01",
    )
    by_day = {row["date"]: row["worked_minutes"] for row in _entries(report)}
    assert by_day["2026-09-30"] == 10
    assert by_day["2026-10-01"] == 20
    assert report["kpis"]["total_worked_minutes"] == 30
    assert report["kpis"]["completed_segments_count"] == 1
    line = _line(report, "bk:8")
    assert line["accounting_date"] == "2026-10-01"
    assert line["compensated_minutes"] == 30


def test_changement_heure_ete_et_hiver():
    # 29.03.2026 01:30 → 03:30 Zurich : une heure d'horloge sautée, 60 min réelles.
    spring = split_minutes_by_zurich_day(
        datetime(2026, 3, 29, 0, 30, tzinfo=UTC),
        datetime(2026, 3, 29, 1, 30, tzinfo=UTC),
    )
    assert sum(minutes for _, minutes in spring) == 60
    # 25.10.2026 01:30 → 03:30 : heure répétée, 180 min réelles.
    autumn = split_minutes_by_zurich_day(
        datetime(2026, 10, 24, 23, 30, tzinfo=UTC),
        datetime(2026, 10, 25, 2, 30, tzinfo=UTC),
    )
    assert sum(minutes for _, minutes in autumn) == 180


def test_bornes_de_periode_inclusives_en_zurich():
    utc_start, utc_end, local_start, local_end = zurich_period_bounds(
        "2026-03-29", "2026-03-29"
    )
    assert local_start == datetime(2026, 3, 29, 0, 0)
    assert local_end == datetime(2026, 3, 30, 0, 0)
    assert utc_end > utc_start
    assert (utc_end - utc_start).total_seconds() == 23 * 3600


def test_correction_snapshot_complet_remplace_la_duree():
    original = TransportInput(
        booking_id=1,
        arrived_at=datetime(2026, 10, 2, 8, 12, tzinfo=UTC),
        boarded_at=None,
        completed_at=datetime(2026, 10, 2, 8, 42, tzinfo=UTC),
        scheduled_time=None,
        status="COMPLETED",
    )
    first = AdjustmentView(
        corrected_arrived_at=datetime(2026, 10, 2, 8, 15, tzinfo=UTC),
        corrected_completed_at=datetime(2026, 10, 2, 8, 42, tzinfo=UTC),
        reason="arrivée",
    )
    second = AdjustmentView(
        corrected_arrived_at=datetime(2026, 10, 2, 8, 15, tzinfo=UTC),
        corrected_completed_at=datetime(2026, 10, 2, 9, 5, tzinfo=UTC),
        reason="fin",
    )
    result = compute_transport_work_time(original, second, CUTOVER)
    assert result.quality == "adjusted"
    assert result.worked_minutes == 50
    assert result.effective_arrived_at == first.corrected_arrived_at


def test_regle_manquante_ne_pas_inventer_de_forfait():
    report = _report([_segment()], policies=[])
    line = _line(report, "bk:")
    assert line["compensation_status"] == "requires_review"
    assert line["compensated_minutes"] == 0
    assert report["kpis"]["review_journeys_count"] == 1


def test_invariant_somme_du_detail_egale_le_resume():
    report = _report(
        [
            _segment(booking_id=1),
            _segment(
                booking_id=2,
                driver_id=8,
                arrived_at=datetime(2026, 9, 16, 9, 0, tzinfo=UTC),
                completed_at=datetime(2026, 9, 16, 9, 25, tzinfo=UTC),
            ),
        ]
    )
    rows = _entries(report)
    assert (
        sum(int(row["worked_minutes"] or 0) for row in rows)
        == report["kpis"]["total_worked_minutes"]
    )
    assert (
        sum(int(row["compensated_minutes"] or 0) for row in rows)
        == report["kpis"]["compensated_minutes"]
    )
    assert report["kpis"]["compensated_minutes"] == 60


def test_chevauchement_signale_sans_fusion():
    report = _report(
        [
            _segment(
                booking_id=1,
                arrived_at=datetime(2026, 9, 18, 8, 0, tzinfo=UTC),
                completed_at=datetime(2026, 9, 18, 9, 0, tzinfo=UTC),
            ),
            _segment(
                booking_id=2,
                arrived_at=datetime(2026, 9, 18, 8, 30, tzinfo=UTC),
                completed_at=datetime(2026, 9, 18, 9, 30, tzinfo=UTC),
            ),
        ]
    )
    flagged = [row for row in _entries(report) if "overlap" in row["anomalies"]]
    assert len(flagged) == 2
    assert sum(row["worked_minutes"] for row in flagged) == 120


def test_structure_indeterminee_et_trajet_multi_chauffeurs():
    unclear = [
        _segment(
            booking_id=1,
            route_group_id="odd",
            route_sequence_number=1,
            pickup_location="Clinique",
            dropoff_location="Foyer",
        ),
        _segment(
            booking_id=2,
            route_group_id="odd",
            route_sequence_number=2,
            pickup_location="Foyer",
            dropoff_location="Ecole",
            arrived_at=datetime(2026, 9, 15, 9, 0, tzinfo=UTC),
            completed_at=datetime(2026, 9, 15, 9, 20, tzinfo=UTC),
        ),
    ]
    unclear_report = _report(unclear)
    assert sorted(
        line["line_key"] for line in unclear_report["compensation_lines"]
    ) == [
        "bk:1",
        "bk:2",
    ]
    assert unclear_report["kpis"]["compensated_minutes"] == 60

    split = [
        _segment(
            booking_id=11, route_group_id="split", route_sequence_number=1, driver_id=7
        ),
        _segment(
            booking_id=12,
            route_group_id="split",
            route_sequence_number=2,
            driver_id=8,
            is_return=True,
            pickup_location="Rue B",
            dropoff_location="Rue A",
            arrived_at=datetime(2026, 9, 15, 11, 0, tzinfo=UTC),
            completed_at=datetime(2026, 9, 15, 11, 30, tzinfo=UTC),
        ),
    ]
    split_report = _report(split)
    assert sorted(line["line_key"] for line in split_report["compensation_lines"]) == [
        "bk:11",
        "bk:12",
    ]
    assert all(
        line["compensation_status"] == "calculated"
        for line in split_report["compensation_lines"]
    )
    assert all(
        line["flat_minutes"] == 30 for line in split_report["compensation_lines"]
    )
    assert split_report["kpis"]["compensated_minutes"] == 60
    assert split_report["kpis"]["flat_transport_minutes"] == 60
    assert split_report["kpis"]["review_count_flat"] == 0


def test_ancienne_course_garde_lancienne_regle():
    old = _policy(
        policy_id=1,
        effective_from=date(2026, 1, 1),
        effective_until=date(2026, 10, 1),
        one_way_minutes=30,
    )
    current = _policy(
        policy_id=2,
        effective_from=date(2026, 10, 1),
        effective_until=None,
        one_way_minutes=45,
    )
    september = _segment(
        booking_id=1,
        arrived_at=datetime(2026, 9, 15, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 15, 8, 20, tzinfo=UTC),
    )
    october = _segment(
        booking_id=2,
        arrived_at=datetime(2026, 10, 5, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 10, 5, 8, 20, tzinfo=UTC),
    )
    report = _report([september, october], policies=[old, current])
    assert _line(report, "bk:1")["policy_id"] == 1
    assert _line(report, "bk:1")["compensated_minutes"] == 30
    assert _line(report, "bk:2")["policy_id"] == 2
    assert _line(report, "bk:2")["compensated_minutes"] == 45


def test_periode_finalisee_ignore_une_regle_plus_recente():
    """Une fiche de septembre figée à 30 min ne devient pas 45 si la règle change."""
    trip = _segment(
        booking_id=1,
        arrived_at=datetime(2026, 9, 15, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 15, 8, 25, tzinfo=UTC),
    )
    frozen = _report([trip], policies=[_policy(policy_id=1, one_way_minutes=30)])
    assert frozen["kpis"]["compensated_minutes"] == 30
    assert frozen["compensation_lines"][0]["driver_id"] == 7
    snapshot = []
    for line in frozen["compensation_lines"]:
        snapshot.append(
            {
                **line,
                "driver_id": 7,
                "generated_at": "2026-10-02T08:00:00Z",
                "finalized_at": "2026-10-02T08:00:00Z",
            }
        )
    recalculated = _report([trip], policies=[_policy(policy_id=9, one_way_minutes=45)])
    assert recalculated["kpis"]["compensated_minutes"] == 45
    apply_ledger_snapshot(recalculated, snapshot)
    assert recalculated["compensation_source"] == "ledger"
    assert recalculated["period_finalized"] is True
    assert recalculated["kpis"]["compensated_minutes"] == 30
    assert (
        sum(int(row["compensated_minutes"] or 0) for row in _entries(recalculated))
        == 30
    )


def _ledger(report):
    return [
        {
            **line,
            "driver_id": line.get("driver_id") or 7,
            "generated_at": "2026-10-02T08:00:00Z",
            "finalized_at": "2026-10-02T08:00:00Z",
        }
        for line in report["compensation_lines"]
    ]


def _course_osrm():
    """Aller simple déjà estimé : 6 min de trajet + 5 min offertes = 11 proposées."""
    return _segment(
        arrived_at=None,
        boarded_at=None,
        completed_at=datetime(2026, 9, 28, 8, 57, tzinfo=UTC),
        pickup_lat=46.19226,
        pickup_lon=6.14262,
        dropoff_lat=46.2117141,
        dropoff_lon=6.1262074,
        **_osrm(route_duration_seconds=354, route_distance_m=3286),
    )


def test_parcours_validation_remuneration_et_cloture():
    """À valider, puis Valider ou Rectifier, puis paie, puis clôture.

    Le moteur OSRM n'est pas recalculé ici : la trace est déjà une réponse voiture.
    """
    trip = _course_osrm()
    pending = _report([trip])
    row = _entries(pending)[0]
    assert row["work_time_status"] == "pending_validation"
    assert row["route_minutes"] == 6
    assert row["margin_minutes"] == 5
    assert row["proposed_worked_minutes"] == 11
    assert row["worked_minutes"] is None
    assert pending["kpis"]["total_worked_minutes"] == 0
    assert pending["kpis"]["pending_validation_minutes"] == 11
    assert pending["kpis"]["real_transport_minutes"] == 0
    assert pending["kpis"]["flat_transport_minutes"] == 30
    assert pending["kpis"]["compensated_minutes"] == 30

    validated_decision = DurationDecisionView(
        validated_worked_minutes=11,
        source="validated_route_estimate",
        proposed_worked_minutes=11,
        route_minutes=6,
        margin_minutes=5,
        route_provider="osrm",
    )
    validated = _report([trip], duration_decisions={1: validated_decision})
    validated_row = _entries(validated)[0]
    assert validated_row["work_time_status"] == "validated_estimate"
    assert validated_row["worked_minutes"] == 11
    assert validated["kpis"]["total_worked_minutes"] == 11
    assert validated["kpis"]["worked_validated_minutes"] == 11
    assert validated["kpis"]["pending_validation_minutes"] == 0
    assert validated["kpis"]["compensated_minutes"] == 30

    rectified = _report(
        [trip],
        duration_decisions={
            1: DurationDecisionView(
                validated_worked_minutes=16,
                source="admin_adjustment",
                proposed_worked_minutes=11,
                route_minutes=6,
                margin_minutes=5,
                route_provider="osrm",
                reason="circulation réelle",
            )
        },
    )
    rectified_row = _entries(rectified)[0]
    assert rectified_row["work_time_status"] == "adjusted"
    assert rectified_row["worked_minutes"] == 16
    assert rectified["kpis"]["total_worked_minutes"] == 16
    assert rectified["kpis"]["worked_adjusted_minutes"] == 16
    assert rectified["kpis"]["worked_validated_minutes"] == 0
    assert rectified["kpis"]["compensated_minutes"] == 30

    paid_like_time = _report(
        [trip],
        policies=[_policy(mode="validated_work_time")],
        duration_decisions={1: validated_decision},
    )
    assert paid_like_time["kpis"]["total_worked_minutes"] == 11
    assert paid_like_time["kpis"]["compensated_minutes"] == 11
    paid_rectified = _report(
        [trip],
        policies=[_policy(mode="validated_work_time")],
        duration_decisions={
            1: DurationDecisionView(
                validated_worked_minutes=16,
                source="admin_adjustment",
                reason="circulation réelle",
            )
        },
    )
    assert paid_rectified["kpis"]["total_worked_minutes"] == 16
    assert paid_rectified["kpis"]["compensated_minutes"] == 16

    live_after_rule_change = _report(
        [trip],
        policies=[_policy(policy_id=9, one_way_minutes=45)],
        duration_decisions={1: validated_decision},
    )
    assert live_after_rule_change["kpis"]["compensated_minutes"] == 45
    apply_ledger_snapshot(live_after_rule_change, _ledger(validated))
    assert live_after_rule_change["compensation_source"] == "ledger"
    assert live_after_rule_change["period_finalized"] is True
    assert live_after_rule_change["kpis"]["compensated_minutes"] == 30
    assert live_after_rule_change["kpis"]["total_worked_minutes"] == 11

    closed_too_early = _report(
        [trip],
        duration_decisions={1: validated_decision},
    )
    apply_ledger_snapshot(closed_too_early, _ledger(pending))
    assert closed_too_early["kpis"]["compensated_minutes"] == 30
    assert closed_too_early["kpis"]["real_transport_minutes"] == 0
    assert _entries(closed_too_early)[0]["worked_minutes"] == 11


def test_saisie_manuelle_annulee_nest_pas_comptee():
    manual = {
        "entry_id": 4,
        "driver_id": 7,
        "work_type": "waiting",
        "duration_minutes": 40,
        "work_date": "2026-09-15",
        "description": "Attente",
        "started_at": datetime(2026, 9, 15, 14, 0, tzinfo=UTC),
        "ended_at": datetime(2026, 9, 15, 14, 40, tzinfo=UTC),
        "cancelled": False,
    }
    policies = [_policy(work_type_rules={"waiting": {"mode": "real_time"}})]
    counted = _report([], manuals=[manual], policies=policies)
    assert counted["kpis"]["manual_minutes"] == 40
    assert counted["kpis"]["compensated_minutes"] == 40
    skipped = _report([], manuals=[{**manual, "cancelled": True}], policies=policies)
    assert skipped["kpis"]["manual_minutes"] == 0
    assert skipped["kpis"]["total_worked_minutes"] == 0


def test_date_de_politique_prend_la_fin_effective_puis_la_prevue():
    scheduled = datetime(2026, 9, 30, 23, 30)
    corrected = datetime(2026, 9, 30, 22, 20, tzinfo=UTC)
    assert (
        business_policy_date(
            effective_completed_at=corrected,
            corrected_completed_at=corrected,
            scheduled_time=scheduled,
        )
        == "2026-10-01"
    )
    assert (
        business_policy_date(
            effective_completed_at=None,
            corrected_completed_at=corrected,
            scheduled_time=scheduled,
        )
        == "2026-10-01"
    )
    assert (
        business_policy_date(
            effective_completed_at=None,
            corrected_completed_at=None,
            scheduled_time=scheduled,
        )
        == "2026-09-30"
    )


def test_fin_rectifiee_le_1er_octobre_prend_la_politique_du_jour():
    trip = _segment(
        booking_id=3,
        scheduled_time=datetime(2026, 9, 30, 23, 30),
        arrived_at=datetime(2026, 9, 30, 21, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 30, 21, 40, tzinfo=UTC),
    )
    adjustment = AdjustmentView(
        corrected_arrived_at=datetime(2026, 9, 30, 21, 50, tzinfo=UTC),
        corrected_completed_at=datetime(2026, 9, 30, 22, 20, tzinfo=UTC),
        reason="fin après minuit",
    )
    september = _policy(
        policy_id=1,
        effective_from=date(2026, 9, 1),
        effective_until=date(2026, 10, 1),
        one_way_minutes=30,
    )
    october = _policy(
        policy_id=2,
        effective_from=date(2026, 10, 1),
        one_way_minutes=45,
    )
    report = _report([trip], adjustments={3: adjustment}, policies=[september, october])
    line = _line(report, "bk:3")
    assert line["accounting_date"] == "2026-10-01"
    assert line["policy_id"] == 2
    assert line["compensated_minutes"] == 45


def test_deux_politiques_du_meme_mois_suivent_chacune_sa_date():
    early = _policy(
        policy_id=1,
        effective_from=date(2026, 9, 1),
        effective_until=date(2026, 9, 16),
        one_way_minutes=30,
    )
    late = _policy(
        policy_id=2,
        effective_from=date(2026, 9, 16),
        one_way_minutes=45,
    )
    first = _segment(
        booking_id=1,
        scheduled_time=datetime(2026, 9, 10, 10, 0),
        arrived_at=datetime(2026, 9, 10, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 10, 8, 40, tzinfo=UTC),
    )
    second = _segment(
        booking_id=2,
        scheduled_time=datetime(2026, 9, 20, 10, 0),
        arrived_at=datetime(2026, 9, 20, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 9, 20, 8, 40, tzinfo=UTC),
    )
    report = _report([first, second], policies=[early, late])
    assert _line(report, "bk:1")["policy_id"] == 1
    assert _line(report, "bk:1")["compensated_minutes"] == 30
    assert _line(report, "bk:2")["policy_id"] == 2
    assert _line(report, "bk:2")["compensated_minutes"] == 45
    assert report["contractual_rules"]["version_count"] == 2


def test_temps_ajoute_sans_regle_reste_reel_et_sort_du_forfait():
    manual = {
        "entry_id": 8,
        "driver_id": 7,
        "work_type": "cleaning",
        "duration_minutes": 20,
        "work_date": "2026-09-15",
        "description": "Nettoyage",
        "cancelled": False,
    }
    report = _report([], manuals=[manual], policies=[_policy(work_type_rules={})])
    row = _entries(report)[0]
    assert row["real_minutes"] == 20
    assert row["flat_minutes"] is None
    assert row["flat_status"] == "requires_review"
    assert row["line_key"] == "manual:8"
    assert "manual_compensation_rule_missing" in row["anomalies"]
    assert report["kpis"]["real_added_minutes"] == 20
    assert report["kpis"]["flat_added_minutes"] == 0
    assert report["kpis"]["compensated_minutes"] == 0
    assert report["kpis"]["review_count_flat"] == 1
    assert report["kpis"]["review_count_real"] == 0


def test_annulation_absente_de_a_verifier():
    canceled = _segment(
        booking_id=9,
        status="CANCELED",
        arrived_at=None,
        completed_at=None,
        scheduled_time=datetime(2026, 9, 12, 9, 0),
    )
    report = _report([canceled])
    assert report["compensation_lines"] == []
    assert _entries(report) == []
    assert report["kpis"]["review_count_real"] == 0
    assert report["kpis"]["transport_count"] == 0


def test_annulation_sans_arrivee_ne_compte_pas():
    report = _report(
        [
            _segment(
                booking_id=9,
                status="CANCELED",
                arrived_at=None,
                completed_at=None,
                assignment_status="EN_ROUTE_PICKUP",
                scheduled_time=datetime(2026, 9, 12, 9, 0),
            )
        ]
    )
    assert report["kpis"]["transport_count"] == 0
    assert report["kpis"]["compensated_minutes"] == 0
    assert report["kpis"]["flat_transport_minutes"] == 0
    assert _entries(report) == []


def test_annulation_sur_place_avec_arrived_at_compte_un_transport():
    report = _report(
        [
            _segment(
                booking_id=11,
                status="CANCELED",
                arrived_at=datetime(2026, 9, 12, 8, 5, tzinfo=UTC),
                boarded_at=None,
                completed_at=None,
                scheduled_time=datetime(2026, 9, 12, 9, 0),
                pickup_location="Domicile",
                dropoff_location="Hopital",
            )
        ]
    )
    assert report["kpis"]["transport_count"] == 1
    assert report["kpis"]["flat_transport_minutes"] == 30
    assert report["kpis"]["compensated_minutes"] == 30
    assert report["kpis"]["real_transport_minutes"] == 0
    row = _entries(report)[0]
    assert row["booking_id"] == 11
    assert row["worked_minutes"] is None
    assert row["proposed_worked_minutes"] in (None, 0)
    assert "cancelled_after_arrival" in row["anomalies"]
    assert row["line_key"] == "bk:11"


def test_annulation_sur_place_avec_arrived_pickup_compte_un_transport():
    report = _report(
        [
            _segment(
                booking_id=12,
                status="CANCELED",
                arrived_at=None,
                completed_at=None,
                assignment_status="ARRIVED_PICKUP",
                scheduled_time=datetime(2026, 9, 12, 9, 0),
            )
        ]
    )
    line = _line(report, "bk:12")
    assert report["kpis"]["transport_count"] == 1
    assert line["compensated_minutes"] == 30
    assert line["rule_type"] == "transport_flat"
    assert report["kpis"]["real_transport_minutes"] == 0


def test_snapshot_legacy_conserve_la_ligne_historique():
    segments = [
        _segment(
            booking_id=1,
            route_group_id="old",
            route_sequence_number=1,
            pickup_location="A",
            dropoff_location="B",
        ),
        _segment(
            booking_id=2,
            route_group_id="old",
            route_sequence_number=2,
            pickup_location="B",
            dropoff_location="A",
            arrived_at=datetime(2026, 9, 15, 9, 0, tzinfo=UTC),
            completed_at=datetime(2026, 9, 15, 9, 20, tzinfo=UTC),
        ),
    ]
    live = _report(segments)
    assert live["kpis"]["compensated_minutes"] == 60
    assert sorted(line["line_key"] for line in live["compensation_lines"]) == [
        "bk:1",
        "bk:2",
    ]
    snapshot = [
        {
            "journey_key": "rg:old",
            "line_key": "legacy:44",
            "driver_id": 7,
            "accounting_date": "2026-09-15",
            "compensated_minutes": 80,
            "compensation_status": "calculated",
            "rule_type": "round_trip",
            "classification_source": "journey",
            "attached_to_booking_id": 1,
            "policy_id": 1,
            "journey_status": "complete",
            "base_minutes": 60,
            "intermediate_stop_count": 0,
            "intermediate_stop_minutes": 0,
            "flat_minutes": None,
        }
    ]
    apply_ledger_snapshot(live, snapshot)
    assert live["compensation_source"] == "ledger"
    assert live["compensation_lines"][0]["line_key"] == "legacy:44"
    assert live["kpis"]["compensated_minutes"] == 80
    assert live["kpis"]["transport_count"] == 1
    assert sum(int(row["compensated_minutes"] or 0) for row in _entries(live)) == 80


def _review_items(report, mode):
    return [item for item in report.get("review_items") or [] if mode in item["modes"]]


def test_file_de_controle_a_la_meme_taille_que_le_compteur():
    """Le clic sur le compteur doit retrouver exactement les éléments comptés."""
    missing = _report(
        [
            _segment(
                booking_id=50,
                arrived_at=None,
                boarded_at=None,
                completed_at=None,
                scheduled_time=datetime(2026, 9, 12, 9, 30),
            )
        ]
    )
    assert (
        len(_review_items(missing, "real")) == missing["kpis"]["review_count_real"] == 1
    )
    assert _entries(missing)[0]["worked_minutes"] is None
    assert "real" in _entries(missing)[0]["review_modes"]
    assert "completion_time_missing" in _review_items(missing, "real")[0]["reasons"]

    closed = _report(
        [
            _segment(
                booking_id=51,
                arrived_at=datetime(2026, 9, 28, 8, 0, tzinfo=UTC),
                completed_at=datetime(2026, 9, 28, 8, 14, tzinfo=UTC),
            )
        ]
    )
    line = dict(closed["compensation_lines"][0])
    line["compensation_status"] = "requires_review"
    line["compensated_minutes"] = 0
    line["flat_minutes"] = None
    line["driver_id"] = 7
    apply_ledger_snapshot(closed, [line])
    assert len(_review_items(closed, "real")) == closed["kpis"]["review_count_real"]
    assert len(_review_items(closed, "flat")) == closed["kpis"]["review_count_flat"]
    marked = [
        row for row in _entries(closed) if "real" in (row.get("review_modes") or [])
    ]
    assert len(marked) == closed["kpis"]["review_count_real"]


def _manual(entry_id, driver_id, work_type="cleaning", duration_minutes=30):
    return {
        "entry_id": entry_id,
        "driver_id": driver_id,
        "work_type": work_type,
        "duration_minutes": duration_minutes,
        "work_date": "2026-09-30",
        "description": "Temps ajouté",
        "started_at": datetime(2026, 9, 30, 16, 0, tzinfo=UTC),
        "ended_at": datetime(2026, 9, 30, 16, 30, tzinfo=UTC),
        "cancelled": False,
    }


def _close(report):
    frozen = deepcopy(report)
    apply_ledger_snapshot(frozen, list(report["compensation_lines"]))
    return frozen


def test_cloture_ne_change_pas_six_forfaits_et_deux_ajouts():
    """6 × 30 min + 2 ajouts de 30 min restent 180 + 60 après clôture."""
    policies = [_policy(work_type_rules={"cleaning": {"mode": "flat", "minutes": 30}})]
    segments = [
        _segment(booking_id=1, driver_id=1),
        _segment(booking_id=2, driver_id=2),
        _segment(
            booking_id=3,
            driver_id=2,
            is_return=True,
            parent_booking_id=2,
            pickup_location="Rue B",
            dropoff_location="Rue A",
            arrived_at=datetime(2026, 9, 16, 8, 0, tzinfo=UTC),
            completed_at=datetime(2026, 9, 16, 8, 20, tzinfo=UTC),
        ),
        _segment(booking_id=4, driver_id=3),
        _segment(booking_id=5, driver_id=4),
        _segment(booking_id=6, driver_id=5),
    ]
    live = _report(
        segments,
        policies=policies,
        manuals=[_manual(6, 1), _manual(7, 2, duration_minutes=60)],
    )
    assert live["kpis"]["transport_count"] == 6
    assert live["kpis"]["flat_transport_minutes"] == 180
    assert live["kpis"]["flat_added_minutes"] == 60
    assert live["kpis"]["real_added_minutes"] == 90
    assert live["kpis"]["review_count_flat"] == 0
    frozen = _close(live)
    assert closure_parity_errors(live, frozen) == []
    assert frozen["kpis"]["transport_count"] == 6
    assert frozen["kpis"]["flat_transport_minutes"] == 180
    assert frozen["kpis"]["flat_added_minutes"] == 60
    assert frozen["kpis"]["real_added_minutes"] == 90
    assert frozen["kpis"]["review_count_flat"] == 0
    reread = _close(live)
    assert closure_parity_errors(frozen, reread) == []
    again = deepcopy(frozen)
    apply_ledger_snapshot(again, list(frozen["compensation_lines"]))
    assert closure_parity_errors(frozen, again) == []


def test_meme_journey_key_deux_lignes_de_trente_minutes():
    segments = [
        _segment(
            booking_id=11, driver_id=7, route_group_id="split", route_sequence_number=1
        ),
        _segment(
            booking_id=12,
            driver_id=8,
            route_group_id="split",
            route_sequence_number=2,
            is_return=True,
            pickup_location="Rue B",
            dropoff_location="Rue A",
        ),
    ]
    live = _report(segments)
    keys = sorted(line["line_key"] for line in live["compensation_lines"])
    assert keys == ["bk:11", "bk:12"]
    assert {line["flat_minutes"] for line in live["compensation_lines"]} == {30}
    assert {line["compensation_status"] for line in live["compensation_lines"]} == {
        "calculated"
    }
    assert all(
        "journey_split_across_drivers" not in (row.get("anomalies") or [])
        for row in _entries(live)
    )
    assert live["kpis"]["flat_transport_minutes"] == 60
    assert live["kpis"]["review_count_flat"] == 0
    frozen = _close(live)
    assert closure_parity_errors(live, frozen) == []
    assert sorted(line["line_key"] for line in frozen["compensation_lines"]) == keys
    assert frozen["kpis"]["flat_transport_minutes"] == 60
    assert frozen["kpis"]["review_count_flat"] == 0


def test_terminee_sans_completed_at_garde_le_forfait():
    live = _report(
        [
            _segment(
                booking_id=70,
                arrived_at=None,
                boarded_at=None,
                completed_at=None,
                scheduled_time=datetime(2026, 9, 12, 9, 30),
            )
        ]
    )
    line = live["compensation_lines"][0]
    assert line["line_key"] == "bk:70"
    assert line["compensation_status"] == "calculated"
    assert line["flat_minutes"] == 30
    assert line["compensated_minutes"] == 30
    assert live["kpis"]["flat_transport_minutes"] == 30
    assert live["kpis"]["review_count_flat"] == 0
    frozen = _close(live)
    assert closure_parity_errors(live, frozen) == []
    assert frozen["kpis"]["flat_transport_minutes"] == 30
    assert frozen["kpis"]["review_count_flat"] == 0
    assert frozen["compensation_lines"][0]["flat_minutes"] == 30


def test_cloture_uniquement_un_mois_civil_termine():
    septembre = (date(2026, 9, 1), date(2026, 9, 30))
    octobre = (date(2026, 10, 1), date(2026, 10, 31))
    novembre = (date(2026, 11, 1), date(2026, 11, 30))
    assert monthly_closure_refusal(
        date(2026, 9, 30), date(2026, 9, 30), date(2026, 10, 1)
    )
    assert monthly_closure_refusal(
        date(2026, 9, 28), date(2026, 10, 4), date(2026, 10, 5)
    )
    assert monthly_closure_refusal(
        date(2026, 9, 15), date(2026, 9, 30), date(2026, 10, 1)
    )
    assert monthly_closure_refusal(
        date(2026, 9, 1), date(2026, 10, 15), date(2026, 11, 1)
    )
    assert monthly_closure_refusal(*octobre, date(2026, 10, 30))
    assert monthly_closure_refusal(*octobre, date(2026, 10, 31))
    assert monthly_closure_refusal(*septembre, date(2026, 9, 30))
    assert monthly_closure_refusal(*octobre, date(2026, 11, 1)) is None
    assert monthly_closure_refusal(*septembre, date(2026, 10, 1)) is None
    assert monthly_closure_refusal(*novembre, date(2026, 12, 1)) is None


def test_snapshot_ancien_en_revue_garde_le_forfait_deja_calcule():
    """Une clôture déjà écrite en revue, mais avec flat_minutes, ne doit pas changer l'écran."""
    live = _report(
        [_segment(booking_id=81, driver_id=7), _segment(booking_id=82, driver_id=8)]
    )
    rows = []
    for line in live["compensation_lines"]:
        copied = dict(line)
        copied["compensation_status"] = "requires_review"
        copied["compensated_minutes"] = 0
        copied["rule_type"] = None
        rows.append(copied)
    frozen = deepcopy(live)
    apply_ledger_snapshot(frozen, rows)
    assert frozen["kpis"]["transport_count"] == live["kpis"]["transport_count"] == 2
    assert frozen["kpis"]["flat_transport_minutes"] == 60
    assert frozen["kpis"]["review_count_flat"] == 0
