"""Encaissement conditionnel en fin de transport. Prototype non activé."""

from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from domain.end_of_trip_settlement.cash_procedure import (
    consult_cash,
    describe_cash_amounts,
    submit_cash_declaration,
)
from domain.end_of_trip_settlement.destination_event import (
    DESTINATION_ARRIVAL_EVENT_ENABLED,
    propose_destination_arrival,
)
from domain.end_of_trip_settlement.driver_read import (
    invalidate_displayed_amount,
    read_driver_settlement,
    read_for_assigned_driver,
    simulated_driver_contract,
)
from domain.end_of_trip_settlement.eligibility import (
    EligibilityDecision,
    SettlementSnapshot,
    driver_collection_view,
    evaluate,
)
from domain.end_of_trip_settlement.group_collection import plan_round_trip
from domain.end_of_trip_settlement.patient_invoice_procedure import (
    DOCUMENT_STATUS,
    prepare_patient_invoice,
)
from domain.end_of_trip_settlement.policy import (
    SettlementPolicy,
    claim_collection,
    propose_policy,
    record_cash_ledger,
    snapshot_from_policy,
)
from domain.end_of_trip_settlement.prototypes import (
    declare_cash,
    open_tap_attempt,
    record_exception,
    resolve_tap_simulation,
)

__all__ = [
    "DESTINATION_ARRIVAL_EVENT_ENABLED",
    "DOCUMENT_STATUS",
    "END_OF_TRIP_COLLECTION_ENABLED",
    "EligibilityDecision",
    "SettlementPolicy",
    "SettlementSnapshot",
    "claim_collection",
    "consult_cash",
    "declare_cash",
    "describe_cash_amounts",
    "driver_collection_view",
    "evaluate",
    "invalidate_displayed_amount",
    "open_tap_attempt",
    "plan_round_trip",
    "prepare_patient_invoice",
    "propose_destination_arrival",
    "propose_policy",
    "read_driver_settlement",
    "read_for_assigned_driver",
    "record_cash_ledger",
    "record_exception",
    "resolve_tap_simulation",
    "simulated_driver_contract",
    "snapshot_from_policy",
    "submit_cash_declaration",
]
