"""Use-cases Clients (Companies)."""

from .aggregate_client_reservations_and_invoices import (
    AggregateClientReservationsAndInvoicesUseCase,
)
from .create_company_client import (
    CreateCompanyClientInput,
    CreateCompanyClientOutput,
    CreateCompanyClientUseCase,
)
from .delete_company_client import DeleteCompanyClientUseCase
from .list_company_clients import (
    ListCompanyClientsInput,
    ListCompanyClientsOutput,
    ListCompanyClientsUseCase,
)
from .update_company_client import UpdateCompanyClientUseCase

__all__ = [
    "AggregateClientReservationsAndInvoicesUseCase",
    "CreateCompanyClientInput",
    "CreateCompanyClientOutput",
    "CreateCompanyClientUseCase",
    "DeleteCompanyClientUseCase",
    "ListCompanyClientsInput",
    "ListCompanyClientsOutput",
    "ListCompanyClientsUseCase",
    "UpdateCompanyClientUseCase",
]
