"""Module de repositories pour abstraire l'accès à la base de données."""

from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.company_repository import CompanyRepository
from repositories.dispatch_run_repository import DispatchRunRepository
from repositories.driver_repository import DriverRepository
from repositories.invoice_repository import InvoiceRepository
from repositories.user_repository import UserRepository
from repositories.vehicle_repository import VehicleRepository

__all__ = [
    "AssignmentRepository",
    "BookingRepository",
    "ClientRepository",
    "CompanyRepository",
    "DispatchRunRepository",
    "DriverRepository",
    "InvoiceRepository",
    "UserRepository",
    "VehicleRepository",
]
