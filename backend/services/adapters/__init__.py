"""Adapters pour transformation de payloads entre canaux (mobile, web, etc.)."""

from .mobile_booking_adapter import map_mobile_ride_payload_to_manual_booking_payload

__all__ = ["map_mobile_ride_payload_to_manual_booking_payload"]
