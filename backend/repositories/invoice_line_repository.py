"""Repository pour l'accès aux données InvoiceLine."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Protocol, cast

from domain.invoice_dto import InvoiceLineDTO
from models import InvoiceLine
from models.enums import InvoiceLineType

logger = logging.getLogger(__name__)


class InvoiceLineRepositoryPort(Protocol):
    """Port (interface) pour le repository InvoiceLine."""

    def create(self, line_data: dict[str, Any]) -> InvoiceLineDTO:
        """Crée une nouvelle ligne de facture.

        Args:
            line_data: Dictionnaire avec les données de la ligne

        Returns:
            InvoiceLineDTO créé
        """
        ...

    def create_batch(self, lines_data: list[dict[str, Any]]) -> list[InvoiceLineDTO]:
        """Crée plusieurs lignes de facture en une seule transaction.

        Args:
            lines_data: Liste de dictionnaires avec les données des lignes

        Returns:
            Liste de InvoiceLineDTO créées
        """
        ...


class InvoiceLineRepository:
    """Repository SQLAlchemy pour InvoiceLine."""

    def _to_dto(self, line: InvoiceLine) -> InvoiceLineDTO:
        """Convertit un modèle SQLAlchemy InvoiceLine en DTO."""
        return InvoiceLineDTO(
            id=line.id,
            invoice_id=line.invoice_id,
            line_type=line.type.value
            if hasattr(line.type, "value")
            else str(line.type),
            description=line.description,
            quantity=line.qty,
            unit_price=line.unit_price,
            line_total=line.line_total,
            vat_rate=line.vat_rate,
            vat_amount=line.vat_amount,
            total_with_vat=line.total_with_vat,
            adjustment_note=line.adjustment_note,
            reservation_id=cast(int | None, line.reservation_id),
            line_meta=cast(dict[str, Any] | None, line.line_meta),
        )

    def create(self, line_data: dict[str, Any]) -> InvoiceLineDTO:
        """Crée une nouvelle ligne de facture.

        Args:
            line_data: Dictionnaire avec les données de la ligne
                (invoice_id, type, description, qty, unit_price, etc.)

        Returns:
            InvoiceLineDTO créé

        Side-effects:
            - DB: Crée InvoiceLine et flush (pas de commit, laisse le use case gérer)
        """
        from ext import db

        line = InvoiceLine()
        line.invoice_id = line_data["invoice_id"]
        line.type = line_data.get("type", InvoiceLineType.RIDE)
        line.description = line_data["description"]
        line.qty = line_data.get("qty", Decimal("1"))
        line.unit_price = line_data["unit_price"]
        line.line_total = line_data["line_total"]
        line.vat_rate = line_data.get("vat_rate")
        line.vat_amount = line_data.get("vat_amount", Decimal("0.00"))
        line.total_with_vat = line_data.get("total_with_vat", line.line_total)
        line.adjustment_note = line_data.get("adjustment_note")
        line.reservation_id = line_data.get("reservation_id")
        _lm = line_data.get("line_meta")
        if _lm is None:
            _lm = line_data.get("meta")
        line.line_meta = _lm

        db.session.add(line)
        db.session.flush()  # Pour obtenir l'ID sans commit

        return self._to_dto(line)

    def create_batch(self, lines_data: list[dict[str, Any]]) -> list[InvoiceLineDTO]:
        """Crée plusieurs lignes de facture en une seule transaction.

        Args:
            lines_data: Liste de dictionnaires avec les données des lignes

        Returns:
            Liste de InvoiceLineDTO créées

        Side-effects:
            - DB: Crée InvoiceLine et flush (pas de commit, laisse le use case gérer)
        """
        from ext import db

        lines = []
        for line_data in lines_data:
            line = InvoiceLine()
            line.invoice_id = line_data["invoice_id"]
            line.type = line_data.get("type", InvoiceLineType.RIDE)
            line.description = line_data["description"]
            line.qty = line_data.get("qty", Decimal("1"))
            line.unit_price = line_data["unit_price"]
            line.line_total = line_data["line_total"]
            line.vat_rate = line_data.get("vat_rate")
            line.vat_amount = line_data.get("vat_amount", Decimal("0.00"))
            line.total_with_vat = line_data.get("total_with_vat", line.line_total)
            line.adjustment_note = line_data.get("adjustment_note")
            line.reservation_id = line_data.get("reservation_id")
            _lm_b = line_data.get("line_meta")
            if _lm_b is None:
                _lm_b = line_data.get("meta")
            line.line_meta = _lm_b
            db.session.add(line)
            lines.append(line)

        db.session.flush()  # Pour obtenir les IDs sans commit

        return [self._to_dto(line) for line in lines]
