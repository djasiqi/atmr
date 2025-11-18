# models/payment.py
"""Model Payment - Gestion des paiements.
Extrait depuis models.py (lignes ~1644-1762).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, Column, DateTime, Float, ForeignKey, Index, Integer, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from ext import db

from .base import _as_float, _iso
from .enums import PaymentMethod, PaymentStatus


class Payment(db.Model):
    __tablename__ = "payment"
    __table_args__ = (
        CheckConstraint("amount > 0", name="chk_payment_amount_positive"),
        Index("ix_payment_booking_status", "booking_id", "status"),
        Index("ix_payment_client_date", "client_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)

    date = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    method = Column(SAEnum("credit_card", "paypal", "bank_transfer", "cash", name="payment_method"), nullable=False)
    status = Column(
        SAEnum(PaymentStatus, name="payment_status"),
        nullable=False,
        default=PaymentStatus.PENDING,
        server_default=PaymentStatus.PENDING.value,
    )

    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("client.id", ondelete="CASCADE"), nullable=False, index=True)
    booking_id = Column(Integer, ForeignKey("booking.id", ondelete="CASCADE"), nullable=False, index=True)

    # Relations
    client = relationship("Client", back_populates="payments", passive_deletes=True)
    booking = relationship("Booking", back_populates="payments", passive_deletes=True)

    @property
    def serialize(self):
        amt = _as_float(self.amount)
        bk = self.booking
        return {
            "id": self.id,
            "amount": amt,
            "method": str(self.method),
            "status": (self.status.value if isinstance(self.status, PaymentStatus) else str(self.status)),
            "date": _iso(self.date),
            "updated_at": _iso(self.updated_at),
            "client_id": self.client_id,
            "booking_id": self.booking_id,
            "booking_info": {
                "pickup_location": bk.pickup_location if bk else None,
                "dropoff_location": bk.dropoff_location if bk else None,
                "scheduled_time": _iso(bk.scheduled_time) if bk else None,
            },
        }

    # Validations
    @validates("user_id")
    def validate_user_id(self, _key, user_id):
        if not isinstance(user_id, int) or user_id <= 0:
            msg = "L'ID utilisateur pour Payment doit être un entier positif."
            raise ValueError(msg)
        return user_id

    @validates("amount")
    def validate_amount(self, _key, amount):
        if amount is None:
            return None
        if amount <= 0:
            msg = "Le montant doit être supérieur à 0"
            raise ValueError(msg)
        return round(amount, 2)

    @validates("method")
    def validate_method(self, _key, method):
        # ✅ Support à la fois anciennes valeurs et PaymentMethod enum
        allowed = {"credit_card", "paypal", "bank_transfer", "cash"}
        if isinstance(method, str):
            method = method.strip()
            # Mapper anciennes valeurs vers enum si possible
            if method == "credit_card":
                return "card"  # Mapper vers la nouvelle valeur
            if method in allowed:
                return method
            # Essayer de valider avec PaymentMethod enum
            try:
                PaymentMethod(method)
                return method
            except ValueError:
                msg = f"Méthode de paiement invalide. Autorisées : {', '.join(sorted(allowed))} ou {[m.value for m in PaymentMethod]}"
                raise ValueError(msg) from None
        if isinstance(method, PaymentMethod):
            return method.value
        if method not in allowed:
            msg = f"Méthode de paiement invalide. Autorisées : {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return method

    @validates("date")
    def validate_date(self, _key, value):
        if value is None:
            msg = "La date de paiement ne peut pas être nulle"
            raise ValueError(msg)
        if value > datetime.now(UTC):
            msg = "La date de paiement ne peut pas être dans le futur"
            raise ValueError(msg)
        return value

    @validates("status")
    def validate_status(self, _key, status):
        if isinstance(status, str):
            key = status.upper().strip()
            if key not in PaymentStatus.__members__:
                msg = f"Statut de paiement invalide : {status}. Attendu: {list(PaymentStatus.__members__.keys())}"
                raise ValueError(msg)
            status = PaymentStatus[key]
        if not isinstance(status, PaymentStatus):
            msg = "Statut invalide (PaymentStatus attendu)."
            raise ValueError(msg)
        return status

    # Métier
    def update_status(self, new_status):
        if isinstance(new_status, str):
            try:
                new_status = PaymentStatus(new_status.lower())
            except Exception:
                msg = "Statut de paiement invalide."
                raise ValueError(msg) from None
        if not isinstance(new_status, PaymentStatus):
            msg = "Statut de paiement invalide."
            raise ValueError(msg)
        self.status = new_status
