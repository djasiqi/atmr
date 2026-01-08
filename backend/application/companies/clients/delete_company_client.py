from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeleteCompanyClientResult:
    ok: bool
    action: str | None = None  # "hard" | "soft"
    payload: dict[str, object] | None = None
    status_code: int | None = None


class DeleteCompanyClientUseCase:
    """Use-case Application: décider soft vs hard delete d'un client."""

    def execute(
        self,
        *,
        hard_delete: bool,
        invoice_count: int,
        booking_count: int,
    ) -> DeleteCompanyClientResult:
        if hard_delete:
            if invoice_count > 0 or booking_count > 0:
                return DeleteCompanyClientResult(
                    ok=False,
                    status_code=400,
                    payload={
                        "error": "Impossible de supprimer définitivement ce client",
                        "reason": (
                            f"Le client a {invoice_count} facture(s) "
                            f"et {booking_count} réservation(s)"
                        ),
                        "suggestion": (
                            "Utilisez la désactivation (soft delete) à la place"
                        ),
                    },
                )
            return DeleteCompanyClientResult(ok=True, action="hard")

        return DeleteCompanyClientResult(ok=True, action="soft")
