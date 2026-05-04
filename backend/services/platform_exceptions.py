"""Exceptions gouvernance plateforme (V1)."""


class PlatformTenantSuspended(Exception):
    """Entreprise (tenant) suspendue au sens plateforme."""

    def __init__(self, message: str | None = None) -> None:
        self.message = message or ("Ce transporteur est suspendu au sens plateforme.")
        super().__init__(self.message)


class PlatformRunbookConflict(Exception):
    """Exécution runbook refusée (conflit concurrent)."""

    def __init__(
        self, message: str | None = None, code: str = "runbook_conflict"
    ) -> None:
        self.code = code
        self.message = (
            message or "Une exécution runbook est déjà en cours pour ce tenant."
        )
        super().__init__(self.message)


class PlatformRollbackNotAllowed(Exception):
    """Rollback refusé (état non terminal)."""

    def __init__(self, message: str | None = None) -> None:
        self.code = "rollback_not_allowed"
        self.message = message or "Rollback non autorisé pour cet état d'exécution."
        super().__init__(self.message)


class PlatformTenantAlreadySuspended(Exception):
    """Suspension tenant déjà en vigueur (idempotence)."""

    def __init__(self, message: str | None = None) -> None:
        self.code = "tenant_already_suspended"
        self.message = message or "Le tenant est déjà suspendu."
        super().__init__(self.message)
