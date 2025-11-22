# backend/services/unified_dispatch/exceptions.py
"""Exceptions personnalisées pour le module unified_dispatch."""


class DispatchError(Exception):
    """Exception de base pour les erreurs de dispatch."""

    def __init__(self, message: str, company_id: int | None = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.company_id = company_id
        self.extra = kwargs

    def __str__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        if self.company_id:
            return f"{self.message} (company_id={self.company_id})"
        return self.message


class CompanyNotFoundError(DispatchError):
    """Exception levée quand une Company est introuvable en DB."""

    def __init__(self, company_id: int, **kwargs):
        message = f"Company {company_id} introuvable en DB. Vérifier que la Company existe et est commitée avant d'appeler engine.run()"
        super().__init__(message, company_id=company_id, **kwargs)
        self.company_id = company_id


class DispatchRunNotFoundError(DispatchError):
    """Exception levée quand un DispatchRun est introuvable en DB."""

    def __init__(self, dispatch_run_id: int, **kwargs):
        message = f"DispatchRun {dispatch_run_id} introuvable en DB"
        super().__init__(message, company_id=None, **kwargs)
        self.dispatch_run_id = dispatch_run_id


class InvalidDispatchModeError(DispatchError):
    """Exception levée quand le mode de dispatch est invalide."""

    def __init__(self, mode: str, valid_modes: list[str] | None = None, **kwargs):
        valid_str = f" (modes valides: {', '.join(valid_modes)})" if valid_modes else ""
        message = f"Mode de dispatch invalide: '{mode}'{valid_str}"
        super().__init__(message, company_id=None, **kwargs)
        self.mode = mode
        self.valid_modes = valid_modes
