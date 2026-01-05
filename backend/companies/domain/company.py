"""Agrégat racine : Company."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from companies.domain.company_id import CompanyId
from companies.domain.value_objects import (
    BillingSettings,
    CompanySettings,
    DispatchMode,
    PlanningSettings,
)


@dataclass
class Company:
    """Agrégat racine : Entreprise.

    Responsabilités :
    - Gérer le profil d'une entreprise
    - Gérer la configuration (dispatch, facturation, etc.)
    - Appliquer les invariants métier
    """

    id: CompanyId
    name: str
    user_id: int
    settings: CompanySettings
    is_approved: bool = False
    is_partner: bool = False
    uid_ide: str | None = None
    logo_url: str | None = None
    created_at: datetime | None = None
    accepted_at: datetime | None = None

    def enable_dispatch(self) -> None:
        """Active le dispatch pour l'entreprise (méthode métier)."""
        self.settings = CompanySettings(
            dispatch_enabled=True,
            dispatch_mode=self.settings.dispatch_mode,
            planning_settings=self.settings.planning_settings,
            billing_settings=self.settings.billing_settings,
        )

    def disable_dispatch(self) -> None:
        """Désactive le dispatch pour l'entreprise (méthode métier)."""
        self.settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=self.settings.dispatch_mode,
            planning_settings=self.settings.planning_settings,
            billing_settings=self.settings.billing_settings,
        )

    def set_dispatch_mode(self, mode: DispatchMode) -> None:
        """Change le mode de dispatch (méthode métier)."""
        if not self.settings.dispatch_enabled:
            raise ValueError("Cannot set dispatch mode: dispatch is disabled")
        self.settings = CompanySettings(
            dispatch_enabled=self.settings.dispatch_enabled,
            dispatch_mode=mode,
            planning_settings=self.settings.planning_settings,
            billing_settings=self.settings.billing_settings,
        )

    def approve(self) -> None:
        """Approuve l'entreprise (méthode métier)."""
        self.is_approved = True
        self.accepted_at = datetime.now()

    def update_planning_settings(self, settings: PlanningSettings) -> None:
        """Met à jour les paramètres de planification (méthode métier)."""
        self.settings = CompanySettings(
            dispatch_enabled=self.settings.dispatch_enabled,
            dispatch_mode=self.settings.dispatch_mode,
            planning_settings=settings,
            billing_settings=self.settings.billing_settings,
        )

    def update_billing_settings(self, settings: BillingSettings) -> None:
        """Met à jour les paramètres de facturation (méthode métier)."""
        self.settings = CompanySettings(
            dispatch_enabled=self.settings.dispatch_enabled,
            dispatch_mode=self.settings.dispatch_mode,
            planning_settings=self.settings.planning_settings,
            billing_settings=settings,
        )

    def validate(self) -> bool:
        """Valide les invariants métier."""
        # Invariant 1: Une entreprise doit avoir un nom
        if not self.name or len(self.name.strip()) == 0:
            return False

        # Invariant 2: user_id doit être positif
        if self.user_id <= 0:
            return False

        # Invariant 3: Si dispatch est activé, le mode doit être valide
        return not (
            self.settings.dispatch_enabled
            and self.settings.dispatch_mode.value
            not in ("manual", "semi_auto", "fully_auto")
        )
