"""Gestion du traffic control (TC) pour injecter latence/packet loss.

Utilise les outils Linux TC pour injecter du chaos réseau.
Nécessite les privilèges root/sudo.
"""

import logging
import os
import re
import subprocess
from subprocess import TimeoutExpired

logger = logging.getLogger(__name__)

# Constantes de sécurité pour validations
INTERFACE_NAME_PATTERN = re.compile(
    r"^[a-zA-Z0-9_-]{1,15}$"
)  # Noms d'interface réseau valides
SUBPROCESS_TIMEOUT = 10  # Timeout en secondes pour commandes TC (10s max)
MAX_LATENCY_MS = 10000  # Latence max : 10 secondes
MAX_JITTER_MS = 1000  # Jitter max : 1 seconde
MAX_PERCENT = 100.0  # Pourcentage max : 100.0


class TrafficControlManager:
    """Gestionnaire de Traffic Control pour chaos réseau."""

    def __init__(self, interface: str = "eth0") -> None:  # type: ignore[no-untyped-def]
        # ✅ Sécurité : Valider le nom d'interface réseau
        if not self._validate_interface(interface):
            raise ValueError(
                f"Invalid interface name: {interface}. Must match pattern: {INTERFACE_NAME_PATTERN.pattern}"
            )
        self.interface = interface
        self.active = False

    @staticmethod
    def _validate_interface(interface: str) -> bool:
        """Valide le nom d'interface réseau.

        Args:
            interface: Nom de l'interface réseau

        Returns:
            True si valide, False sinon
        """
        # Défense en profondeur : vérifier le type même si annoté (runtime validation)
        if not isinstance(interface, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return False
        return bool(INTERFACE_NAME_PATTERN.match(interface))

    @staticmethod
    def _validate_latency_ms(ms: int) -> bool:
        """Valide la latence en millisecondes.

        Args:
            ms: Latence en millisecondes

        Returns:
            True si valide (0 < ms <= MAX_LATENCY_MS), False sinon
        """
        # Défense en profondeur : vérifier le type même si annoté (runtime validation)
        return isinstance(ms, int) and 0 < ms <= MAX_LATENCY_MS  # pyright: ignore[reportUnnecessaryIsInstance]

    @staticmethod
    def _validate_jitter_ms(jitter_ms: int) -> bool:
        """Valide le jitter en millisecondes.

        Args:
            jitter_ms: Jitter en millisecondes

        Returns:
            True si valide (0 <= jitter_ms <= MAX_JITTER_MS), False sinon
        """
        # Défense en profondeur : vérifier le type même si annoté (runtime validation)
        return isinstance(jitter_ms, int) and 0 <= jitter_ms <= MAX_JITTER_MS  # pyright: ignore[reportUnnecessaryIsInstance]

    @staticmethod
    def _validate_percent(percent: float) -> bool:
        """Valide le pourcentage de perte de paquets.

        Args:
            percent: Pourcentage (0-100)

        Returns:
            True si valide (0.0 <= percent <= MAX_PERCENT), False sinon
        """
        # Défense en profondeur : vérifier le type même si annoté (runtime validation)
        if not isinstance(percent, (int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
            return False
        return 0.0 <= float(percent) <= MAX_PERCENT

    def add_latency(self, ms: int, jitter_ms: int = 0) -> bool:  # noqa: PLR0911
        """Ajoute de la latence réseau via TC.

        Args:
            ms: Latence en millisecondes (0 < ms <= 10000)
            jitter_ms: Jitter (variation) en millisecondes (0 <= jitter_ms <= 1000)

        Returns:
            True si succès
        """
        # ✅ Sécurité : Valider les privilèges root
        if os.geteuid() != 0:
            logger.error("[TC] Requires root privileges")
            return False

        # ✅ Sécurité : Valider les paramètres d'entrée
        if not self._validate_latency_ms(ms):
            logger.error(
                "[TC] Invalid latency: %s (must be 0 < ms <= %d)", ms, MAX_LATENCY_MS
            )
            return False
        if not self._validate_jitter_ms(jitter_ms):
            logger.error(
                "[TC] Invalid jitter: %s (must be 0 <= jitter_ms <= %d)",
                jitter_ms,
                MAX_JITTER_MS,
            )
            return False

        try:
            # ✅ Sécurité : Utiliser liste d'arguments (pas shell=True) pour éviter injection shell
            # Ajouter une qdisc netem pour la latence
            cmd = [
                "tc",
                "qdisc",
                "add",
                "dev",
                self.interface,
                "root",
                "netem",
                "delay",
                f"{ms}ms",
            ]
            if jitter_ms > 0:
                cmd.extend([f"{jitter_ms}ms"])

            # ✅ Sécurité : Ajouter timeout pour éviter blocage indéfini
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                # shell=False par défaut (sécurisé avec listes)
            )
            if result.returncode == 0:
                self.active = True
                logger.info("[TC] Added %sms latency on %s", ms, self.interface)
                return True
            logger.error("[TC] Failed to add latency: %s", result.stderr)
            return False
        except TimeoutExpired as e:
            # ✅ Sécurité : Gérer timeout avec log approprié
            logger.error(
                "[TC] Command timeout after %ds: %s", SUBPROCESS_TIMEOUT, e.cmd
            )
            return False
        except Exception as e:
            logger.error("[TC] Error adding latency: %s", e)
            return False

    def add_packet_loss(self, percent: float) -> bool:
        """Ajoute une perte de paquets via TC.

        Args:
            percent: Pourcentage de perte (0.0-100.0)

        Returns:
            True si succès
        """
        # ✅ Sécurité : Valider les privilèges root
        if os.geteuid() != 0:
            logger.error("[TC] Requires root privileges")
            return False

        # ✅ Sécurité : Valider le paramètre d'entrée
        if not self._validate_percent(percent):
            logger.error(
                "[TC] Invalid packet loss percent: %s (must be 0.0 <= percent <= 100.0)",
                percent,
            )
            return False

        try:
            # ✅ Sécurité : Utiliser liste d'arguments (pas shell=True) pour éviter injection shell
            cmd = [
                "tc",
                "qdisc",
                "replace",
                "dev",
                self.interface,
                "root",
                "netem",
                "loss",
                f"{percent}%",
            ]
            # ✅ Sécurité : Ajouter timeout pour éviter blocage indéfini
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                # shell=False par défaut (sécurisé avec listes)
            )
            if result.returncode == 0:
                logger.info(
                    "[TC] Added %.1f%% packet loss on %s", percent, self.interface
                )
                return True
            logger.error("[TC] Failed to add packet loss: %s", result.stderr)
            return False
        except TimeoutExpired as e:
            # ✅ Sécurité : Gérer timeout avec log approprié
            logger.error(
                "[TC] Command timeout after %ds: %s", SUBPROCESS_TIMEOUT, e.cmd
            )
            return False
        except Exception as e:
            logger.error("[TC] Error adding packet loss: %s", e)
            return False

    def clear(self) -> bool:
        """Supprime toutes les règles TC."""
        # ✅ Sécurité : Valider les privilèges root
        if os.geteuid() != 0:
            logger.error("[TC] Requires root privileges")
            return False

        try:
            # ✅ Sécurité : Utiliser liste d'arguments (pas shell=True) pour éviter injection shell
            cmd = ["tc", "qdisc", "del", "dev", self.interface, "root"]
            # ✅ Sécurité : Ajouter timeout pour éviter blocage indéfini
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                # shell=False par défaut (sécurisé avec listes)
            )
            if result.returncode == 0 or "Cannot delete" not in result.stderr:
                self.active = False
                logger.info("[TC] Cleared rules on %s", self.interface)
                return True
            # Ignorer si aucune règle n'existe
            self.active = False
            return True
        except TimeoutExpired as e:
            # ✅ Sécurité : Gérer timeout avec log approprié
            logger.error(
                "[TC] Command timeout after %ds: %s", SUBPROCESS_TIMEOUT, e.cmd
            )
            return False
        except Exception as e:
            logger.error("[TC] Error clearing rules: %s", e)
            return False

    def is_active(self) -> bool:
        """Vérifie si des règles TC sont actives."""
        try:
            # ✅ Sécurité : Utiliser liste d'arguments (pas shell=True) pour éviter injection shell
            cmd = ["tc", "qdisc", "show", "dev", self.interface]
            # ✅ Sécurité : Ajouter timeout pour éviter blocage indéfini
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                # shell=False par défaut (sécurisé avec listes)
            )
            return "netem" in result.stdout
        except TimeoutExpired as e:
            # ✅ Sécurité : Gérer timeout avec log approprié
            logger.error(
                "[TC] Command timeout after %ds: %s", SUBPROCESS_TIMEOUT, e.cmd
            )
            return False
        except Exception:
            return False
