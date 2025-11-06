"""Gestion du traffic control (TC) pour injecter latence/packet loss.

Utilise les outils Linux TC pour injecter du chaos réseau.
Nécessite les privilèges root/sudo.
"""
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class TrafficControlManager:
    """Gestionnaire de Traffic Control pour chaos réseau."""
    
    def __init__(self, interface: str = "eth0") -> None:  # type: ignore[no-untyped-def]
        self.interface = interface
        self.active = False
    
    def add_latency(self, ms: int, jitter_ms: int = 0) -> bool:
        """Ajoute de la latence réseau via TC.
        
        Args:
            ms: Latence en millisecondes
            jitter_ms: Jitter (variation) en millisecondes
        
        Returns:
            True si succès
        """
        if os.geteuid() != 0:
            logger.error("[TC] Requires root privileges")
            return False
        
        try:
            # Ajouter une qdisc netem pour la latence
            cmd = [
                "tc", "qdisc", "add", "dev", self.interface,
                "root", "netem", "delay", f"{ms}ms"
            ]
            if jitter_ms > 0:
                cmd.extend([f"{jitter_ms}ms"])
            
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                self.active = True
                logger.info("[TC] Added %sms latency on %s", ms, self.interface)
                return True
            logger.error("[TC] Failed to add latency: %s", result.stderr)
            return False
        except Exception as e:
            logger.error("[TC] Error adding latency: %s", e)
            return False
    
    def add_packet_loss(self, percent: float) -> bool:
        """Ajoute une perte de paquets via TC.
        
        Args:
            percent: Pourcentage de perte (0-100)
        
        Returns:
            True si succès
        """
        if os.geteuid() != 0:
            logger.error("[TC] Requires root privileges")
            return False
        
        try:
            cmd = [
                "tc", "qdisc", "replace", "dev", self.interface,
                "root", "netem", "loss", f"{percent}%"
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("[TC] Added %.1f%% packet loss on %s", percent, self.interface)
                return True
            logger.error("[TC] Failed to add packet loss: %s", result.stderr)
            return False
        except Exception as e:
            logger.error("[TC] Error adding packet loss: %s", e)
            return False
    
    def clear(self) -> bool:
        """Supprime toutes les règles TC."""
        if os.geteuid() != 0:
            logger.error("[TC] Requires root privileges")
            return False
        
        try:
            cmd = ["tc", "qdisc", "del", "dev", self.interface, "root"]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0 or "Cannot delete" not in result.stderr:
                self.active = False
                logger.info("[TC] Cleared rules on %s", self.interface)
                return True
            # Ignorer si aucune règle n'existe
            self.active = False
            return True
        except Exception as e:
            logger.error("[TC] Error clearing rules: %s", e)
            return False
    
    def is_active(self) -> bool:
        """Vérifie si des règles TC sont actives."""
        try:
            cmd = ["tc", "qdisc", "show", "dev", self.interface]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            return "netem" in result.stdout
        except Exception:
            return False

