"""✅ Phase 3: Décorateur IP whitelist pour protéger les endpoints admin.

Permet de restreindre l'accès aux endpoints critiques à une liste d'IPs autorisées.
"""

import ipaddress
import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from flask import abort, request
from flask_jwt_extended import get_jwt_identity

from security.ip_whitelist_alerts import send_ip_whitelist_alert

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _parse_ip_whitelist(whitelist_str: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse une chaîne de whitelist IP (séparée par des virgules).

    Args:
        whitelist_str: Chaîne contenant les IPs/réseaux (ex: "192.168.1.0/24,10.0.0.1")

    Returns:
        Liste de réseaux IP
    """
    networks = []
    for raw_item in whitelist_str.split(","):
        item = raw_item.strip()
        if not item:
            continue

        try:
            # Essayer comme réseau CIDR
            network = ipaddress.ip_network(item, strict=False)
            networks.append(network)
        except ValueError:
            try:
                # Essayer comme IP simple (convertir en /32 ou /128)
                ip = ipaddress.ip_address(item)
                if isinstance(ip, ipaddress.IPv4Address):
                    networks.append(ipaddress.IPv4Network(f"{item}/32", strict=False))
                else:
                    networks.append(ipaddress.IPv6Network(f"{item}/128", strict=False))
            except ValueError:
                logger.warning("[IP Whitelist] IP/réseau invalide ignoré: %s", item)

    return networks


def _get_client_ip() -> str | None:
    """Récupère l'IP réelle du client (en tenant compte des proxies).

    Returns:
        IP du client ou None
    """
    # Vérifier les headers de proxy courants
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For peut contenir plusieurs IPs (première = client original)
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fallback vers remote_addr
    return request.environ.get("REMOTE_ADDR")


def _is_ip_allowed(client_ip: str, allowed_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network]) -> bool:
    """Vérifie si l'IP du client est dans un des réseaux autorisés.

    Args:
        client_ip: IP du client
        allowed_networks: Liste de réseaux autorisés

    Returns:
        True si autorisé, False sinon
    """
    try:
        ip = ipaddress.ip_address(client_ip)
        return any(ip in network for network in allowed_networks)
    except ValueError:
        logger.warning("[IP Whitelist] IP invalide: %s", client_ip)
        return False


def ip_whitelist_required(
    allowed_ips: list[str] | None = None,
    env_key: str = "ADMIN_IP_WHITELIST",
    allow_localhost: bool = True,
) -> Callable[[F], F]:
    """Décorateur pour restreindre l'accès à une liste d'IPs.

    Args:
        allowed_ips: Liste d'IPs/réseaux autorisés (ex: ["192.168.1.0/24", "10.0.0.1"])
        env_key: Clé de variable d'environnement pour la whitelist (par défaut: ADMIN_IP_WHITELIST)
        allow_localhost: Si True, autorise localhost/127.0.0.1 en développement

    Returns:
        Décorateur Flask

    Exemple:
        @admin_ns.route("/critical-action")
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required(["192.168.1.0/24"])
        def post(self):
            ...
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Récupérer la whitelist depuis les arguments ou l'environnement
            whitelist_str = ",".join(allowed_ips) if allowed_ips else os.getenv(env_key)

            # Si pas de whitelist configurée, autoriser l'accès (fail-open pour développement)
            if not whitelist_str:
                if os.getenv("FLASK_ENV") == "production":
                    logger.warning("[IP Whitelist] ⚠️ Pas de whitelist configurée en production pour %s", request.path)
                else:
                    logger.debug("[IP Whitelist] Pas de whitelist configurée, accès autorisé (dev)")
                return fn(*args, **kwargs)

            # Parser la whitelist
            allowed_networks = _parse_ip_whitelist(whitelist_str)

            # Ajouter localhost si demandé
            if allow_localhost:
                allowed_networks.extend(
                    [
                        ipaddress.IPv4Network("127.0.0.0/8", strict=False),
                        ipaddress.IPv6Network("::1/128", strict=False),
                    ]
                )

            # Récupérer l'IP du client
            client_ip = _get_client_ip()
            if not client_ip:
                logger.warning("[IP Whitelist] Impossible de déterminer l'IP du client")
                abort(403, description="Accès non autorisé (IP non déterminable)")

            # Vérifier si l'IP est autorisée
            if not _is_ip_allowed(client_ip, allowed_networks):
                logger.warning(
                    "[IP Whitelist] ⛔ Accès refusé depuis IP: %s pour %s %s",
                    client_ip,
                    request.method,
                    request.path,
                )

                # ✅ Envoyer une alerte pour cette tentative d'accès non autorisée
                try:
                    # Récupérer l'ID utilisateur si authentifié
                    user_id = None
                    try:
                        user_identity = get_jwt_identity()
                        if user_identity:
                            # user_identity peut être un string (public_id) ou un int
                            # On essaie de le convertir en int si possible
                            if isinstance(user_identity, str) and user_identity.isdigit():
                                user_id = int(user_identity)
                            elif isinstance(user_identity, int):
                                user_id = user_identity
                    except Exception:
                        # Si pas de token JWT ou erreur, user_id reste None
                        pass

                    send_ip_whitelist_alert(
                        client_ip=client_ip,
                        endpoint=request.path,
                        method=request.method,
                        user_agent=request.headers.get("User-Agent"),
                        user_id=user_id,
                    )
                except Exception as e:
                    # Ne pas faire échouer la requête si l'alerte échoue
                    logger.exception("[IP Whitelist] Erreur lors de l'envoi d'alerte: %s", e)

                abort(403, description="Accès non autorisé (IP non autorisée)")

            logger.debug("[IP Whitelist] ✅ Accès autorisé depuis IP: %s", client_ip)
            return fn(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
