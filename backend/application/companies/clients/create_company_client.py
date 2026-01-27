from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


def _norm_str(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return str(x).strip() or None


def _generate_password(*, length: int = 16) -> str:
    import secrets
    import string

    MIN_LEN = 12
    length = max(length, MIN_LEN)

    chars_upper = string.ascii_uppercase
    chars_lower = string.ascii_lowercase
    chars_digits = string.digits
    chars_special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    all_chars = chars_upper + chars_lower + chars_digits + chars_special

    pwd = (
        secrets.choice(chars_upper)
        + secrets.choice(chars_lower)
        + secrets.choice(chars_digits)
        + secrets.choice(chars_special)
        + "".join(secrets.choice(all_chars) for _ in range(length - 4))
    )
    pwd_list = list(pwd)
    secrets.SystemRandom().shuffle(pwd_list)
    return "".join(pwd_list)


class _ClientWriterPort(Protocol):
    def create_client_for_company(
        self,
        *,
        company_id: int,
        user_attrs: dict[str, Any],
        client_attrs: dict[str, Any],
    ) -> tuple[Any, Any]: ...


@dataclass(frozen=True, slots=True)
class CreateCompanyClientInput:
    """Input pour créer un client d'entreprise.

    Attributes:
        company_id: ID de l'entreprise
        validated_data: Données validées du client
    """

    company_id: int
    validated_data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CreateCompanyClientOutput:
    """Output pour créer un client d'entreprise.

    Attributes:
        success: True si l'opération a réussi
        client: Client créé (si succès)
        user: Utilisateur créé (si succès)
        generated_password: Mot de passe généré (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    client: Any | None = None
    user: Any | None = None
    generated_password: str | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class CreateCompanyClientUseCase:
    """Use-case Application: création d'un client (User + Client) pour une company.

    - Gère les règles SELF_SERVICE vs PRIVATE/CORPORATE.
    - Génère username + password.
    - La route reste responsable du commit + audit logging.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        client_writer: _ClientWriterPort,
        make_public_id_fn: Callable[[], str],
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            client_writer: Port pour créer un client
            make_public_id_fn: Fonction pour générer un ID public
        """
        self._writer = client_writer
        self._make_public_id = make_public_id_fn

    def execute(
        self, input_data: CreateCompanyClientInput
    ) -> CreateCompanyClientOutput:
        validated_data = input_data.validated_data
        client_type = _norm_str(validated_data.get("client_type"))
        if not client_type:
            return CreateCompanyClientOutput(
                success=False, error={"error": "client_type requis"}, status_code=400
            )

        ct_upper = str(client_type).upper()
        if ct_upper not in {"SELF_SERVICE", "PRIVATE", "CORPORATE"}:
            return CreateCompanyClientOutput(
                success=False,
                error={
                    "error": (
                        "client_type invalide. Valeurs possibles: "
                        "SELF_SERVICE, PRIVATE, CORPORATE"
                    )
                },
                status_code=400,
            )

        email = _norm_str(validated_data.get("email"))
        first_name = _norm_str(validated_data.get("first_name")) or ""
        last_name = _norm_str(validated_data.get("last_name")) or ""
        address = _norm_str(validated_data.get("address"))
        # ✅ Validation du genre obligatoire
        gender = _norm_str(validated_data.get("gender"))

        if ct_upper == "SELF_SERVICE" and not email:
            return CreateCompanyClientOutput(
                success=False,
                error={"error": "email requis pour self-service"},
                status_code=400,
            )

        if ct_upper != "SELF_SERVICE":
            missing = [
                f
                for f, v in (
                    ("first_name", first_name),
                    ("last_name", last_name),
                    ("address", address),
                    ("gender", gender),  # ✅ Le genre est maintenant obligatoire
                )
                if not v
            ]
            if missing:
                return CreateCompanyClientOutput(
                    success=False,
                    error={
                        "error": (
                            f"Champs manquants pour facturation : {', '.join(missing)}"
                        )
                    },
                    status_code=400,
                )

        # username (comportement identique à l'existant)
        if email:
            username = email.split("@")[0]
        else:
            import uuid

            fn = first_name.strip().lower()
            ln = last_name.strip().lower()
            username = f"{fn}.{ln}-{uuid.uuid4().hex[:6]}"

        pwd = _generate_password(length=16 if ct_upper == "SELF_SERVICE" else 20)

        user_attrs: dict[str, Any] = {
            "public_id": self._make_public_id(),
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": validated_data.get("phone"),
            "address": validated_data.get("address"),
            "birth_date": validated_data.get("birth_date"),
            # ✅ Civilité (gender) ajoutée
            "gender": validated_data.get("gender"),
            "role": "client",
            "password": pwd,
        }

        client_attrs: dict[str, Any] = {
            "client_type": ct_upper,
            "billing_address": validated_data.get("billing_address")
            or validated_data.get("address"),
            "billing_lat": validated_data.get("billing_lat"),
            "billing_lon": validated_data.get("billing_lon"),
            "contact_email": validated_data.get("contact_email") or email,
            "contact_phone": validated_data.get("contact_phone")
            or validated_data.get("phone"),
            "is_institution": bool(validated_data.get("is_institution", False)),
            "institution_name": validated_data.get("institution_name")
            if bool(validated_data.get("is_institution", False))
            else None,
            "domicile_address": validated_data.get("domicile_address"),
            "domicile_zip": validated_data.get("domicile_zip"),
            "domicile_city": validated_data.get("domicile_city"),
            "domicile_lat": validated_data.get("domicile_lat"),
            "domicile_lon": validated_data.get("domicile_lon"),
            "preferential_rate": validated_data.get("preferential_rate"),
            "notes": validated_data.get("notes"),
            "residence_facility": validated_data.get("residence_facility"),
            # ✅ Numéro AVS ajouté
            "avs_number": validated_data.get("avs_number"),
            # Accès logement
            "door_code": validated_data.get("door_code"),
            "floor": validated_data.get("floor"),
            "access_notes": validated_data.get("access_notes"),
            # Médecin traitant
            "gp_name": validated_data.get("gp_name"),
            "gp_phone": validated_data.get("gp_phone"),
            # Facturation par défaut
            "default_billed_to_type": validated_data.get("default_billed_to_type"),
            "default_billed_to_contact": validated_data.get("default_billed_to_contact"),
        }

        try:
            user, client = self._writer.create_client_for_company(
                company_id=input_data.company_id,
                user_attrs=user_attrs,
                client_attrs=client_attrs,
            )
            return CreateCompanyClientOutput(
                success=True, user=user, client=client, generated_password=pwd
            )
        except Exception:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception("Erreur lors de la création du client")
            return CreateCompanyClientOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
