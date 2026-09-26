#!/usr/bin/env python3
"""Sonde isolée du provider SMS (Twilio Programmable Messaging).

Ne log jamais : OTP, AUTH_TOKEN, SID (même partiel), numéro complet,
ni les noms d'env secrets associés à une valeur.

Usage (dans le container backend) :
    python -m scripts.probe_sms_provider
    python -m scripts.probe_sms_provider --to +41XXXXXXXX
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.notifications.phone_e164 import mask_phone_for_log, normalize_e164_phone
from services.notifications.sms import (
    describe_sms_config,
    get_sms_config,
    send_sms_notification,
)


def _print_config() -> dict:
    """Affiche uniquement des indicateurs de présence (pas de secrets)."""
    snapshot = describe_sms_config()
    credentials = (
        "PRESENT"
        if snapshot["twilio_account_sid"] == "PRESENT"
        and snapshot["twilio_auth_token"] == "PRESENT"
        else "MISSING"
    )
    sender = (
        "PRESENT"
        if snapshot["twilio_phone_number"] == "PRESENT"
        or snapshot["twilio_messaging_service_sid"] == "PRESENT"
        else "MISSING"
    )
    print("SMS PROVIDER PROBE")
    print("==================")
    print(f"mode                        : {snapshot['twilio_mode']}")
    print(f"enabled                     : {snapshot['enabled']}")
    print(f"credentials                 : {credentials}")
    print(f"sender                      : {sender}")
    print(f"sender_mode                 : {snapshot['sender_mode']}")
    print(f"ready                       : {snapshot['ready']}")
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Sonde le canal SMS Twilio.")
    parser.add_argument(
        "--to",
        dest="to_phone",
        default="",
        help="Numéro E.164 de contrôle (jamais le numéro d'un client bloqué).",
    )
    args = parser.parse_args()
    snapshot = _print_config()
    cfg = get_sms_config()

    if not args.to_phone:
        if not cfg.enabled:
            print("RESULT                      : CONFIG ERROR (SMS désactivé)")
            return 2
        if not cfg.has_credentials:
            print("RESULT                      : CONFIG ERROR (credentials)")
            return 2
        if not cfg.has_sender:
            print("RESULT                      : SENDER ERROR (expéditeur manquant)")
            return 2
        print("RESULT                      : READY (aucun envoi, --to omis)")
        return 0

    destination = normalize_e164_phone(args.to_phone)
    if not destination:
        print("RESULT                      : DESTINATION ERROR")
        print("destination                 : invalide")
        return 3

    print(f"destination                 : {mask_phone_for_log(destination)}")
    result = send_sms_notification(
        destination,
        "LIRIE: test de canal SMS. Ignorez ce message.",
        notification_type="sms_provider_probe",
    )
    # Presence uniquement — jamais le SID (même masqué).
    has_message_sid = bool(result.get("message_sid"))
    print(f"error_class                 : {result.get('error_class')}")
    print(f"provider_status             : {result.get('provider_status') or '-'}")
    print(f"provider_error_code         : {result.get('provider_error_code') or '-'}")
    print(f"message_sid_present         : {'yes' if has_message_sid else 'no'}")
    print(f"RESULT                      : {result.get('error_class')}")
    print(
        json.dumps(
            {
                "enabled": snapshot["enabled"],
                "ready": snapshot["ready"],
                "sender_mode": snapshot["sender_mode"],
                "twilio_mode": snapshot["twilio_mode"],
                "probe": result.get("error_class"),
                "message_sid_present": has_message_sid,
            },
            ensure_ascii=True,
        )
    )
    return 0 if result.get("ok") else 4


if __name__ == "__main__":
    raise SystemExit(main())
