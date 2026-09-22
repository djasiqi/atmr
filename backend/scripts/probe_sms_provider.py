#!/usr/bin/env python3
"""Sonde isolée du provider SMS (Twilio Programmable Messaging).

Ne log jamais : OTP, AUTH_TOKEN, SID complet, numéro complet.

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
    snapshot = describe_sms_config()
    print("SMS PROVIDER PROBE")
    print("==================")
    print(f"TWILIO MODE                 : {snapshot['twilio_mode']}")
    print(f"SMS_NOTIFICATIONS_ENABLED   : {snapshot['enabled']}")
    print(f"TWILIO_ACCOUNT_SID          : {snapshot['twilio_account_sid']}")
    print(f"TWILIO_AUTH_TOKEN           : {snapshot['twilio_auth_token']}")
    print(f"TWILIO_PHONE_NUMBER         : {snapshot['twilio_phone_number']}")
    print(f"TWILIO_MESSAGING_SERVICE_SID: {snapshot['twilio_messaging_service_sid']}")
    print(f"SENDER MODE                 : {snapshot['sender_mode']}")
    print(f"PROVIDER READY              : {snapshot['ready']}")
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
    print(f"error_class                 : {result.get('error_class')}")
    print(f"provider_status             : {result.get('provider_status') or '-'}")
    print(f"provider_error_code         : {result.get('provider_error_code') or '-'}")
    print(f"message_sid                 : {result.get('message_sid') or '-'}")
    print(f"RESULT                      : {result.get('error_class')}")
    print(
        json.dumps({**snapshot, "probe": result.get("error_class")}, ensure_ascii=True)
    )
    return 0 if result.get("ok") else 4


if __name__ == "__main__":
    raise SystemExit(main())
