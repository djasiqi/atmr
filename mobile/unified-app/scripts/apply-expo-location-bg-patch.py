#!/usr/bin/env python3
"""Applique le patch natif expo-location pour le tracking BG Android 14+/16."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / (
    "node_modules/expo-location/android/src/main/java/expo/modules/location/taskConsumers/LocationTaskConsumer.kt"
)

PATCHED_SOURCE = ROOT / "native-patches/expo-location/LocationTaskConsumer.kt"


def main() -> None:
    if not PATCHED_SOURCE.exists():
        raise SystemExit(f"Source patch manquant: {PATCHED_SOURCE}")
    if not TARGET.parent.exists():
        raise SystemExit(f"expo-location non installé: {TARGET.parent}")

    TARGET.write_text(PATCHED_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"Copié vers: {TARGET}")


if __name__ == "__main__":
    main()
