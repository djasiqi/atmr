"""Restauration d'un PDF original depuis une sauvegarde vérifiable.

Ne génère aucun document. Ne suppose pas qu'une archive existe.
Sans ``--apply``, aucune écriture. Un fichier de destination déjà présent
et différent est refusé : on ne remplace pas un original par un autre contenu.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_original_pdf_restore(
    *,
    source: Path,
    destination: Path,
    expected_sha256: str,
    apply: bool,
) -> dict[str, object]:
    """Vérifie une sauvegarde et, seulement si ``apply``, copie le fichier.

    Le contenu copié est celui de ``source`` après contrôle d'empreinte.
    Aucun PDF n'est fabriqué.
    """
    expected = (expected_sha256 or "").strip().lower()
    if not source.is_file():
        return {"status": "refused", "reason": "source_missing", "applied": False}
    if len(expected) != 64:
        return {"status": "refused", "reason": "sha256_invalid", "applied": False}

    digest = file_sha256(source)
    if digest != expected:
        return {"status": "refused", "reason": "sha256_mismatch", "applied": False}

    if destination.exists():
        current = file_sha256(destination)
        if current == digest:
            return {
                "status": "already_present",
                "reason": "identical",
                "applied": False,
                "sha256": digest,
            }
        return {
            "status": "refused",
            "reason": "destination_differs",
            "applied": False,
            "sha256": digest,
        }

    if not apply:
        return {
            "status": "preview",
            "reason": "dry_run",
            "applied": False,
            "sha256": digest,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".restore-tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)
    if file_sha256(destination) != digest:
        return {"status": "refused", "reason": "copy_mismatch", "applied": False}
    return {"status": "restored", "reason": "copied", "applied": True, "sha256": digest}
