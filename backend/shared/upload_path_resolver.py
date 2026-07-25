"""Résolution sécurisée des chemins fichiers sous uploads/ (Lot 0 SEC-06)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from flask import Response, current_app
from werkzeug.exceptions import NotFound

PUBLIC_UPLOAD_PREFIXES = ("company_logos/", "institution_logos/")


def extract_upload_relative_path(stored_url: str) -> str:
    """Extrait le chemin relatif sous uploads/ depuis une URL stockée."""
    if not stored_url or not str(stored_url).strip():
        raise NotFound()
    raw = str(stored_url).strip()
    parsed = urlparse(raw)
    path = parsed.path if parsed.scheme else raw
    path = path.replace("\\", "/")
    if path.startswith("/uploads/"):
        return path[len("/uploads/") :]
    if path.startswith("uploads/"):
        return path[len("uploads/") :]
    # Chemin déjà relatif (ex. invoices/foo.pdf)
    return path.lstrip("/")


def resolve_safe_upload_path(
    stored_url: str,
    *,
    uploads_base: Path,
) -> Path:
    """Résout un chemin fichier sous uploads_base après résolution des symlinks.

    Lève NotFound si le chemin sort du répertoire autorisé.
    """
    relative = extract_upload_relative_path(stored_url)
    if not relative or ".." in relative.split("/"):
        raise NotFound()

    base = Path(uploads_base).resolve()
    try:
        candidate = (base / relative).resolve()
        candidate.relative_to(base)
    except (ValueError, RuntimeError, OSError) as exc:
        raise NotFound() from exc

    if not candidate.is_file():
        raise NotFound()
    return candidate


def is_public_upload_prefix(filename: str) -> bool:
    """True si le chemin relatif est un préfixe public (logos)."""
    normalized = str(filename or "").replace("\\", "/").lstrip("/")
    return any(normalized.startswith(prefix) for prefix in PUBLIC_UPLOAD_PREFIXES)


def get_uploads_base() -> Path:
    return Path(
        current_app.config.get("UPLOADS_DIR")
        or current_app.config.get("UPLOAD_FOLDER")
        or (Path(current_app.root_path) / "uploads")
    ).resolve()


def build_file_response(candidate: Path, *, as_attachment: bool = False) -> Response:
    """Construit une Response Flask depuis un fichier déjà validé."""
    import mimetypes as _mt

    mimetypes_map = {
        ".pdf": "application/pdf",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
    }
    ext = candidate.suffix.lower()
    mimetype = mimetypes_map.get(ext)
    if mimetype is None:
        guessed, _ = _mt.guess_type(candidate.name) or (None, None)
        mimetype = guessed or "application/octet-stream"

    inline_extensions = {
        ".pdf",
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    }
    disposition = (
        "attachment"
        if as_attachment or ext not in inline_extensions
        else "inline"
    )
    data = candidate.read_bytes()
    headers = {
        "Content-Length": str(len(data)),
        "Content-Disposition": f'{disposition}; filename="{candidate.name}"',
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=300",
    }
    return Response(data, mimetype=mimetype, headers=headers)


def serve_stored_upload(stored_url: str, *, as_attachment: bool = False) -> Response:
    """Résout et sert un fichier privé à partir de l'URL stockée en base."""
    candidate = resolve_safe_upload_path(stored_url, uploads_base=get_uploads_base())
    return build_file_response(candidate, as_attachment=as_attachment)
