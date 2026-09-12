"""Résolution sécurisée des chemins fichiers sous uploads/ (Lot 0 SEC-06)."""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from urllib.parse import urlparse

from flask import Response, current_app
from werkzeug.exceptions import NotFound
from werkzeug.utils import safe_join

PUBLIC_UPLOAD_PREFIXES = ("company_logos/", "institution_logos/")


class InvalidUploadPath(ValueError):
    """Chemin hors de la racine uploads ou segment non confinable."""


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


def _resolved_uploads_base(uploads_base: Path | None = None) -> Path:
    if uploads_base is not None:
        return Path(uploads_base).resolve()
    return get_uploads_base()


def canonical_upload_extension(
    filename: str, allowed: set[str] | frozenset[str]
) -> str:
    """Extension whitelistée du dernier segment. Jamais un sous-chemin."""
    if not filename or not str(filename).strip():
        raise InvalidUploadPath("nom de fichier manquant")
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    if name in {".", ".."} or "." not in name:
        raise InvalidUploadPath("extension manquante")
    ext = name.rsplit(".", 1)[1].lower()
    if not ext.isalnum():
        raise InvalidUploadPath("extension invalide")
    allowed_map = {item: item for item in allowed}
    canonical = allowed_map.get(ext)
    if canonical is None:
        raise InvalidUploadPath("extension non autorisée")
    return canonical


def server_upload_filename(extension: str, *, prefix: str | None = None) -> str:
    """Nom serveur : UUID + extension canonique. Aucune entrée utilisateur."""
    if (
        not extension
        or not extension.isalnum()
        or "/" in extension
        or "\\" in extension
    ):
        raise InvalidUploadPath("extension invalide")
    stem = uuid.uuid4().hex
    if prefix:
        if "/" in prefix or "\\" in prefix or ".." in prefix:
            raise InvalidUploadPath("préfixe invalide")
        return f"{prefix}_{stem}.{extension}"
    return f"{stem}.{extension}"


def _uploads_ancestor(resolved: Path) -> Path | None:
    """Racine ``uploads/`` du chemin déjà résolu (suit les symlinks)."""
    for parent in (resolved, *resolved.parents):
        if parent.name == "uploads":
            return parent
    return None


def _join_under_base(base: Path, relative: str) -> Path:
    parts = relative.split("/")
    if not relative or any(part in {"", ".."} for part in parts):
        raise InvalidUploadPath("chemin invalide")
    joined = safe_join(str(base), relative)
    if joined is None:
        raise InvalidUploadPath("chemin invalide")
    try:
        candidate = Path(joined).resolve()
        candidate.relative_to(base)
    except (ValueError, RuntimeError, OSError) as exc:
        raise InvalidUploadPath("chemin hors uploads") from exc
    return candidate


def confine_upload_destination(
    target: Path | str,
    *,
    uploads_base: Path | None = None,
) -> Path:
    """Confine un chemin (existant ou non) sous une racine uploads via safe_join.

    Lève InvalidUploadPath si le candidat sort de la racine (traversal, absolu
    hors uploads, symlink hors base). N'ouvre pas le fichier.

    Un chemin absolu déjà sous un dossier nommé ``uploads/`` reste accepté
    même si ``UPLOADS_DIR`` courant diffère (tests / service qui a figé sa
    racine). La lecture via ``resolve_safe_upload_path`` passe toujours une
    racine explicite et des chemins relatifs.
    """
    raw = Path(target)
    if raw.is_absolute():
        try:
            resolved = raw.resolve()
        except (RuntimeError, OSError) as exc:
            raise InvalidUploadPath("chemin invalide") from exc
        bases: list[Path] = []
        if uploads_base is not None:
            bases.append(Path(uploads_base).resolve())
        else:
            with contextlib.suppress(RuntimeError):
                bases.append(get_uploads_base())
        inferred = _uploads_ancestor(resolved)
        if inferred is not None and inferred not in bases:
            bases.append(inferred)
        last_exc: Exception | None = None
        for base in bases:
            try:
                return _join_under_base(base, resolved.relative_to(base).as_posix())
            except (ValueError, InvalidUploadPath) as exc:
                last_exc = exc
                continue
        raise InvalidUploadPath("chemin hors uploads") from last_exc

    base = _resolved_uploads_base(uploads_base)
    relative = raw.as_posix().replace("\\", "/").lstrip("/")
    return _join_under_base(base, relative)


def build_confined_upload_path(
    *parts: str,
    uploads_base: Path | None = None,
) -> Path:
    """Construit un chemin sous uploads à partir de segments sans séparateur."""
    if not parts:
        raise InvalidUploadPath("segments requis")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or "/" in part
            or "\\" in part
            or ".." in part
        ):
            raise InvalidUploadPath("segment invalide")
    base = _resolved_uploads_base(uploads_base)
    joined = safe_join(str(base), *parts)
    if joined is None:
        raise InvalidUploadPath("chemin invalide")
    try:
        candidate = Path(joined).resolve()
        candidate.relative_to(base)
    except (ValueError, RuntimeError, OSError) as exc:
        raise InvalidUploadPath("chemin hors uploads") from exc
    return candidate


def resolve_safe_upload_path(
    stored_url: str,
    *,
    uploads_base: Path,
) -> Path:
    """Résout un fichier existant sous uploads_base après résolution des symlinks.

    Lève NotFound si le chemin sort du répertoire autorisé ou n'existe pas.
    """
    relative = extract_upload_relative_path(stored_url)
    try:
        candidate = confine_upload_destination(relative, uploads_base=uploads_base)
    except InvalidUploadPath as exc:
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


def _safe_content_disposition_filename(filename: str | None, fallback: str) -> str:
    """Nom ASCII sûr pour l'en-tête Content-Disposition (filename=)."""
    import re
    import unicodedata

    raw = (filename or "").strip() or fallback
    normalized = unicodedata.normalize("NFKD", raw)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace('"', "").replace("\\", "").replace("/", "_")
    ascii_name = re.sub(r"[\r\n\t]+", " ", ascii_name).strip() or fallback
    return ascii_name[:180]


def build_file_response(
    candidate: Path,
    *,
    as_attachment: bool = False,
    download_filename: str | None = None,
) -> Response:
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
        ".mp3",
        ".m4a",
        ".aac",
        ".wav",
        ".ogg",
        ".caf",
        ".3gp",
        ".webm",
    }
    disposition = (
        "attachment" if as_attachment or ext not in inline_extensions else "inline"
    )
    safe_name = _safe_content_disposition_filename(download_filename, candidate.name)
    data = candidate.read_bytes()
    headers = {
        "Content-Length": str(len(data)),
        "Content-Disposition": f'{disposition}; filename="{safe_name}"',
        "X-Content-Type-Options": "nosniff",
        # Pas de cache navigateur : même URL API sert un fichier régénéré (ex. facture PDF).
        "Cache-Control": "private, no-store, must-revalidate",
        "Pragma": "no-cache",
    }
    return Response(data, mimetype=mimetype, headers=headers)


def serve_stored_upload(
    stored_url: str,
    *,
    as_attachment: bool = False,
    download_filename: str | None = None,
) -> Response:
    """Résout et sert un fichier privé à partir de l'URL stockée en base."""
    candidate = resolve_safe_upload_path(stored_url, uploads_base=get_uploads_base())
    return build_file_response(
        candidate,
        as_attachment=as_attachment,
        download_filename=download_filename,
    )
