#!/usr/bin/env python3
"""Script pour générer une URL de base de données avec échappement correct du mot de passe.

Usage:
    python scripts/generate_database_url.py USER PASSWORD HOST PORT DB

Écrit l'URL dans database_url.env (mode 0600). N'affiche pas le mot de passe.

Exemple:
    python scripts/generate_database_url.py atmr "VOTRE_MOT_DE_PASSE" postgres 5432 atmr
"""

import contextlib
import sys
from pathlib import Path
from urllib.parse import quote_plus


def generate_database_url(
    user: str, password: str, host: str, port: str, db: str
) -> str:
    """Génère une URL PostgreSQL avec échappement correct du mot de passe.

    Args:
        user: Nom d'utilisateur PostgreSQL
        password: Mot de passe (peut contenir des caractères spéciaux)
        host: Hôte PostgreSQL
        port: Port PostgreSQL
        db: Nom de la base de données

    Returns:
        URL PostgreSQL avec mot de passe échappé
    """
    # Échapper le mot de passe pour l'URL
    password_escaped = quote_plus(password)

    # Construire l'URL
    url = f"postgresql+psycopg2://{user}:{password_escaped}@{host}:{port}/{db}"

    return url


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    user = sys.argv[1]
    password = sys.argv[2]
    host = sys.argv[3]
    port = sys.argv[4]
    db = sys.argv[5]

    url = generate_database_url(user, password, host, port, db)
    out_path = Path.cwd() / "database_url.env"
    out_path.write_text(f"{url}\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        out_path.chmod(0o600)
    print(
        f"URL écrite dans {out_path.resolve()} (mode 0600). "
        "Ne pas journaliser ni coller ce fichier dans un ticket."
    )
