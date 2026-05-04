"""
Helpers PostgreSQL pour migrations idempotentes (Alembic).
Utilise information_schema / pg_* pour éviter DuplicateTable/DuplicateObject.

Schema: tous les helpers prennent schema="public" par défaut. Si l'environnement
utilise un autre schéma (ex: multi-tenant), passer le schema explicitement.
"""

from sqlalchemy import text


def table_exists(bind, name: str, schema: str = "public") -> bool:
    """Vrai si la table existe (PostgreSQL information_schema)."""
    q = text(
        "SELECT EXISTS ("
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = :s AND table_name = :t"
        ")"
    )
    return bind.execute(q, {"s": schema, "t": name}).scalar()


def index_exists(bind, index_name: str, schema: str = "public") -> bool:
    """Vrai si l'index existe (PostgreSQL pg_indexes)."""
    q = text(
        "SELECT EXISTS ("
        "SELECT 1 FROM pg_indexes "
        "WHERE schemaname = :s AND indexname = :n"
        ")"
    )
    return bind.execute(q, {"s": schema, "n": index_name}).scalar()


def constraint_exists(
    bind, table_name: str, constraint_name: str, schema: str = "public"
) -> bool:
    """Vrai si la contrainte existe (PostgreSQL information_schema)."""
    q = text(
        "SELECT EXISTS ("
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE table_schema = :s AND table_name = :t AND constraint_name = :c"
        ")"
    )
    return bind.execute(
        q, {"s": schema, "t": table_name, "c": constraint_name}
    ).scalar()


def column_exists(
    bind, table_name: str, column_name: str, schema: str = "public"
) -> bool:
    """Vrai si la colonne existe (PostgreSQL information_schema.columns)."""
    q = text(
        "SELECT EXISTS ("
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = :s AND table_name = :t AND column_name = :c"
        ")"
    )
    return bind.execute(q, {"s": schema, "t": table_name, "c": column_name}).scalar()


def get_fk_constraint_name(
    bind,
    table_name: str,
    column_name: str,
    referred_table: str,
    schema: str = "public",
    referred_column: str = "id",
) -> str | None:
    """Nom de la contrainte FK (table.col → referred_table.referred_column), ou None.

    Détection par définition (information_schema), pas par nom prédéfini.
    Évite les faux négatifs si Alembic/naming convention ou DDL manuel a donné un autre nom.
    """
    q = text(
        "SELECT tc.constraint_name FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name "
        "  AND tc.table_schema = kcu.table_schema AND tc.table_name = kcu.table_name "
        "JOIN information_schema.constraint_column_usage ccu "
        "  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema "
        "WHERE tc.constraint_type = 'FOREIGN KEY' "
        "  AND tc.table_schema = :s AND tc.table_name = :t AND kcu.column_name = :col "
        "  AND ccu.table_schema = :s AND ccu.table_name = :ref_t AND ccu.column_name = :ref_col "
        "LIMIT 1"
    )
    row = bind.execute(
        q,
        {
            "s": schema,
            "t": table_name,
            "col": column_name,
            "ref_t": referred_table,
            "ref_col": referred_column,
        },
    ).fetchone()
    return row[0] if row else None


def fk_exists(
    bind,
    table_name: str,
    column_name: str,
    referred_table: str,
    schema: str = "public",
    referred_column: str = "id",
) -> bool:
    """Vrai si une FK existe de (table, column) vers (referred_table, referred_column)."""
    return (
        get_fk_constraint_name(
            bind, table_name, column_name, referred_table, schema, referred_column
        )
        is not None
    )
