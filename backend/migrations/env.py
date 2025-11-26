import logging
import os
from logging.config import fileConfig

from alembic import context
from flask import current_app, has_app_context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import URL

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
logger = logging.getLogger("alembic.env")


def get_engine():
    if has_app_context():
        try:
            return current_app.extensions["migrate"].db.get_engine()
        except (TypeError, AttributeError):
            return current_app.extensions["migrate"].db.engine

    section = config.get_section(config.config_ini_section) or {}
    section.setdefault("sqlalchemy.url", get_engine_url())
    return engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)


def get_database_url() -> str:
    """
    Source de vérité pour l'URL de base de données.

    Ordre de priorité :
      1) DATABASE_URL (env)
      2) SQLALCHEMY_DATABASE_URI (env)
      3) Reconstruction à partir des variables POSTGRES_* (env)

    ⚠️ IMPORTANT: On utilise directement les variables d'environnement
    sans les reconstruire pour éviter les problèmes d'échappement.
    Si reconstruction nécessaire, on utilise URL.create() qui gère correctement
    les caractères spéciaux dans le mot de passe.
    """
    # Priorité 1: DATABASE_URL (utilisé en priorité par Flask config.py)
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        logger.info(
            "[Alembic] Using DB URL from DATABASE_URL: %s@***", env_url.split("@")[0]
        )
        return env_url

    # Priorité 2: SQLALCHEMY_DATABASE_URI
    env_url = os.getenv("SQLALCHEMY_DATABASE_URI")
    if env_url:
        logger.info(
            "[Alembic] Using DB URL from SQLALCHEMY_DATABASE_URI: %s@***",
            env_url.split("@")[0],
        )
        return env_url

    # Priorité 3: Flask app context (si disponible)
    if has_app_context():
        try:
            engine = current_app.extensions["migrate"].db.get_engine()
        except (TypeError, AttributeError):
            engine = current_app.extensions["migrate"].db.engine
        try:
            url_str = engine.url.render_as_string(hide_password=False)
            logger.info(
                "[Alembic] Using DB URL from Flask app context: %s@***",
                url_str.split("@")[0],
            )
            return url_str
        except AttributeError:
            url_str = str(engine.url)
            logger.info(
                "[Alembic] Using DB URL from Flask app context: %s@***",
                url_str.split("@")[0],
            )
            return url_str

    # Priorité 4: Reconstruction propre avec URL.create() pour gérer les caractères spéciaux
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "")
    pg_host = os.getenv("POSTGRES_HOST", "postgres")
    pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_db = os.getenv("POSTGRES_DB", pg_user)

    # Utiliser URL.create() qui gère correctement l'échappement des caractères spéciaux
    url_obj = URL.create(
        "postgresql+psycopg2",
        username=pg_user,
        password=pg_password,
        host=pg_host,
        port=pg_port,
        database=pg_db,
    )
    url_str = str(url_obj)
    logger.info(
        "[Alembic] Using DB URL reconstructed from POSTGRES_*: %s@***",
        url_str.split("@")[0],
    )
    return url_str


def get_engine_url():
    """Alias pour compatibilité avec le code existant."""
    return get_database_url()


# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# Note: L'URL sera forcée dans run_migrations_online() via get_database_url()
# On ne la définit pas ici pour éviter les problèmes d'échappement et de reconstruction

target_db = current_app.extensions["migrate"].db if has_app_context() else None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_metadata():
    if target_db is None:
        return None
    if hasattr(target_db, "metadatas"):
        return target_db.metadatas[None]
    return target_db.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=get_metadata(), literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # ⚠️ On force l'URL utilisée par Alembic avant de créer l'engine
    # Cela garantit que DATABASE_URL/SQLALCHEMY_DATABASE_URI est utilisé tel quel
    # sans reconstruction qui pourrait casser l'URL (ex: host="37_46!!@postgres")
    db_url = get_database_url()
    config.set_main_option("sqlalchemy.url", db_url)
    logger.info("[Alembic] Database URL forced in config: %s@***", db_url.split("@")[0])

    # this callback is used to prevent an auto-migration from being generated
    # when there are no changes to the schema
    # reference: http://alembic.zzzcomputing.com/en/latest/cookbook.html
    def process_revision_directives(context, revision, directives):
        del context  # unused but required by Alembic signature
        del revision  # unused but required by Alembic signature
        if getattr(config.cmd_opts, "autogenerate", False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info("No changes in schema detected.")

    conf_args = {}
    if has_app_context():
        conf_args = current_app.extensions["migrate"].configure_args
        if conf_args.get("process_revision_directives") is None:
            conf_args["process_revision_directives"] = process_revision_directives

    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=get_metadata(), **conf_args
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
