import logging
import os
from logging.config import fileConfig

from alembic import context
from flask import current_app, has_app_context
from sqlalchemy import engine_from_config, pool

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


def get_engine_url():
    if has_app_context():
        try:
            engine = current_app.extensions["migrate"].db.get_engine()
        except (TypeError, AttributeError):
            engine = current_app.extensions["migrate"].db.engine
        try:
            return engine.url.render_as_string(hide_password=False).replace("%", "%%")
        except AttributeError:
            return str(engine.url).replace("%", "%%")

    return os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")


# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
engine_url = get_engine_url()
if engine_url:
    safe_engine_url = engine_url.replace("%", "%%")
    config.set_main_option("sqlalchemy.url", safe_engine_url)
else:
    safe_engine_url = None

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
        context.configure(connection=connection, target_metadata=get_metadata(), **conf_args)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
