from logging.config import fileConfig

from sqlalchemy import Connection, engine_from_config, pool

from alembic import context

from pacer.config import get_settings
from pacer.repositories.tables import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# disable_existing_loggers=False keeps loggers of the code that called Alembic (e.g. pytest) working.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# The tables Alembic compares against the database when autogenerating.
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode, against a live database.

    Two ways in:
    - From the command line (`alembic upgrade head`): connect to the database in .env.
    - From code (the test fixtures in tests/conftest.py): the caller already has an open
      connection, e.g. to a throwaway test container, and passes it in
      config.attributes["connection"]. Then we use it and never read .env.
    """
    connection = config.attributes.get("connection")
    if connection is not None:
        _run_migrations(connection)
        return

    # Take the database URL from our settings (.env), not from alembic.ini.
    # '%' is escaped because alembic.ini values support %-interpolation.
    config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _run_migrations(connection)


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
