"""Application settings, read from environment variables and the .env file."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    sql_echo: bool = False  # SQL_ECHO=true in .env prints every SQL statement (debugging)

    @property
    def database_url(self) -> str:
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


class LoggingSettings(BaseSettings):
    """Separate from Settings on purpose: logging must work even where there is no
    database configuration (CI, the fast tests), so nothing here is required."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"  # LOG_LEVEL=DEBUG in .env shows everything


class AISettings(BaseSettings):
    """Model access. Separate from Settings: nothing here is required, so the fast tests and CI
    run without any key. A missing key fails only when a real cloud call is attempted."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = None  # ANTHROPIC_API_KEY in .env, never in Git
    ollama_base_url: str = "http://localhost:11434"


@lru_cache
def get_ai_settings() -> AISettings:
    return AISettings()
