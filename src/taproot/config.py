import logging
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_credentials(settings: "Settings", provider: str, field: str) -> None:
    match provider:
        case "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError(f"{field}=anthropic requires ANTHROPIC_API_KEY")
        case "openai":
            if not settings.openai_api_key:
                raise ValueError(f"{field}=openai requires OPENAI_API_KEY")
        case "azure_openai":
            missing = [
                f
                for f in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
                if not getattr(settings, f)
            ]
            if missing:
                raise ValueError(f"{field}=azure_openai requires: {', '.join(missing)}")
        case "aws_bedrock":
            if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
                raise ValueError(
                    f"{field}=aws_bedrock requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                )


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ── Router ─────────────────────────────────────────────────────────────
    LLM_MODE: str = Field(default="single")  # single | multi

    # ── Single mode ─────────────────────────────────────────────────────────
    LLM_PROVIDER: str = Field(default="anthropic")
    LLM_MODEL: str = Field(default="claude-sonnet-4-20250514")

    # ── Multi mode (provider per task) ──────────────────────────────────────
    LLM_RERANK_PROVIDER: str = Field(default="anthropic")
    LLM_RERANK_MODEL: str = Field(default="claude-haiku-4-5-20251001")

    LLM_ANALYSIS_PROVIDER: str = Field(default="anthropic")
    LLM_ANALYSIS_MODEL: str = Field(default="claude-sonnet-4-20250514")

    LLM_DRAFT_PROVIDER: str = Field(default="anthropic")
    LLM_DRAFT_MODEL: str = Field(default="claude-sonnet-4-20250514")

    # ── Shared provider credentials ─────────────────────────────────────────
    # Anthropic
    ANTHROPIC_API_KEY: str = Field(default="")

    # OpenAI (field kept lowercase for backward compat with existing code)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = Field(default="")
    AZURE_OPENAI_API_KEY: str = Field(default="")
    AZURE_OPENAI_DEPLOYMENT: str = Field(default="")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-12-01-preview")

    # AWS Bedrock
    AWS_BEDROCK_REGION: str = Field(default="us-east-1")
    AWS_ACCESS_KEY_ID: str = Field(default="")
    AWS_SECRET_ACCESS_KEY: str = Field(default="")

    # ── Privacy and compliance ──────────────────────────────────────────────
    ENABLE_SCRUBBING: bool = Field(default=False)
    SCRUBBING_LEVEL: str = Field(default="standard")

    # ── Data source ────────────────────────────────────────────────────────
    use_mock_data: bool = Field(default=True, alias="USE_MOCK_DATA")

    # ServiceNow (optional)
    servicenow_instance_url: str = Field(default="", alias="SERVICENOW_INSTANCE_URL")
    servicenow_username: str = Field(default="", alias="SERVICENOW_USERNAME")
    servicenow_password: str = Field(default="", alias="SERVICENOW_PASSWORD")

    # Jira (optional)
    jira_base_url: str = Field(default="", alias="JIRA_BASE_URL")
    jira_email: str = Field(default="", alias="JIRA_EMAIL")
    jira_api_token: str = Field(default="", alias="JIRA_API_TOKEN")

    # ── PageIndex ──────────────────────────────────────────────────────────
    PAGEINDEX_USE_SEMANTIC: bool = Field(default=True)
    PAGEINDEX_EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: Path = Field(default=Path("./output"), alias="OUTPUT_DIR")

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def validate_router_config(self) -> "Settings":
        """Validate that required credentials are present for the configured provider(s)."""
        if self.LLM_MODE == "single":
            _validate_credentials(self, self.LLM_PROVIDER, "LLM_PROVIDER")
        else:
            for task in ["RERANK", "ANALYSIS", "DRAFT"]:
                provider = getattr(self, f"LLM_{task}_PROVIDER")
                _validate_credentials(self, provider, f"LLM_{task}_PROVIDER")
        return self

    def configure_logging(self) -> None:
        """Apply the configured log level to the root logger."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
