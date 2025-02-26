import base64

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict()

    HF_TOKEN: str = ""
    HF_USERNAME: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "https://cloud.langfuse.com/api/public/otel"
    SERPAPI_API_KEY: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def LANGFUSE_AUTH(self) -> str:
        return base64.b64encode(
            f"{self.LANGFUSE_PUBLIC_KEY}:{self.LANGFUSE_SECRET_KEY}".encode()
        ).decode()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def OTEL_EXPORTER_OTLP_HEADERS(self) -> str:
        return f"Authorization=Basic {self.LANGFUSE_AUTH}"


settings = Settings()
