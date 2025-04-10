from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.singleton import singleton


@singleton
class AppConfig(BaseSettings):
    seed: int = Field(default=42, description="Seed for random number generation.")

    WANDB_API_KEY: str = None

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env")
    )


config = AppConfig()
