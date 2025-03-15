import sys
import logging
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class ModelInfo(BaseSettings):
    family: str = Field(..., description="LLM family")
    function_calling: bool = Field(..., description="Enable function calling")
    json_output: bool = Field(..., description="Enable JSON output")
    vision: bool = Field(..., description="Enable vision support")

class Settings(BaseSettings):
    # Required API/Backend configuration.
    openai_api_key: str = Field(..., description="OpenAI API key (must be set)")
    backend_url: str = Field("https://api.openai.com/v1", description="Backend URL")
    backend_request_timeout: int = Field(60, description="Timeout for backend requests (seconds)")
    default_llm: str = Field("gpt-4o-mini", description="Default LLM to use")

    # Optional fields for model info.
    default_llm_family: Optional[str] = Field(None, description="Default LLM family")
    default_llm_function_calling: Optional[bool] = Field(None, description="Enable function calling for LLM")
    default_llm_json_output: Optional[bool] = Field(None, description="Enable JSON output for LLM")
    default_llm_vision: Optional[bool] = Field(None, description="Enable vision support for LLM")
    model_info: Optional[ModelInfo] = None

    # Server configuration.
    server_host: str = Field("0.0.0.0", description="Host to listen on")
    server_port: int = Field(11435, description="Port to listen on")
    auto_reload: bool = Field(False, description="Enable auto-reload")

    agent_dir: str = Field("server_agents", description="Agent plugin directory")

    @field_validator("model_info", mode="before")
    def assemble_model_info(cls, v, info):
        mapping = {
            "default_llm_family": "family",
            "default_llm_function_calling": "function_calling",
            "default_llm_json_output": "json_output",
            "default_llm_vision": "vision",
        }
        values = info.data
        provided = {key: values.get(key) for key in mapping}
        if all(val is None for val in provided.values()):
            return None
        missing = [key for key, val in provided.items() if val is None]
        if missing:
            raise ValueError(
                f"Incomplete model info configuration. Missing: {', '.join(missing)}."
            )
        model_info_kwargs = {mapping[key]: provided[key] for key in mapping}
        return ModelInfo(**model_info_kwargs)

    def default_model_config(self, 
                             llm: str | None = None,
                             base_url: str | None = None,
                             timeout: int | None = None,
                             model_info: Optional[Dict] = None):
        config = {
            "provider": "OpenAIChatCompletionClient",
            "config": {
                "model": llm if llm else self.default_llm,
                "api_key": self.openai_api_key,
                "base_url": base_url if base_url else self.backend_url,
                "timeout": timeout if timeout else self.backend_request_timeout,
            }
        }
        if self.model_info is not None:
            config["config"]["model_info"] = self.model_info.model_dump()
        if model_info is not None:
            config["config"]["model_info"] = model_info
        return config

    def log_config(self, logger: Optional[logging.Logger] = None, mask_key: bool = True) -> None:
        """
        Log all configuration settings using the provided logger, or the default logger if none is provided.
        The openai_api_key is masked for security.
        """
        if logger is None:
            logger = logging.getLogger()
        # Dump the settings to a dict
        config_dict = self.model_dump()
        # Mask the API key
        if "openai_api_key" in config_dict and mask_key:
            config_dict["openai_api_key"] = "****"
        logger.info("Loaded configuration settings:")
        for key, value in config_dict.items():
            # Skip derived settings
            if key in ["model_info"]:
                continue
            logger.info("  %s: %s", key, value)

    class Config:
        env_prefix = ""
        env_file = ".env"

# Wrap instantiation in a try/except to catch configuration issues.
try:
    settings = Settings()
except ValidationError as exc:
    logger.debug("Configuration error when loading settings", exc_info=exc)
    # Extract a user-friendly error message from the validation errors.
    error_messages = []
    for error in exc.errors():
        # 'loc' is a tuple indicating where the error occurred.
        loc = error.get("loc", [])
        # Optionally remove technical parts like '__root__'
        if loc and loc[0] == "__root__":
            loc = loc[1:]
        loc_str = ".".join(map(str, loc)) if loc else "configuration"
        # 'msg' is the human-readable error message.
        msg = error.get("msg", "Invalid configuration")
        error_messages.append(f"{loc_str}: {msg}")
    final_error = "\n".join(error_messages)
    print(
        "Configuration Error: Failed to load settings. Please ensure that all required settings are properly set.\n"
        f"{final_error}"
    )
    sys.exit(1)
except Exception as exc:
    logger.debug("Configuration error when loading settings", exc_info=exc)
    print(
        "Configuration Error: An unexpected error occurred when loading settings. "
        "Please check your configuration."
    )
    sys.exit(1)