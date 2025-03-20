import argparse
import sys
import logging
from typing import Optional, Dict, get_origin, get_args, Union
from pydantic import ValidationError, Field, field_validator
from pydantic_settings import BaseSettings

config_logger = logging.getLogger(__name__)

class ModelInfo(BaseSettings):
    """
    Encapsulates configuration details that describe a language model's capabilities.
    """
    family: str = Field(..., description="LLM family")
    function_calling: bool = Field(..., description="Enable function calling")
    json_output: bool = Field(..., description="Enable JSON output")
    vision: bool = Field(..., description="Enable vision support")

class Settings(BaseSettings):
    """
    Aggregates configuration parameters for connecting to the OpenAI API backend,
    setting up default language model configurations, and configuring the server.
    """
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

    debug_log: bool = Field(False, description="Enable debug logging for the agent server")
    autogen_debug_log: bool = Field(False, description="Enable debug logging for AutoGen")

    headless: bool = Field(False, description="Disable program elements requiring a display")

    @field_validator("model_info", mode="before")
    @classmethod
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
        model_info_kwargs = {dest: provided[src] for src, dest in mapping.items()}
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
            if key in ["model_info"]:
                continue
            logger.info("  %s: %s", key, value)

    class Config: # pylint: disable=too-few-public-methods
        """
        Pydantic configuration for the Settings model.
        """
        env_prefix = ""
        env_file = ".env"

# Helper: decide if a type is "simple" (i.e. str, int, float, bool)
def is_simple_type(field_type: type) -> bool:
    return field_type in (str, int, float, bool)

# Helper: add a field to the argparse parser
def add_field_to_parser(_parser: argparse.ArgumentParser, field_name: str, field, default_value):
    field_type = field.annotation
    origin = get_origin(field_type)
    if origin is Union:
        field_args = get_args(field_type)
        non_none = [arg for arg in field_args if arg is not type(None)]
        if len(non_none) == 1:
            field_type = non_none[0]
    if not is_simple_type(field_type):
        return

    description = field.description or ""
    # Use argparse.SUPPRESS for the default value so that if the flag is not provided,
    # the key will not be present in the parsed arguments.
    suppress_default = argparse.SUPPRESS

    if field_type is bool:
        def str_to_bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ("true", "1", "yes"):
                return True
            elif v.lower() in ("false", "0", "no"):
                return False
            else:
                raise argparse.ArgumentTypeError("Boolean value expected (true/false).")
        _parser.add_argument(f"--{field_name}", type=str_to_bool, default=suppress_default,
                            help=f"{description} (bool) (default: {default_value})")
    else:
        _parser.add_argument(f"--{field_name}", type=field_type, default=suppress_default,
                            help=f"{description} (default: {default_value})")

# Generate an argparse parser automatically from the Settings model.
def generate_arg_parser_from_settings(settings_cls: type[BaseSettings]) -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser(
        description="Agent Server Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for field_name, field in settings_cls.model_fields.items():
        if field_name == "model_info":
            continue
        add_field_to_parser(_parser, field_name, field, field.default)
    return _parser

# Load the settings once.
try:
    arg_parser = generate_arg_parser_from_settings(Settings)
    args = arg_parser.parse_args()
    settings = Settings(**vars(args))
except ValidationError as exc:
    config_logger.debug("Configuration error when loading settings", exc_info=exc)
    error_messages = []
    for error in exc.errors():
        loc = error.get("loc", [])
        if loc and loc[0] == "__root__":
            loc = loc[1:]
        LOC_STR = ".".join(map(str, loc)) if loc else "configuration"
        msg = error.get("msg", "Invalid configuration")
        error_messages.append(f"{LOC_STR}: {msg}")
    FINAL_ERROR = "\n".join(error_messages)
    print(
        "Configuration Error: Failed to load settings. Please ensure that all required settings are properly set.\n"
        f"{FINAL_ERROR}"
    )
    sys.exit(1)
except Exception as exc:
    config_logger.debug("Configuration error when loading settings", exc_info=exc)
    print(
        "Configuration Error: An unexpected error occurred when loading settings. "
        "Please check your configuration."
    )
    sys.exit(1)
