import os
import importlib.util
import logging
from typing import Callable, List
from autogen_core.models import ChatCompletionClient
from autogen_core.model_context import ChatCompletionContext
from config import settings

agent_registry = {}

def add_agent(name: str,
              constructor: Callable[[str, ChatCompletionContext], ChatCompletionClient]
              ) -> None:
    """Register an agent with its constructor callback."""
    agent_registry[name] = constructor

def get_agent(name: str) -> Callable[[str, ChatCompletionContext], ChatCompletionClient]:
    """Retrieve the agent constructor by name."""
    return agent_registry.get(name)

def list_agents() -> List[str]:
    """List all registered agent names."""
    return list(agent_registry.keys())

def get_default_model() -> ChatCompletionClient:
    return ChatCompletionClient.load_component(settings.default_model_config())

def load_agent_files(directory, logger: logging.Logger):
    if not os.path.isdir(directory):
        logger.error(f"Agent directory {directory} does not exist")
        raise Exception(f"Agent directory {directory} does not exist")
    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Consider only Python files (and skip __init__.py if present)
        if filename.endswith('.py') and filename != '__init__.py':
            filepath = os.path.join(directory, filename)
            module_name = os.path.splitext(filename)[0]

            logger.info(f"Loading {module_name}")

            try:
                # Create a module spec from the file location
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)

                # Load the module (executes its top-level code)
                spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"Failed to load {module_name}")
                raise e

def get_logger() -> logging.Logger:
    return logging.getLogger("uvicorn.error")