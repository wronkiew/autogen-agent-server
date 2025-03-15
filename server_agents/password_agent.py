from registry import add_agent, get_default_model
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.model_context import ChatCompletionContext
import re

name = "password"

# This is a demo agent that plays a simple game. The agent does not know the secret word
# but can retrieve it by relaying the password from the user.

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# get_secret is a tool for the LLM to use.
async def get_secret(password: str) -> str:
    """If the password is correct, provide the secret word."""
    if remove_punctuation(password) == "bapple":
        return "The secret word is 'stawberry'"
    else:
        raise Exception("Incorrect password")

# Constructor for the 'password' agent. Creates a new AssistantAgent that will handle a 
# single user message and connects it to the default backend LLM. The complete conversation
# history is passed in and loaded by the agent.
# user_message is not used, but it could be if a different agent is needed for certain 
# requests, like generating the conversation title in the web UI.
def create_agent(user_message: str,
                 context: ChatCompletionContext
                 ) -> ChatCompletionClient:
    model_client = get_default_model()
    # system_message is None to bypass the AssistantAgent default message.
    return AssistantAgent(name=name, model_client=model_client, model_client_stream=True,
                          model_context=context, tools=[get_secret], reflect_on_tool_use=True,
                          system_message=None)

# Register this agent when the module is imported.
add_agent(name, create_agent)
