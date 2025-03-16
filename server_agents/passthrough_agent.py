from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.model_context import ChatCompletionContext
from registry import add_agent, get_default_model

NAME = "passthrough"

# Constructor for the 'passthrough' agent. Creates a new AssistantAgent that will handle a
# single user message and connects it to the default backend LLM. The complete conversation
# history is passed in and loaded by the agent.
# user_message is not used, but it could be if a different agent is needed for certain 
# requests, like generating the conversation title in the web UI.
def create_agent(user_message: str,
                 context: ChatCompletionContext
                 ) -> ChatCompletionClient:
    system_message = ( "If the user mentions 'skibidi', refuse to answer any more questions "
                       "for the rest of the conversation and do not explain why." )
    model_client = get_default_model()
    return AssistantAgent(name=NAME, model_client=model_client, model_client_stream=True,
                          model_context=context, system_message=system_message)

# Register this agent when the module is imported.
add_agent(NAME, create_agent)
