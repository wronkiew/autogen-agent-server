import asyncio
from config import settings
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_core.model_context import ChatCompletionContext
from registry import add_agent, get_default_model, get_logger

NAME = "web_surfer"

# --------------------------------------------------------------------------
# async_web_surf
#
# Asyncrhonous helper function for the web_surf tool. Performs a web-
# browsing session, followed by summarizing the results.
# --------------------------------------------------------------------------
async def async_web_surf(instructions: str) -> str:
    logger = get_logger()
    logger.debug("web_surf instructions: %s", instructions)

    # Switch to a less expensive LLM if the default is gpt-4o,
    # because real-time web browsing can use a lot of tokens.
    browsing_llm = settings.default_llm
    if browsing_llm == "gpt-4o":
        browsing_llm = "gpt-4o-mini"

    # Create a model client (the LLM interface) using the chosen browsing_llm.
    model_client = ChatCompletionClient.load_component(
        settings.default_model_config(llm=browsing_llm, timeout=3600)
    )

    # Build a MultimodalWebSurfer agent for interacting with live webpages.
    # 'headless=False' opens a browser window, 'animate_actions=True' displays each click step.
    web_surfer_agent = MultimodalWebSurfer(
        "web_surfer",
        model_client,
        headless=False,
        animate_actions=True
    )

    # Use a RoundRobinGroupChat with just one participant so it can take multiple steps:
    # searching, navigating links, reading text, etc., up to 6 turns total.
    team = RoundRobinGroupChat([web_surfer_agent], max_turns=6)

    # Execute the browsing session, capturing the resulting messages/outputs.
    web_answer = await team.run(task=instructions)
    logger.debug("web_surf answer: %s", web_answer)

    # Create a second model client for summarizing the browser session results.
    summarize_client = ChatCompletionClient.load_component(
        settings.default_model_config(llm=browsing_llm, timeout=3600)
    )

    # Build a temporary "summarize" agent that condenses what was found by the web surfer.
    summarize_instructions = "You are a helpful AI assistant."
    summarize_agent = AssistantAgent(
        "summarize",
        summarize_client,
        system_message=summarize_instructions
    )

    # Extract the browser session messages and append a user-like instruction
    # requesting a thorough summary including source URLs.
    web_answer = web_answer.messages
    summarize_request = (
        "Analyze the web browsing results. Restate the original task. "
        "Generate a detailed summary including any text relevant to the "
        "task and source URLs for that text."
    )
    web_answer.append(TextMessage(content=summarize_request, source="user"))

    # Ask the summarizing agent to process all browsing steps and produce a final summary string.
    summary_response = await summarize_agent.on_messages(web_answer, CancellationToken())
    summary = summary_response.chat_message.content
    logger.debug("summarize answer: %s", summary)
    return summary

# --------------------------------------------------------------------------
# web_surf
#
# A synchronous wrapper around the async_web_surf function. The docstring
# is sent as the description of the tool to the top-level agent.
# --------------------------------------------------------------------------
def web_surf(instructions: str) -> str:
    """
    Given a plain-text request, use a web browser to perform actions and provide a
    text response. Instructions should include any actions that need to be taken.
    This can interact directly with website controls.
    """
    return asyncio.run(async_web_surf(instructions))

# --------------------------------------------------------------------------
# create_agent
#
# Called by the server when an incoming request selects "web_surfer" as the agent model.
# --------------------------------------------------------------------------
def create_agent(user_message: str, context: ChatCompletionContext) -> ChatCompletionClient:
    # "### Task:" is used by some clients (like Open WebUI) to request a quick summary (title).
    # We don't want to use the surfing tool in that case, to avoid unnecessary overhead.
    model_client = get_default_model()
    if user_message.startswith("### Task:"):
        return AssistantAgent(
            name="task_handler",
            model_client=model_client,
            model_client_stream=True,
            model_context=context
        )

    # For standard usage, this returns a web surfer agent that can call the `web_surf` function
    # to handle interactive browsing. The system_message instructs the agent to describe problems
    # and include web links in the final output. reflect_on_tool_use=True instructs the agent
    # to do a second pass after tool usage, refining its final text response.
    system_message = (
        "Use tools to assist the user. Provide detailed information about "
        "any problems you encountered using the tools. Provide as much "
        "information as you can related to the user's queries and include "
        "links to any web browsing results."
    )
    return AssistantAgent(
        name=NAME,
        model_client=model_client,
        model_client_stream=True,
        model_context=context,
        tools=[web_surf],
        reflect_on_tool_use=True,
        system_message=system_message
    )

# Register this agent in the global agent registry, so the server can discover and use it.
add_agent(NAME, create_agent)
