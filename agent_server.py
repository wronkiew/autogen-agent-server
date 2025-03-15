import time
import uuid
import json
import logging
import traceback
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from autogen_agentchat.base import Response as AgentResponse
from autogen_agentchat.messages import ChatMessage, TextMessage, ModelClientStreamingChunkEvent
from autogen_core import CancellationToken, TRACE_LOGGER_NAME
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage
from contextlib import asynccontextmanager
from registry import get_agent, list_agents, load_agent_files, get_logger
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.WARNING)
    autogen_logger = logging.getLogger(TRACE_LOGGER_NAME)
    autogen_logger.addHandler(logging.StreamHandler())
    autogen_logger.setLevel(logging.INFO)

    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logger = get_logger()
    logger.setLevel(logging.INFO)

    load_agent_files(settings.agent_dir, logger)

    settings.log_config(logger)
    logger.info("Loaded agents:")
    for name in list_agents():
        logger.info(f"  {name}")

    yield

app = FastAPI(lifespan=lifespan)

# Helper for streaming responses (SSE)
async def stream_response(agent, user_message: ChatMessage, model_name: str, 
                          logger: logging.Logger):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    async def sse_stream():
        got_chunk = False
        try:
            async for item in agent.on_messages_stream([user_message], CancellationToken()):
                create_time = int(time.time())
                logger.debug(f"sse_stream item {type(item).__name__}: {item}")
                if isinstance(item, ModelClientStreamingChunkEvent) or isinstance(item, TextMessage):
                    chunk_dict = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": create_time,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": item.content},
                            "finish_reason": None
                        }]
                    }
                    got_chunk = True
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                elif isinstance(item, AgentResponse):
                    final_chunk_dict = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": create_time,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": "" if got_chunk else item.chat_message.content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk_dict)}\n\n"

                    stop_chunk_dict = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": create_time,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(stop_chunk_dict)}\n\n"

                    # Sentinel for completion.
                    yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error(f"Error during streaming response: {type(exc).__name__} {exc}")
            stack_trace = traceback.format_exc()
            logger.debug(stack_trace)
            error_chunk = {"error": "Internal server error (streaming)."}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")

# Helper for non-streaming responses
async def complete_response(agent, user_message: ChatMessage, model_name: str,
                             logger: logging.Logger):
    try:
        # Call the agent's on_messages function to get a full response
        response = await agent.on_messages([user_message], CancellationToken())
    except Exception as exc:
        logger.error("Error during response: %s", exc)
        return {"error": f"Internal server error."}
    created_time = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    prompt_tokens = 0
    completion_tokens = 0
    if (response.chat_message.models_usage):
        models_usage = response.chat_message.models_usage
        prompt_tokens = models_usage.prompt_tokens
        completion_tokens = models_usage.completion_tokens
    final_response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.chat_message.content,
                "refusal": None
            },
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        },
        "system_fingerprint": "fp_06737a9306"
    }
    return JSONResponse(final_response)

async def message_history_to_context(message_history, agent_name):
    context = UnboundedChatCompletionContext()
    if not message_history:
        return context
    role_to_class = {
        "user": UserMessage,
        "assistant": AssistantMessage,
        "system": SystemMessage,
    }
    for msg in message_history:
        message_class = role_to_class.get(msg.get("role"))
        if message_class:
            await context.add_message(message_class(content=msg.get("content"), 
                                                    source=agent_name))
    return context

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, logger: logging.Logger = Depends(get_logger)):
    body = await request.json()
    logger.debug(f"Request: {body}")
    model_name = body.get("model", "")
    stream = body.get("stream", False)

    messages = body.get("messages", [])
    last_user_message = ""
    if messages and messages[-1].get("role") == "user":
        last_message = messages.pop()
        last_user_message = last_message.get("content", "")

    # Lookup the agent constructor using the registry.
    agent_constructor = get_agent(model_name)
    if not agent_constructor:
        logger.error(f"Model '{model_name}' not supported.")
        return {"error": f"Model '{model_name}' not supported."}
    
    context = await message_history_to_context(messages, model_name)
    
    # Construct the agent.
    agent = agent_constructor(last_user_message, context)
    logger.info(f"Request sent to agent {model_name}.")

    user_message = TextMessage(content=last_user_message, source=model_name)
    
    # Process based on whether streaming is enabled.
    if stream:
        return await stream_response(agent, user_message, model_name, logger)
    else:
        return await complete_response(agent, user_message, model_name, logger)

@app.get("/v1/models")
async def list_models(logger: logging.Logger = Depends(get_logger)):
    # Note: the model names sent to the UI are the agent names, not the backend
    # LLM name.
    return {
        "data": [
            {
                "id": name,
                "object": "model",
                "owned_by": "agent"
            }
            for name in list_agents()
        ],
        "object": "list"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_server:app",
                host=settings.server_host, 
                port=settings.server_port, 
                reload=settings.auto_reload,
                reload_dirs=[settings.agent_dir])