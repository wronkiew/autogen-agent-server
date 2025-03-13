### Project Overview: AutoGen API Server

**Overview**  
AutoGen API Server is a lightweight and customizable server for developing conversational agents using Microsoft [AutoGen 0.4](https://github.com/microsoft/autogen). Built on FastAPI and uvicorn, it offers a straightforward way to test, deploy, and interact with conversational agents. The server integrates easily with backend solutions like OpenAI, LM Studio, or other custom servers.

**Key Features:**
- **Plugin-Based Design:** Extend agent functionalities with a simple plugin system. Example plugins, including a password-game agent, are provided.
- **Flexible Configuration:** Configuration managed through Pydantic and environment variables, supporting easy customization and testing.
- **Streaming and Standard Responses:** Supports both Server-Sent Events (SSE) for streaming and traditional request-response interactions.
- **Stateless Architecture:** Each interaction is independent, simplifying deployment and scalability.
- **Efficient Operation:** FastAPI and uvicorn ensure efficient performance and quick response handling.

### 项目概述：AutoGen API 服务器

**简介**  
AutoGen API 服务器是一款基于 Microsoft [AutoGen 0.4](https://github.com/microsoft/autogen) 的轻量、可定制化对话代理服务器。项目使用 FastAPI 和 uvicorn 构建，提供了一个简单的平台，帮助开发者测试、部署和交互对话式代理程序。支持与后端服务器（如 OpenAI、LM Studio）或自定义方案的便捷集成。

**主要功能：**
- **插件化设计：** 通过插件轻松扩展代理功能。项目包含示例插件，例如密码游戏代理，帮助您快速入门。
- **灵活的配置：** 使用 Pydantic 和环境变量进行配置管理，易于测试和自定义部署。
- **流式和标准响应：** 同时支持流式响应（Server-Sent Events，SSE）和传统的请求-响应交互。
- **无状态架构：** 每个交互独立进行，简化了部署过程和扩展性。
- **高效运行：** 利用 FastAPI 和 uvicorn 提供快速响应和高效性能。

---

### Example agent

This is an example passthrough agent. It connects `AssistantAgent` to the server.
```python
from registry import add_agent, get_default_model
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.model_context import ChatCompletionContext

name = "hello-world"

# Constructor for the 'passthrough' agent. Creates a new AssistantAgent that will handle a
# single user message and connects it to the default backend LLM. The complete conversation
# history is passed in and loaded by the agent.
def create_agent(user_message: str,
                 context: ChatCompletionContext
                 ) -> ChatCompletionClient:
    system_message = ( "You are a helpful assistant." )
    model_client = get_default_model()
    return AssistantAgent(name=name, model_client=model_client, model_client_stream=True,
                          model_context=context, system_message=system_message)

# Register this agent when the module is imported.
add_agent(name, create_agent)
```
