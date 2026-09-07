"""Assemble a LangChain v1 agent with create_agent."""

from __future__ import annotations

import os
from collections.abc import Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from dashscope_chat import build_dashscope_model
from openai_chat import build_openai_model
from tools import calculator, get_current_time, get_weather
from mcp_tools import load_amap_store_tools, load_tencent_tools

# 在此增删工具；MCP 加载函数不加括号，未选中的服务不会连接。
DEFAULT_TOOLS = [
    get_current_time,
    # get_weather,
    load_amap_store_tools,
    # load_tencent_tools,
]

SYSTEM_PROMPT = """你是一个简洁、可靠的助手。
用户需要当前时间或位置相关信息是，必须调用对应工具，不要编造数字。
用用户使用的语言回答。
"""

AMAP_MCP_PROMPT = """
查询地址周边门店时，先调用 maps_geo 将用户地址转换为经纬度，再调用
maps_around_search 搜索品牌或门店类型；缺少详细地址或搜索关键词时先询问用户。
"""

TENCENT_MCP_PROMPT = """
以 tencent_ 开头的工具来自腾讯位置服务 MCP。用户指定腾讯地图时使用这些工具。
根据各工具的描述及参数 Schema 完成地址解析、地点搜索等任务，不要猜测参数。
"""

_NATIVE_PROVIDERS = {"dashscope", "tongyi", "qwen"}


def _positive_limit(name: str, default: int) -> int:
    """读取正整数限制，启动时尽早暴露错误配置。"""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是正整数，当前值为 {raw!r}。") from exc
    if value < 1:
        raise RuntimeError(f"{name} 必须大于等于 1，当前值为 {value}。")
    return value


def _agent_limits():
    """限制一次用户提问中的工具调用总数和模型循环次数。"""
    return [
        # 初次回答算一次模型调用；每次工具执行完再次进入模型也再算一次。
        ModelCallLimitMiddleware(
            run_limit=_positive_limit("AGENT_MAX_LOOPS", 6),
            exit_behavior="end",
        ),
        # 同一 AIMessage 中并行请求的多个工具分别计数。超过限制的工具不会执行，
        # 但允许模型使用已有结果生成回答，最终仍受模型循环次数保护。
        ToolCallLimitMiddleware(
            run_limit=_positive_limit("AGENT_MAX_TOOL_CALLS", 10),
            exit_behavior="continue",
        ),
    ]


def _build_model():
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider in _NATIVE_PROVIDERS:
        return build_dashscope_model()
    return build_openai_model()


def build_agent(tools: Sequence[BaseTool] = ()):
    """接收 main.py 已经加载好的工具列表。"""
    names = {tool.name for tool in tools}
    map_prompt = AMAP_MCP_PROMPT if "maps_geo" in names else ""
    if any(name.startswith("tencent_") for name in names):
        map_prompt += TENCENT_MCP_PROMPT
    # Agent 只在 CLI 启动时构建一次；交互中的后续轮次会复用同一个模型实例、
    # HTTP 连接池和内存检查点，避免每轮重复初始化带来的额外延迟。
    return create_agent(
        model=_build_model(),
        tools=list(tools),
        # 只有成功加载地图 MCP 工具时才向模型注入对应调用说明。
        system_prompt=SYSTEM_PROMPT + map_prompt,
        middleware=_agent_limits(),
        checkpointer=InMemorySaver(),
        name="simple-agent",
    )
