"""Assemble a LangChain v1 agent with create_agent."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from dashscope_chat import build_dashscope_model
from openai_chat import build_openai_model
from tools import calculator, get_current_time, get_weather

SYSTEM_PROMPT = """你是一个简洁、可靠的助手。
需要精确计算、当前时间或天气时，必须调用对应工具，不要编造数字。
用用户使用的语言回答。
"""

_NATIVE_PROVIDERS = {"dashscope", "tongyi", "qwen"}


def _build_model():
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider in _NATIVE_PROVIDERS:
        return build_dashscope_model()
    return build_openai_model()


def build_agent():
    # Agent 只在 CLI 启动时构建一次；交互中的后续轮次会复用同一个模型实例、
    # HTTP 连接池和内存检查点，避免每轮重复初始化带来的额外延迟。
    return create_agent(
        model=_build_model(),
        tools=[calculator, get_current_time, get_weather],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        name="simple-agent",
    )


def build_chat_agent():
    # 不绑定工具的轻量版本：减少工具 schema 带来的输入 token，便于单独测量
    # 模型自身的首 token 延迟，排除工具调用和额外模型轮次的影响。
    return create_agent(
        model=_build_model(),
        tools=[],
        system_prompt="你是一个简洁、可靠的助手。用用户使用的语言回答。",
        checkpointer=InMemorySaver(),
        name="chat-agent",
    )
