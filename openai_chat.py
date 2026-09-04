"""OpenAI-compatible chat model (DashScope compatible-mode, DeepSeek, etc.)."""

from __future__ import annotations

import os

import httpx
from langchain_openai import ChatOpenAI

_http_client: httpx.Client | None = None


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class ChatOpenAICompat(ChatOpenAI):
    """Keep Qwen/DeepSeek reasoning_content that stock ChatOpenAI drops."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None
        choices = chunk.get("choices") or chunk.get(
            "chunk", {}).get("choices") or []
        if not choices:
            return generation_chunk
        delta = choices[0].get("delta") or {}
        reasoning = ""
        if isinstance(delta, dict):
            reasoning = delta.get(
                "reasoning_content") or delta.get("reasoning") or ""
        if reasoning:
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk


def _shared_http_client(base_url: str | None) -> httpx.Client:
    """返回进程级共享客户端，复用 TCP/TLS 连接以降低后续请求延迟。"""

    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=60,
            # 保留空闲连接供多轮对话复用；连接上限也为未来并发请求留出余量。
            limits=httpx.Limits(
                max_keepalive_connections=10, max_connections=20),
        )
        if base_url:
            try:
                # 提前完成 DNS、TCP 和 TLS 握手。即使基础地址返回 404，底层连接
                # 通常仍可进入连接池，从而避免首个正式模型请求承担建连成本。
                _http_client.get(base_url, timeout=3)
            except Exception:
                pass
    return _http_client


def build_openai_model() -> ChatOpenAICompat:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY / DASHSCOPE_API_KEY。")
    base_url = os.getenv("OPENAI_BASE_URL")
    return ChatOpenAICompat(
        model=os.getenv("MODEL_NAME", "qwen3.7-flash"),
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.3")),
        # 默认关闭思考模式，减少首个正文 token 前的推理等待和额外输出 token。
        extra_body={"enable_thinking": _env_flag("ENABLE_THINKING", "false")},
        use_responses_api=False,
        # 显式注入共享客户端，否则每个模型实例可能各自维护连接池。
        # http_client仅限同步调用，异步调用请使用http_async_client
        http_client=_shared_http_client(base_url),
    )
