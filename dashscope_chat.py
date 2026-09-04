"""Native DashScope Generation chat model (same path as Dify's Tongyi plugin)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from typing import Any

import requests
from dashscope import Generation, MultiModalConversation
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

DEFAULT_HTTP_BASE = "https://dashscope.aliyuncs.com/api/v1"
INTL_HTTP_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"

# 这些型号在 DashScope 原生口必须走 multimodal-generation，否则会 url error。
_MULTIMODAL_MARKERS = (
    "-vl",
    "omni",
    "qwen3.5-",
    "qwen3.7-plus",
    "qwen3.7-flash",
    "qwen3.8-",
    "qwen-vl",
)


def is_multimodal_model(model_name: str) -> bool:
    name = model_name.lower()
    if "qwen3.7-max" in name:
        return False
    return any(marker in name for marker in _MULTIMODAL_MARKERS)


def _as_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _as_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_dict(v) for v in value]
    if hasattr(value, "model_dump"):
        return _as_dict(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            k: _as_dict(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return value


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _wrap_content(text: str, *, rich: bool) -> Any:
    if rich:
        return [{"text": text or " "}]
    return text


def _message_to_dict(message: BaseMessage, *, rich: bool = False) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": _wrap_content(_text(message.content), rich=rich)}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": _wrap_content(_text(message.content), rich=rich)}
    if isinstance(message, ToolMessage):
        payload = {
            "role": "tool",
            "content": _text(message.content),
            "tool_call_id": message.tool_call_id,
        }
        if message.name:
            payload["name"] = message.name
        return payload
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": _wrap_content(_text(message.content), rich=rich),
        }
        raw_calls = message.additional_kwargs.get("tool_calls")
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["args"]
                        if isinstance(call["args"], str)
                        else json.dumps(call["args"], ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        elif raw_calls:
            payload["tool_calls"] = raw_calls
        return payload
    return {"role": "user", "content": _wrap_content(_text(message.content), rich=rich)}


def _delta_text(current: str, previous: str) -> str:
    if previous and current.startswith(previous):
        return current[len(previous) :]
    return current


class ChatDashScope(BaseChatModel):
    """DashScope native Generation API: /api/v1/services/aigc/text-generation/generation."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    model_name: str = Field(default="qwen-plus", alias="model")
    api_key: str
    temperature: float = 0.3
    enable_thinking: bool = False
    base_address: str = DEFAULT_HTTP_BASE
    max_retries: int = 1
    _session: requests.Session = PrivateAttr(default_factory=requests.Session)

    def model_post_init(self, __context: Any) -> None:
        # Session 会为后续模型请求复用 TCP/TLS 连接。这里先访问一次基础地址，
        # 提前完成 DNS、TCP 和 TLS 握手，避免首句把冷连接耗时也算进去。
        # 基础地址即使返回非 2xx，也不影响已经建立的底层连接被复用。
        try:
            self._session.get(self.base_address, timeout=3)
        except Exception:
            pass

    @property
    def _llm_type(self) -> str:
        return "dashscope-generation"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_address": self.base_address,
            "enable_thinking": self.enable_thinking,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        kwargs.pop("strict", None)
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        resp = self._call_generation(messages, stream=False, stop=stop, **kwargs)
        message = self._message_from_choice(self._choice_message(resp), is_chunk=False)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # 让服务端直接返回增量 token，收到后即可向 LangChain 上游转发；相比
        # 每个分片返回累计全文，可避免本地反复做字符串差分和传输重复内容。
        incremental = True
        stream = self._call_generation(
            messages,
            stream=True,
            stop=stop,
            incremental_output=incremental,
            **kwargs,
        )
        prev_text = ""
        prev_reason = ""
        for raw in stream:
            resp = self._check(raw)
            msg = self._choice_message(resp)
            text = _text(msg.get("content"))
            reason = msg.get("reasoning_content") or ""
            if not incremental:
                text, prev_text = _delta_text(text, prev_text), text
                reason, prev_reason = _delta_text(reason, prev_reason), reason
            extra: dict[str, Any] = {}
            if reason:
                extra["reasoning_content"] = reason
            tool_chunks = []
            for index, call in enumerate(msg.get("tool_calls") or []):
                fn = call.get("function") or {}
                extra.setdefault("tool_calls", msg.get("tool_calls"))
                tool_chunks.append(
                    tool_call_chunk(
                        name=fn.get("name"),
                        args=fn.get("arguments"),
                        id=call.get("id"),
                        index=call.get("index", index),
                    )
                )
            if not text and not reason and not tool_chunks:
                continue
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=text,
                    additional_kwargs=extra,
                    tool_call_chunks=tool_chunks,
                )
            )
            if run_manager:
                # 立即触发 token 回调，使 CLI 能边生成边显示，降低感知延迟。
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _call_generation(
        self,
        messages: list[BaseMessage],
        *,
        stream: bool,
        stop: list[str] | None,
        incremental_output: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        multimodal = is_multimodal_model(self.model_name)
        params: dict[str, Any] = {
            "model": self.model_name,
            "api_key": self.api_key,
            "messages": [_message_to_dict(m, rich=multimodal) for m in messages],
            "stream": stream,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "base_address": self.base_address,
            "session": self._session,
        }
        if not multimodal:
            params["result_format"] = "message"
        if stop:
            params["stop"] = stop
        if incremental_output is not None:
            params["incremental_output"] = incremental_output
        for key in ("tools", "tool_choice"):
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        # 千问优先使用 DashScope 原生调用链；同时按模型类型选择正确端点，
        # 避免先请求错误端点再回退所造成的额外往返。
        client = MultiModalConversation if multimodal else Generation
        response = client.call(**params)
        if stream:
            return response
        return self._check(response)

    def _check(self, resp: Any) -> dict[str, Any]:
        data = _as_dict(resp)
        status = data.get("status_code")
        if status not in (None, HTTPStatus.OK):
            raise RuntimeError(
                f"DashScope {status} {data.get('code')}: {data.get('message')}"
            )
        return data

    def _choice_message(self, resp: dict[str, Any]) -> dict[str, Any]:
        choices = (resp.get("output") or {}).get("choices") or []
        if not choices:
            return {}
        return choices[0].get("message") or {}

    def _message_from_choice(self, msg: dict[str, Any], *, is_chunk: bool) -> AIMessage:
        extra: dict[str, Any] = {}
        reason = msg.get("reasoning_content")
        if reason:
            extra["reasoning_content"] = reason
        raw_calls = msg.get("tool_calls") or []
        if raw_calls:
            extra["tool_calls"] = raw_calls
        tool_calls = []
        for call in raw_calls:
            fn = call.get("function") or {}
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(
                {
                    "name": fn.get("name") or "",
                    "args": args,
                    "id": call.get("id") or "",
                    "type": "tool_call",
                }
            )
        if is_chunk:
            return AIMessageChunk(
                content=_text(msg.get("content")),
                additional_kwargs=extra,
            )
        return AIMessage(
            content=_text(msg.get("content")),
            additional_kwargs=extra,
            tool_calls=tool_calls,
        )


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def build_dashscope_model() -> ChatDashScope:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 DASHSCOPE_API_KEY / OPENAI_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
        )
    base = os.getenv("DASHSCOPE_HTTP_BASE_URL")
    if not base:
        base = INTL_HTTP_BASE if _env_flag("USE_INTERNATIONAL_ENDPOINT") else DEFAULT_HTTP_BASE
    return ChatDashScope(
        model=os.getenv("MODEL_NAME", "qwen-plus"),
        api_key=api_key,
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.3")),
        enable_thinking=_env_flag("ENABLE_THINKING", "false"),
        base_address=base,
    )
