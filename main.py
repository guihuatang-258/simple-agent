"""CLI chat loop for the LangChain create_agent demo."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from collections.abc import Callable

from agent import build_agent
from mcp_tools import load_amap_store_tools

load_dotenv(Path(__file__).resolve().parent / ".env")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class _FirstTokenTimer:
    """测量用户看到第一个正文 token 前的等待时间（不是纯网络耗时）。"""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._done = False

    def mark(self) -> None:
        if self._done:
            return
        self._done = True
        ms = (time.perf_counter() - self._start) * 1000
        print(f"[首token] {ms:.0f} ms", flush=True)


def _print_tool_updates(data: dict) -> None:
    for update in data.values():
        if not isinstance(update, dict):
            continue
        for message in update.get("messages", []):
            kind = getattr(message, "type", "")
            name = getattr(message, "name", "") or ""
            content = getattr(message, "content", "") or ""
            tool_calls = getattr(message, "tool_calls", None) or []
            if kind == "ai" and tool_calls:
                for call in tool_calls:
                    print(f"\n[tool] {call.get('name')}({call.get('args')})")
            elif kind == "tool":
                print(f"[result] {name}: {content}")


def _split_stream_chunk(chunk: object) -> tuple[str, object]:
    if isinstance(chunk, dict) and "type" in chunk:
        return chunk["type"], chunk.get("data")
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return chunk[0], chunk[1]
    return "updates", chunk


async def _run_turn(agent, user: str, config: dict) -> None:
    payload = {"messages": [{"role": "user", "content": user}]}
    ttft = _FirstTokenTimer()
    printed = False
    thinking = False

    # 同时订阅 token 消息流和节点更新流：正文可以逐 token 展示，工具调用过程
    # 也能立即反馈。流式输出主要降低用户的感知等待，不会缩短完整生成时间。
    async for chunk in agent.astream(
        payload,
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        kind, data = _split_stream_chunk(chunk)
        if kind == "messages":
            token = data[0] if isinstance(data, tuple) else data
            metadata = data[1] if isinstance(
                data, tuple) and len(data) > 1 else {}
            node = metadata.get("langgraph_node") if isinstance(
                metadata, dict) else None
            # tools 节点也会往 messages 流里塞 ToolMessage，不能当成助手正文
            if node == "tools" or getattr(token, "type", "") == "tool":
                continue
            text = getattr(token, "text", "") or ""
            extra = getattr(token, "additional_kwargs", None) or {}
            reasoning = extra.get(
                "reasoning_content") or extra.get("reasoning") or ""
            if reasoning:
                if not thinking:
                    print("\n[思考] ", end="", flush=True)
                    thinking = True
                print(reasoning, end="", flush=True)
            if text:
                if not printed:
                    # 只在首个正文 token 到达时计时；若模型先思考或调用工具，
                    # 这些阶段也会计入，因此这是端到端的“可见回答”TTFT。
                    ttft.mark()
                    if thinking:
                        print()
                    print("\n助手: ", end="", flush=True)
                    printed = True
                print(text, end="", flush=True)
        elif kind == "updates" and isinstance(data, dict):
            _print_tool_updates(data)

    if printed or thinking:
        print("\n")


async def _chat_loop(agent, prompt: str | None, title: str) -> None:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if prompt:
        await _run_turn(agent, prompt, config)
        return

    print(f"{title}\n")
    while True:
        try:
            user = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            print("再见。")
            return
        try:
            await _run_turn(agent, user, config)
        except Exception as exc:
            print(f"出错: {exc}", file=sys.stderr)


async def _chat(
    prompt: str | None = None,
    *,
    builder: Callable = build_agent,
    title: str = "简易 Agent 已启动。输入问题开始对话，exit / quit 退出。",
    enable_amap_mcp: bool = True,
) -> None:
    if enable_amap_mcp:
        # MCP 工具及 stdio 子进程的生命周期覆盖整个聊天会话。
        async with load_amap_store_tools() as mcp_tools:
            if not mcp_tools:
                print(
                    "[MCP] 未配置 AMAP_MAPS_API_KEY，已跳过高德地图工具。",
                    file=sys.stderr,
                )
            agent = builder(mcp_tools)
            await _chat_loop(agent, prompt, title)
        return

    # 纯聊天模式不启动 MCP Server，用来测量没有工具开销时的模型延迟。
    await _chat_loop(builder(), prompt, title)


def chat(
    prompt: str | None = None,
    *,
    builder: Callable = build_agent,
    title: str = "简易 Agent 已启动。输入问题开始对话，exit / quit 退出。",
    enable_amap_mcp: bool = True,
) -> None:
    """同步 CLI 入口；内部事件循环用于模型和 MCP 工具的异步流式调用。"""

    asyncio.run(
        _chat(
            prompt,
            builder=builder,
            title=title,
            enable_amap_mcp=enable_amap_mcp,
        )
    )


if __name__ == "__main__":
    # oneshot模式，即命令行直接传入问题，程序会在回答后退出。
    oneshot = " ".join(sys.argv[1:]).strip() or None
    chat(oneshot)
