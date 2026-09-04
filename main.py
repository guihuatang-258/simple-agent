"""CLI chat loop for the LangChain create_agent demo."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from collections.abc import Callable

from agent import build_agent

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


def _run_turn(agent, user: str, config: dict) -> None:
    payload = {"messages": [{"role": "user", "content": user}]}
    ttft = _FirstTokenTimer()
    printed = False
    thinking = False

    # 同时订阅 token 消息流和节点更新流：正文可以逐 token 展示，工具调用过程
    # 也能立即反馈。流式输出主要降低用户的感知等待，不会缩短完整生成时间。
    for chunk in agent.stream(
        payload,
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        kind, data = _split_stream_chunk(chunk)
        if kind == "messages":
            token = data[0] if isinstance(data, tuple) else data
            metadata = data[1] if isinstance(data, tuple) and len(data) > 1 else {}
            node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
            # tools 节点也会往 messages 流里塞 ToolMessage，不能当成助手正文
            if node == "tools" or getattr(token, "type", "") == "tool":
                continue
            text = getattr(token, "text", "") or ""
            extra = getattr(token, "additional_kwargs", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
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


def chat(
    prompt: str | None = None,
    *,
    builder: Callable = build_agent,
    title: str = "简易 Agent 已启动。输入问题开始对话，exit / quit 退出。",
) -> None:
    # 整个交互会话只构建一次，后续轮次复用模型、连接池与会话记忆。
    agent = builder()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if prompt:
        _run_turn(agent, prompt, config)
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
            _run_turn(agent, user, config)
        except Exception as exc:
            print(f"出错: {exc}", file=sys.stderr)


if __name__ == "__main__":
    oneshot = " ".join(sys.argv[1:]).strip() or None
    chat(oneshot)
