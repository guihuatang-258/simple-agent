"""Pure chat agent without tools, for latency comparison."""

from __future__ import annotations

import sys

from agent import build_chat_agent
from main import chat

if __name__ == "__main__":
    oneshot = " ".join(sys.argv[1:]).strip() or None
    chat(
        oneshot,
        builder=build_chat_agent,
        title="纯聊天 Agent 已启动（无工具）。输入问题开始对话，exit / quit 退出。",
        enable_amap_mcp=False,
    )
