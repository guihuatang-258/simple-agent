"""Load the official AMap MCP tools used for nearby-store searches."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


_AMAP_SERVER_NAME = "amap-maps"
_AMAP_STORE_TOOL_NAMES = {
    "maps_geo",
    "maps_around_search",
    "maps_search_detail",
}


def _amap_api_key() -> str:
    # AMAP_MAPS_API_KEY 是官方 MCP Server 使用的变量名；兼容上一版 REST
    # 工具使用的 AMAP_API_KEY，方便已有本地配置平滑迁移。
    return (
        os.getenv("AMAP_MAPS_API_KEY") or os.getenv("AMAP_API_KEY") or ""
    ).strip()


def _npx_command() -> str:
    # Windows 的可执行入口是 npx.cmd，直接传给 stdio transport 可避免 shell。
    candidates = ("npx.cmd", "npx") if sys.platform == "win32" else ("npx",)
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError("未找到 npx，请先安装 Node.js 22.14 或更高版本。")


# 启动一个异步上下文管理器
@asynccontextmanager
async def load_amap_store_tools() -> AsyncGenerator[list[BaseTool]]:
    """启动高德官方 MCP Server，并在整个聊天期间保持同一个会话。"""

    api_key = _amap_api_key()
    if not api_key:
        # 地图能力是可选项；没有 Key 时不阻断计算、时间、天气或纯聊天。
        yield []
        return

    client = MultiServerMCPClient(
        {
            _AMAP_SERVER_NAME: {
                "transport": "stdio",
                "command": _npx_command(),
                "args": ["-y", "@amap/amap-maps-mcp-server"],
                # Key 只传给本地 MCP 子进程，不拼接到 URL 中。
                "env": {"AMAP_MAPS_API_KEY": api_key},
            }
        }
    )

    # 显式保持 ClientSession；否则适配器默认会在每次工具调用时重新创建 MCP 会话和 stdio 子进程，显著增加连续两步“地址解析→周边搜索”的延迟。
    async with client.session(_AMAP_SERVER_NAME) as session:
        discovered = await load_mcp_tools(session)
        selected = [
            tool for tool in discovered if tool.name in _AMAP_STORE_TOOL_NAMES]
        loaded_names = {tool.name for tool in selected}
        missing = _AMAP_STORE_TOOL_NAMES - loaded_names
        if missing:
            raise RuntimeError(
                "高德 MCP Server 缺少所需工具：" + ", ".join(sorted(missing))
            )
        # 将选中的工具提供给调用者，整个聊天会话期间保持同一个 MCP 子进程。
        # yield保持session存活
        yield selected
