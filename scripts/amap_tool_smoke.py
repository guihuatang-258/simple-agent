"""Command-line smoke tests for the selected AMap MCP tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool

from components.maps.mcp_tools import load_amap_store_tools


_TOOL_NAMES = ("maps_geo", "maps_around_search", "maps_search_detail")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分别或串联测试项目当前使用的高德 MCP 工具。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("schema", help="打印三个工具的描述和参数 schema。")

    geo = subparsers.add_parser("geo", help="测试地址解析 maps_geo。")
    geo.add_argument("--address", required=True, help="详细地址或地标。")
    geo.add_argument("--city", default="", help="可选城市提示，例如天津。")

    around = subparsers.add_parser(
        "around", help="测试周边搜索 maps_around_search。"
    )
    around.add_argument(
        "--location", required=True, help="中心点，经度和纬度用逗号分隔。"
    )
    around.add_argument("--keywords", default="", help="可选搜索关键词。")
    around.add_argument("--radius", default="1000", help="搜索半径，默认1000米。")

    detail = subparsers.add_parser(
        "detail", help="测试 POI 详情 maps_search_detail。")
    detail.add_argument("--id", required=True, help="周边搜索返回的 POI ID。")

    all_tools = subparsers.add_parser("all", help="串联测试地址解析、周边搜索和详情。")
    all_tools.add_argument("--address", required=True, help="详细地址或地标。")
    all_tools.add_argument("--city", default="", help="可选城市提示，例如天津。")
    all_tools.add_argument("--keywords", default="停车场", help="周边搜索关键词。")
    all_tools.add_argument("--radius", default="1000", help="搜索半径，默认1000米。")
    return parser


def _result_text(result: Any) -> str:
    """Extract text from the content blocks returned by MCP adapter tools."""

    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def _result_payload(result: Any) -> dict[str, Any]:
    """Parse a JSON object from an MCP result for chained smoke tests."""

    text = _result_text(result).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"工具未返回 JSON 对象：{text[:300]}")
    try:
        payload = json.loads(text[start: end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"工具返回的 JSON 无法解析：{text[:300]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("工具返回的 JSON 顶层不是对象。")
    return payload


def _tool_schema(tool: BaseTool) -> dict[str, Any]:
    schema = tool.args_schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return schema if isinstance(schema, dict) else {}


async def _invoke_tool(tool: BaseTool, arguments: dict[str, str]) -> Any:
    print(f"\n[tool] {tool.name}")
    print("[input] " + json.dumps(arguments, ensure_ascii=False))
    started = time.perf_counter()
    result = await tool.ainvoke(arguments)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"[elapsed] {elapsed_ms:.0f} ms")
    try:
        display = json.dumps(_result_payload(
            result), ensure_ascii=False, indent=2)
    except RuntimeError:
        display = _result_text(result)
    print("[result]")
    print(display)
    return result


def _geo_arguments(args: argparse.Namespace) -> dict[str, str]:
    arguments = {"address": args.address}
    if args.city:
        arguments["city"] = args.city
    return arguments


def _location_from_geo(result: Any) -> str:
    payload = _result_payload(result)
    candidates = payload.get("return") or payload.get("geocodes") or []
    if not isinstance(candidates, list):
        raise RuntimeError("maps_geo 返回结果中没有坐标列表。")
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("location"):
            return str(candidate["location"])
    raise RuntimeError("maps_geo 没有返回可用于周边搜索的 location。")


def _first_poi_id(result: Any) -> str:
    pois = _result_payload(result).get("pois") or []
    if not isinstance(pois, list):
        raise RuntimeError("maps_around_search 返回结果中没有 POI 列表。")
    for poi in pois:
        if isinstance(poi, dict) and poi.get("id"):
            return str(poi["id"])
    raise RuntimeError("maps_around_search 没有返回可查询详情的 POI ID。")


async def _run(args: argparse.Namespace) -> int:
    async with load_amap_store_tools() as loaded:
        tools = {tool.name: tool for tool in loaded}
        if not tools:
            raise RuntimeError("高德工具未加载，请检查 .env 中的 AMAP_MAPS_API_KEY。")
        missing = set(_TOOL_NAMES) - set(tools)
        if missing:
            raise RuntimeError("高德 MCP 缺少工具：" + ", ".join(sorted(missing)))

        if args.command == "schema":
            for name in _TOOL_NAMES:
                tool = tools[name]
                print(f"\n=== {name} ===")
                print(tool.description)
                print(json.dumps(_tool_schema(tool), ensure_ascii=False, indent=2))
            return 0

        if args.command == "geo":
            await _invoke_tool(tools["maps_geo"], _geo_arguments(args))
            return 0

        if args.command == "around":
            await _invoke_tool(
                tools["maps_around_search"],
                {
                    "location": args.location,
                    "keywords": args.keywords,
                    "radius": args.radius,
                },
            )
            return 0

        if args.command == "detail":
            await _invoke_tool(tools["maps_search_detail"], {"id": args.id})
            return 0

        geo_result = await _invoke_tool(tools["maps_geo"], _geo_arguments(args))
        location = _location_from_geo(geo_result)
        around_result = await _invoke_tool(
            tools["maps_around_search"],
            {
                "location": location,
                "keywords": args.keywords,
                "radius": args.radius,
            },
        )
        poi_id = _first_poi_id(around_result)
        await _invoke_tool(tools["maps_search_detail"], {"id": poi_id})
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(_PROJECT_ROOT / ".env")
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n测试已中止。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"高德工具测试失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
