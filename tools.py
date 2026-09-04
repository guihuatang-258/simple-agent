"""Agent tools: calculator, clock, and a demo weather lookup."""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from langchain.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_WEATHER = {
    "beijing": "晴，24°C，东北风 2 级",
    "上海": "多云，27°C，东南风 3 级",
    "shanghai": "多云，27°C，东南风 3 级",
    "北京": "晴，24°C，东北风 2 级",
    "shenzhen": "阵雨，30°C，湿度 80%",
    "深圳": "阵雨，30°C，湿度 80%",
    "hangzhou": "阴，22°C，能见度良好",
    "杭州": "阴，22°C，能见度良好",
    "san francisco": "Sunny, 70°F, light breeze",
    "tokyo": "Clear, 26°C",
    "东京": "Clear, 26°C",
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("只支持数字和 + - * / ** % 运算")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、幂运算和取余，例如 12 * (3 + 5) / 2。"""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_expr(tree))
    except Exception as exc:
        return f"计算失败: {exc}"


_TZ_ALIASES = {
    "cst": "Asia/Shanghai",
    "china": "Asia/Shanghai",
    "cn": "Asia/Shanghai",
    "prc": "Asia/Shanghai",
    "北京": "Asia/Shanghai",
    "上海": "Asia/Shanghai",
    "中国": "Asia/Shanghai",
}


def _resolve_timezone(timezone_name: str):
    raw = (timezone_name or "Asia/Shanghai").strip()
    if not raw or raw.upper() == "UTC":
        return timezone.utc
    key = _TZ_ALIASES.get(raw.lower(), raw)
    try:
        return ZoneInfo(key)
    except Exception:
        if key == "Asia/Shanghai" or raw.lower() in _TZ_ALIASES:
            return timezone(timedelta(hours=8), name="CST")
        return None


@tool
def get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """返回指定时区的当前日期和时间。timezone_name 例如 Asia/Shanghai、UTC、America/Los_Angeles。"""
    tz = _resolve_timezone(timezone_name)
    if tz is None:
        return f"未知时区: {timezone_name}"
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


@tool
def get_weather(city: str) -> str:
    """查询城市天气（演示数据）。city 使用城市名，如 Beijing、上海、Tokyo。"""
    key = city.strip().lower()
    weather = _WEATHER.get(key) or _WEATHER.get(city.strip())
    if weather:
        return f"{city}: {weather}"
    return (
        f"{city}: 暂无演示数据。"
        "可试 Beijing / Shanghai / Shenzhen / Hangzhou / Tokyo / San Francisco。"
    )
