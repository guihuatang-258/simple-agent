"""Deterministic branch recommendation, marketing, and handoff tools."""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain.tools import tool


_DATA_DIR = Path(__file__).resolve().parent / "data"
_BRANCHES_PATH = _DATA_DIR / "branches.json"
_MARKETING_PATH = _DATA_DIR / "marketing_policy.json"
_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise RuntimeError(f"数据文件格式错误: {path.name}")
    return data


@lru_cache(maxsize=1)
def load_branch_catalog() -> dict[str, Any]:
    """Load and minimally validate the POC branch catalog."""

    catalog = _load_json(_BRANCHES_PATH)
    branches = catalog.get("branches")
    if not isinstance(branches, list) or not branches:
        raise RuntimeError("branches.json 必须包含至少一个网点。")
    required = {
        "id",
        "name",
        "full_address",
        "short_address",
        "longitude",
        "latitude",
        "is_24_hours",
        "business_hours",
    }
    for branch in branches:
        missing = required - set(branch)
        if missing:
            raise RuntimeError(
                f"网点 {branch.get('id', '<unknown>')} 缺少字段: "
                + ", ".join(sorted(missing))
            )
    return catalog


@lru_cache(maxsize=1)
def load_marketing_policy() -> dict[str, Any]:
    return _load_json(_MARKETING_PATH)


def _parse_clock(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"无效营业时间: {value!r}") from exc


def _query_datetime(value: str, timezone_name: str = "Asia/Shanghai") -> datetime:
    timezone = ZoneInfo(timezone_name)
    if not value.strip():
        return datetime.now(timezone)
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(
            "current_time 必须是 ISO 8601 格式，例如 2026-09-07T10:30:00+08:00。"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone)
    return parsed.astimezone(timezone)


def _intervals_for_date(branch: dict[str, Any], day: date) -> list[dict[str, str]]:
    hours = branch["business_hours"]
    special = hours.get("special_hours", {})
    if day.isoformat() in special:
        return special[day.isoformat()] or []
    return hours.get("weekly", {}).get(_WEEKDAYS[day.weekday()], [])


def _is_open(branch: dict[str, Any], at: datetime) -> bool:
    if branch.get("is_24_hours"):
        return True

    timezone = ZoneInfo(branch["business_hours"].get("timezone", "Asia/Shanghai"))
    local = at.astimezone(timezone)
    clock = local.timetz().replace(tzinfo=None)

    for interval in _intervals_for_date(branch, local.date()):
        opens = _parse_clock(interval["open"])
        closes = _parse_clock(interval["close"])
        if opens == closes or (opens < closes and opens <= clock < closes):
            return True
        if opens > closes and clock >= opens:
            return True

    previous_day = local.date() - timedelta(days=1)
    for interval in _intervals_for_date(branch, previous_day):
        opens = _parse_clock(interval["open"])
        closes = _parse_clock(interval["close"])
        if opens > closes and clock < closes:
            return True
    return False


def _distance_km(
    longitude: float,
    latitude: float,
    branch_longitude: float,
    branch_latitude: float,
) -> float:
    """Calculate approximate great-circle distance between two GCJ-02 points."""

    radius_km = 6371.0088
    lat1, lat2 = math.radians(latitude), math.radians(branch_latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(branch_longitude - longitude)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _public_branch(branch: dict[str, Any], distance: float, is_open: bool) -> dict[str, Any]:
    return {
        "id": branch["id"],
        "name": branch["name"],
        "short_address": branch["short_address"],
        "full_address": branch["full_address"],
        "phone": branch.get("phone"),
        "phone_status": "configured" if branch.get("phone") else "not_configured",
        "business_hours": "24小时" if branch.get("is_24_hours") else _display_hours(branch),
        "is_24_hours": bool(branch.get("is_24_hours")),
        "is_open": is_open,
        "distance_km": round(distance, 2),
    }


def _display_hours(branch: dict[str, Any]) -> str:
    weekly = branch["business_hours"].get("weekly", {})
    schedules = list(weekly.values())
    if schedules and all(schedule == schedules[0] for schedule in schedules):
        intervals = schedules[0]
        return ", ".join(
            f"{item['open']}-{item['close']}" for item in intervals
        ) or "休息"
    return "营业时间按星期变化，请以结构化时段为准"


def recommend_nearby_branch_data(
    longitude: float,
    latitude: float,
    current_time: str = "",
) -> dict[str, Any]:
    """Pure implementation used by the LangChain tool and unit tests."""

    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        return {
            "status": "invalid_location",
            "message": "经纬度超出有效范围，请重新解析用户地址。",
        }

    branches = load_branch_catalog()["branches"]
    query_at = _query_datetime(current_time)
    ranked = []
    for branch in branches:
        distance = _distance_km(
            longitude,
            latitude,
            float(branch["longitude"]),
            float(branch["latitude"]),
        )
        ranked.append(
            (
                distance,
                branch,
                _is_open(branch, query_at),
            )
        )
    ranked.sort(key=lambda item: item[0])

    nearest_distance, nearest_branch, nearest_open = ranked[0]
    nearest = _public_branch(nearest_branch, nearest_distance, nearest_open)
    recommendation = nearest if nearest_open else None
    reason = "nearest_open" if nearest_open else "no_open_branch"

    if not nearest_open:
        fallback_24h = next(
            (item for item in ranked[1:] if item[1].get("is_24_hours") and item[2]),
            None,
        )
        fallback_open = next((item for item in ranked[1:] if item[2]), None)
        selected = fallback_24h or fallback_open
        if selected:
            recommendation = _public_branch(selected[1], selected[0], selected[2])
            reason = "nearest_24h_fallback" if fallback_24h else "nearest_open_fallback"

    if reason == "nearest_open":
        message = "已找到距离最近且当前营业的网点。"
    elif reason == "nearest_24h_fallback":
        message = "距离最近的网点当前未营业，已推荐附近的24小时营业网点。"
    elif reason == "nearest_open_fallback":
        message = "距离最近的网点当前未营业；预设库暂无24小时网点，已推荐其他当前营业网点。"
    else:
        message = "距离最近的网点当前未营业，预设库内暂无当前营业或24小时网点。"

    return {
        "status": "ok",
        "query_time": query_at.isoformat(timespec="seconds"),
        "message": message,
        "recommendation_reason": reason,
        "nearest_branch": nearest,
        "recommended_branch": recommendation,
        "all_candidates": [
            _public_branch(branch, distance, is_open)
            for distance, branch, is_open in ranked
        ],
    }


@tool
def recommend_nearby_branch(
    longitude: float,
    latitude: float,
    current_time: str = "",
) -> str:
    """根据用户地址解析出的GCJ-02经纬度，从预设网点库中确定性计算最近网点、营业状态和营业兜底推荐。longitude和latitude必须来自地图地址解析结果，不能猜测。current_time留空表示使用当前北京时间；测试指定时间时传ISO 8601格式。"""

    try:
        result = recommend_nearby_branch_data(longitude, latitude, current_time)
    except (OSError, RuntimeError, ValueError) as exc:
        result = {"status": "error", "message": str(exc)}
    return json.dumps(result, ensure_ascii=False)


def get_compliant_marketing_message_data(
    scenario: str = "branch_query",
) -> dict[str, Any]:
    """Return the configured marketing decision without tool serialization."""

    policy = load_marketing_policy()
    configured = policy.get("scenarios", {}).get(scenario)
    if not configured:
        return {
            "status": "unsupported_scenario",
            "message": "",
            "should_speak": False,
        }
    should_speak = bool(configured.get("enabled")) and bool(
        policy.get("peak_season")
    )
    return {
        "status": "ok",
        "peak_season": bool(policy.get("peak_season")),
        "should_speak": should_speak,
        "message": configured.get("message", "") if should_speak else "",
        "must_use_verbatim": bool(
            policy.get("compliance", {}).get("must_use_verbatim")
        ),
        "compliance_rules": policy.get("compliance", {}).get("rules", []),
    }


@tool
def get_compliant_marketing_message(scenario: str = "branch_query") -> str:
    """网点查询答复完成后调用，返回配置中已审核的营销话术和合规要求。只能原样使用message，不得自行添加具体车型库存紧张、价格、优惠或预订承诺。"""

    return json.dumps(
        get_compliant_marketing_message_data(scenario), ensure_ascii=False
    )


@tool
def request_human_handoff(user_confirmed: bool, reason: str = "用户请求转人工") -> str:
    """处理转人工请求。只有用户在当前对话中明确同意转人工时，user_confirmed才能为true；否则必须传false并先向用户确认。当前工具生成POC转接事件，尚未连接真实坐席平台。"""

    if not user_confirmed:
        return json.dumps(
            {
                "status": "confirmation_required",
                "question": "需要我为您转接人工客服吗？",
            },
            ensure_ascii=False,
        )
    safe_reason = reason.strip()[:200] or "用户请求转人工"
    return json.dumps(
        {
            "status": "handoff_requested",
            "mode": "poc_mock",
            "request_id": uuid.uuid4().hex,
            "reason": safe_reason,
            "message": "转人工请求已生成；当前为POC模拟，尚未连接真实坐席平台。",
        },
        ensure_ascii=False,
    )
