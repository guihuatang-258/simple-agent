"""Command-line smoke tests for AMap Web Service REST APIs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from components.maps.geo_level import classify_first_geocode, classify_first_poi


_ENDPOINTS = {
    "text": "https://restapi.amap.com/v5/place/text",
    "geo": "https://restapi.amap.com/v3/geocode/geo",
    "around": "https://restapi.amap.com/v5/place/around",
    "detail": "https://restapi.amap.com/v5/place/detail",
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="直接测试高德 Web Service API。"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="单次 HTTP 请求超时秒数，默认15秒。需放在子命令之前。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    text = subparsers.add_parser("text", help="测试 v5 地点关键字搜索。")
    text.add_argument("--keywords", default="", help="单个地点关键字，最长80字符。")
    text.add_argument("--types", default="", help="POI 分类码，多个用 | 分隔。")
    text.add_argument("--region", default="天津市", help="城市名、citycode 或 adcode。")
    text.add_argument(
        "--city-limit", action="store_true", help="严格限制在 region 范围内。"
    )
    text.add_argument("--page-size", type=int, default=3, help="每页记录数。")
    text.add_argument("--page-num", type=int, default=None, help="页码。")
    text.add_argument("--show-fields", default="", help="需要额外返回的字段。")
    text.add_argument(
        "--geocode-fallback",
        action="store_true",
        help="typecode 无法判断时，再调用 geocode/geo 获取官方 level。",
    )

    geo = subparsers.add_parser("geo", help="测试 v3 地址地理编码。")
    geo.add_argument("--address", required=True, help="详细地址或地标。")
    geo.add_argument("--city", default="", help="可选城市提示，例如天津。")

    around = subparsers.add_parser("around", help="测试 v5 周边搜索。")
    around.add_argument(
        "--location", required=True, help="中心点，经度和纬度用逗号分隔。"
    )
    around.add_argument("--keywords", default="", help="单个搜索关键字。")
    around.add_argument("--types", default="", help="POI 分类码，多个用 | 分隔。")
    around.add_argument("--radius", default="1000", help="搜索半径，默认1000米。")
    around.add_argument("--region", default="", help="城市名、citycode 或 adcode。")
    around.add_argument("--page-size", type=int, default=3, help="每页记录数。")
    around.add_argument("--page-num", type=int, default=None, help="页码。")
    around.add_argument("--show-fields", default="", help="需要额外返回的字段。")

    detail = subparsers.add_parser("detail", help="测试 v5 POI 详情查询。")
    detail.add_argument("--id", required=True, help="搜索接口返回的 POI ID。")
    detail.add_argument("--show-fields", default="", help="需要额外返回的字段。")

    all_tools = subparsers.add_parser(
        "all", help="串联测试地理编码、周边搜索和 POI 详情。"
    )
    all_tools.add_argument("--address", required=True, help="详细地址或地标。")
    all_tools.add_argument("--city", default="", help="可选城市提示，例如天津。")
    all_tools.add_argument("--keywords", default="停车场", help="周边搜索关键字。")
    all_tools.add_argument("--types", default="", help="POI 分类码，多个用 | 分隔。")
    all_tools.add_argument("--radius", default="1000", help="搜索半径，默认1000米。")
    all_tools.add_argument(
        "--show-fields", default="business", help="详情额外字段，默认 business。"
    )
    return parser


def _api_key() -> str:
    key = os.getenv("AMAP_MAPS_API_KEY") or os.getenv("AMAP_API_KEY") or ""
    key = key.strip()
    if not key:
        raise RuntimeError("请在 .env 中配置 AMAP_MAPS_API_KEY。")
    return key


def _compact_params(**values: Any) -> dict[str, str]:
    """Drop unset optional arguments and serialize values for requests."""

    params = {}
    for name, value in values.items():
        if value is None or value == "" or value is False:
            continue
        params[name] = str(value).lower() if isinstance(
            value, bool) else str(value)
    return params


def _validate_text_search(params: dict[str, str]) -> None:
    if not params.get("keywords") and not params.get("types"):
        raise ValueError("地点搜索的 keywords 和 types 至少需要提供一个。")
    if len(params.get("keywords", "")) > 80:
        raise ValueError("keywords 总长度不能超过80字符。")


def _request(
    session: requests.Session,
    endpoint: str,
    params: dict[str, str],
    *,
    key: str,
    timeout: float,
) -> dict[str, Any]:
    """Call AMap without ever including the key in logs or raised errors."""

    print(f"\n[request] GET {endpoint}")
    print("[params] " + json.dumps(params, ensure_ascii=False))
    started = time.perf_counter()
    try:
        response = session.get(
            endpoint,
            params={**params, "key": key},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP 请求失败：{type(exc).__name__}") from None
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"[http] {response.status_code}  [elapsed] {elapsed_ms:.0f} ms")

    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"高德接口返回 HTTP {response.status_code}。")
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError("高德接口没有返回合法 JSON。") from None
    if not isinstance(payload, dict):
        raise RuntimeError("高德接口返回的 JSON 顶层不是对象。")

    print("[result]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if str(payload.get("status")) != "1":
        info = payload.get("info") or "unknown error"
        infocode = payload.get("infocode") or "unknown"
        raise RuntimeError(f"高德业务请求失败：{info}（{infocode}）")
    return payload


def _text_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _first_poi(payload: dict[str, Any]) -> dict[str, Any]:
    pois = payload.get("pois") or []
    if isinstance(pois, list):
        for poi in pois:
            if isinstance(poi, dict):
                return poi
    raise RuntimeError("地点搜索没有返回可用于地理编码的 POI。")


def _detailed_address_from_first_poi(
    payload: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Build one geocodable address from the first place/text POI."""

    poi = _first_poi(payload)
    parts = []
    for field in ("pname", "cityname", "adname", "address", "name"):
        value = _text_value(poi.get(field))
        if value and value not in parts and value not in "".join(parts):
            parts.append(value)
    address = "".join(parts)
    if not address:
        raise RuntimeError("第一条 POI 缺少可用于地理编码的详细地址。")
    city = _text_value(poi.get("cityname")) or _text_value(poi.get("pname"))
    return address, city, poi


def _print_geocode_level(payload: dict[str, Any]) -> None:
    level = classify_first_geocode(payload)
    if level is None:
        raise RuntimeError("geocode/geo 没有返回可判断的 geocodes[0]。")
    print("[geo-level]")
    print(json.dumps(level, ensure_ascii=False, indent=2))


def _print_poi_level(payload: dict[str, Any], query: str = "") -> dict[str, Any]:
    level = classify_first_poi(payload, query=query)
    if level is None:
        raise RuntimeError("place/text 没有返回可判断的 pois[0]。")
    print("[poi-level]")
    print(json.dumps(level, ensure_ascii=False, indent=2))
    return level


def _location_from_geo(payload: dict[str, Any]) -> str:
    geocodes = payload.get("geocodes") or []
    if isinstance(geocodes, list):
        for item in geocodes:
            if isinstance(item, dict) and item.get("location"):
                return str(item["location"])
    raise RuntimeError("地理编码结果中没有 location。")


def _first_poi_id(payload: dict[str, Any]) -> str:
    pois = payload.get("pois") or []
    if isinstance(pois, list):
        for poi in pois:
            if isinstance(poi, dict) and poi.get("id"):
                return str(poi["id"])
    raise RuntimeError("搜索结果中没有可查询详情的 POI ID。")


def _run(args: argparse.Namespace) -> int:
    key = _api_key()
    with requests.Session() as session:
        if args.command == "text":
            params = _compact_params(
                keywords=args.keywords,
                types=args.types,
                region=args.region,
                city_limit=args.city_limit,
                page_size=args.page_size,
                page_num=args.page_num,
                show_fields=args.show_fields,
            )
            _validate_text_search(params)
            search_result = _request(
                session, _ENDPOINTS["text"], params, key=key, timeout=args.timeout
            )
            poi = _first_poi(search_result)
            print("[selected-place]")
            print(
                json.dumps(
                    {
                        "id": poi.get("id"),
                        "name": poi.get("name"),
                        "address": poi.get("address"),
                        "location": poi.get("location"),
                        "typecode": poi.get("typecode"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            level = _print_poi_level(search_result, query=args.keywords)
            if args.geocode_fallback and level["level_code"] == "unknown":
                address, city, _ = _detailed_address_from_first_poi(
                    search_result)
                print("[fallback] typecode 无法确定层级，调用 geocode/geo。")
                geocode_result = _request(
                    session,
                    _ENDPOINTS["geo"],
                    _compact_params(address=address, city=city or args.region),
                    key=key,
                    timeout=args.timeout,
                )
                _print_geocode_level(geocode_result)
            return 0

        if args.command == "geo":
            params = _compact_params(address=args.address, city=args.city)
            result = _request(
                session, _ENDPOINTS["geo"], params, key=key, timeout=args.timeout
            )
            _print_geocode_level(result)
            return 0

        if args.command == "around":
            params = _compact_params(
                location=args.location,
                keywords=args.keywords,
                types=args.types,
                radius=args.radius,
                region=args.region,
                page_size=args.page_size,
                page_num=args.page_num,
                show_fields=args.show_fields,
            )
            _request(session, _ENDPOINTS["around"],
                     params, key=key, timeout=args.timeout)
            return 0

        if args.command == "detail":
            params = _compact_params(id=args.id, show_fields=args.show_fields)
            _request(session, _ENDPOINTS["detail"],
                     params, key=key, timeout=args.timeout)
            return 0

        geo = _request(
            session,
            _ENDPOINTS["geo"],
            _compact_params(address=args.address, city=args.city),
            key=key,
            timeout=args.timeout,
        )
        _print_geocode_level(geo)
        location = _location_from_geo(geo)
        around_params = _compact_params(
            location=location,
            keywords=args.keywords,
            types=args.types,
            radius=args.radius,
        )
        around = _request(
            session,
            _ENDPOINTS["around"],
            around_params,
            key=key,
            timeout=args.timeout,
        )
        poi_id = _first_poi_id(around)
        _request(
            session,
            _ENDPOINTS["detail"],
            _compact_params(id=poi_id, show_fields=args.show_fields),
            key=key,
            timeout=args.timeout,
        )
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(_PROJECT_ROOT / ".env")
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("\n测试已中止。", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"高德 REST API 测试失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
