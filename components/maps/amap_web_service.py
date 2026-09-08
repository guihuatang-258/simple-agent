"""Fast in-process client for the AMap place/text Web Service API."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict
from typing import Any

import requests

from .geo_level import classify_first_poi


_PLACE_TEXT_ENDPOINT = "https://restapi.amap.com/v5/place/text"


class AMapWebServiceError(RuntimeError):
    """Public, credential-safe AMap location error."""


def _configured_api_key() -> str:
    return (
        os.getenv("AMAP_MAPS_API_KEY") or os.getenv("AMAP_API_KEY") or ""
    ).strip()


def _positive_timeout(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正数。") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须是正数。")
    return value


def _first_poi(payload: dict[str, Any]) -> dict[str, Any]:
    pois = payload.get("pois") or []
    if isinstance(pois, list):
        for poi in pois:
            if isinstance(poi, dict):
                return poi
    raise AMapWebServiceError("地点搜索没有返回匹配结果。")


def _coordinates(poi: dict[str, Any]) -> tuple[float, float]:
    location = str(poi.get("location") or "").strip()
    parts = location.split(",")
    if len(parts) != 2:
        raise AMapWebServiceError("地点搜索结果缺少有效坐标。")
    try:
        longitude, latitude = (float(part.strip()) for part in parts)
    except ValueError:
        raise AMapWebServiceError("地点搜索结果缺少有效坐标。") from None
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise AMapWebServiceError("地点搜索结果坐标超出有效范围。")
    return longitude, latitude


class AMapWebServiceClient:
    """Resolve addresses with one place/text call and reuse HTTP connections."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        cache_size: int = 128,
        session: requests.Session | None = None,
    ) -> None:
        # Key 只保存在客户端实例中。后续异常不会拼接请求 URL，避免日志泄露 Key。
        self._api_key = _configured_api_key() if api_key is None else api_key.strip()
        # requests 支持 (connect, read) 两段超时，分别控制建连失败和服务端慢响应。
        self._timeout = (
            connect_timeout
            if connect_timeout is not None
            else _positive_timeout("AMAP_CONNECT_TIMEOUT", 3.0),
            read_timeout
            if read_timeout is not None
            else _positive_timeout("AMAP_READ_TIMEOUT", 5.0),
        )
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict[tuple[str, str],
                                 dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._thread_local = threading.local()
        # session 参数仅用于测试或外部注入；生产环境默认使用线程本地连接池。
        self._injected_session = session

    def _session(self) -> requests.Session:
        if self._injected_session is not None:
            return self._injected_session
        # resolve_location 通过 asyncio.to_thread 执行。Session 并非线程安全，
        # 因此每个工作线程各自复用连接，兼顾 keep-alive 和并发安全。
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _cache_get(self, key: tuple[str, str]) -> dict[str, Any] | None:
        if not self._cache_size:
            return None
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            # 读取即提升为最近使用项；命中缓存时没有地图 HTTP 耗时。
            self._cache.move_to_end(key)
            return {**cached, "cache_hit": True, "elapsed_ms": 0}

    def _cache_put(self, key: tuple[str, str], result: dict[str, Any]) -> None:
        if not self._cache_size:
            return
        with self._cache_lock:
            # elapsed_ms/cache_hit 属于本次调用状态，不写入可复用的业务结果。
            self._cache[key] = {
                item: value
                for item, value in result.items()
                if item not in {"cache_hit", "elapsed_ms"}
            }
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def resolve_location_sync(self, address: str, city: str = "") -> dict[str, Any]:
        """Resolve one address; successful results are cached for this process."""

        address = address.strip()
        city = city.strip()
        if not address:
            raise AMapWebServiceError("地址不能为空。")
        if not self._api_key:
            raise AMapWebServiceError("地址解析服务暂未配置。")

        cache_key = (address, city)
        # 同一会话重复查询相同地址时直接返回，省掉完整的外部网络往返。
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {
            "key": self._api_key,
            "keywords": address,
            # Graph 只消费第一条候选，限制为 1 可减少高德检索和传输的数据量。
            "page_size": "1",
        }
        if city:
            # 已从地址提取出城市时严格限城，降低同名地标跨城误召回的概率。
            params.update({"region": city, "city_limit": "true"})

        started = time.perf_counter()
        # 调用 GET 请求
        try:
            response = self._session().get(
                _PLACE_TEXT_ENDPOINT,
                params=params,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise AMapWebServiceError(
                f"地点搜索请求失败：{type(exc).__name__}。"
            ) from None
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        if not 200 <= response.status_code < 300:
            raise AMapWebServiceError(
                f"地点搜索服务返回 HTTP {response.status_code}。"
            )
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            raise AMapWebServiceError("地点搜索服务没有返回合法 JSON。") from None
        if not isinstance(payload, dict) or str(payload.get("status")) != "1":
            info = payload.get("info") if isinstance(payload, dict) else None
            raise AMapWebServiceError(f"地点搜索失败：{info or 'unknown error'}。")

        poi = _first_poi(payload)
        longitude, latitude = _coordinates(poi)
        # place/text 已同时返回 typecode 和坐标，在同一次响应内完成粒度判断，
        # 不再为了 geocode/geo.level 追加第二个地图请求。
        classification = classify_first_poi(payload, query=address)
        if classification is None:
            raise AMapWebServiceError("地点搜索结果无法判断地理粒度。")
        result = {
            "status": "ok",
            "longitude": longitude,
            "latitude": latitude,
            "poi": {
                "id": poi.get("id"),
                "name": poi.get("name"),
                "address": poi.get("address"),
                "location": poi.get("location"),
                "typecode": poi.get("typecode"),
            },
            "classification": classification,
            "cache_hit": False,
            "elapsed_ms": elapsed_ms,
        }
        self._cache_put(cache_key, result)
        return result

    async def resolve_location(self, address: str, city: str = "") -> dict[str, Any]:
        """Run blocking requests I/O off the LangGraph event loop."""

        # requests 是同步库，放到线程池后不会卡住其它异步 Graph 会话。
        return await asyncio.to_thread(self.resolve_location_sync, address, city)
