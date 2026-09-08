"""Infer geographic scope from AMap geocode and POI result fields."""

from __future__ import annotations

from typing import Any


# scope_rank 越小表示覆盖范围越大，越大表示位置越精细。
_LEVELS = {
    "country": (0, "国家级"),
    "province": (1, "省级"),
    "city": (2, "市级"),
    "district": (3, "区级"),
    "township": (4, "乡镇/街道级"),
    "neighborhood": (5, "村庄/商圈级"),
    "street": (6, "道路级"),
    "address": (7, "门牌级"),
    "poi": (7, "POI点位级"),
}

_AMAP_LEVEL_MAP = {
    "国家": "country",
    "省": "province",
    "市": "city",
    "区县": "district",
    "开发区": "district",
    "乡镇": "township",
    "村庄": "neighborhood",
    "热点商圈": "neighborhood",
    "道路": "street",
    "道路交叉口": "street",
    "门址": "address",
    "门牌号": "address",
    "单元号": "address",
    "兴趣点": "poi",
    "公交站台": "poi",
    "地铁站": "poi",
}

_MUNICIPALITIES = {"北京", "北京市", "上海", "上海市", "天津", "天津市", "重庆", "重庆市"}


def _has_value(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, dict):
        return any(_has_value(item) for item in value.values())
    return True


def compare_geographic_scope(level_code: str, reference_code: str) -> str:
    """Return broader/same/finer by geographic coverage, not numeric precision."""

    actual = _LEVELS.get(level_code)
    reference = _LEVELS.get(reference_code)
    if actual is None or reference is None:
        return "unknown"
    if actual[0] < reference[0]:
        return "broader"
    if actual[0] > reference[0]:
        return "finer"
    return "same"


def _infer_from_fields(record: dict[str, Any]) -> tuple[str, str]:
    # place/text、place/around、place/detail 的 POI 同时具有 id、name 和 location。
    if all(_has_value(record.get(field)) for field in ("id", "name", "location")):
        return "poi", "poi_fields"
    if _has_value(record.get("number")) or _has_value(record.get("building")):
        return "address", "field_inference"
    if _has_value(record.get("street")):
        return "street", "field_inference"
    if _has_value(record.get("neighborhood")):
        return "neighborhood", "field_inference"
    if _has_value(record.get("township")):
        return "township", "field_inference"
    if _has_value(record.get("district")) or _has_value(record.get("adname")):
        return "district", "field_inference"
    if _has_value(record.get("city")) or _has_value(record.get("cityname")):
        return "city", "field_inference"
    if _has_value(record.get("province")) or _has_value(record.get("pname")):
        return "province", "field_inference"
    if _has_value(record.get("country")):
        return "country", "field_inference"
    return "unknown", "unknown"


def classify_amap_location(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one AMap geocode or POI record and compare it with city/district."""

    raw_level = str(record.get("level") or "").strip()
    level_code = _AMAP_LEVEL_MAP.get(raw_level)
    source = "amap_level" if level_code else ""

    # 高德把直辖市单独查询标为“省”；业务上的地址完整度按市级理解更实用。
    province = str(record.get("province") or record.get("pname") or "").strip()
    city = str(record.get("city") or record.get("cityname") or "").strip()
    if level_code == "province" and province in _MUNICIPALITIES and city == province:
        level_code = "city"
        source = "amap_level_municipality_normalized"

    # level=未知或字段缺失时，按结果中最精细的非空字段保守推断。
    if level_code is None:
        level_code, source = _infer_from_fields(record)

    level = _LEVELS.get(level_code)
    if level is None:
        return {
            "level_code": "unknown",
            "level_label": "未知",
            "raw_level": raw_level or None,
            "source": source or "unknown",
            "relative_to_city": "unknown",
            "relative_to_district": "unknown",
            "is_city_or_finer": False,
            "is_district_or_finer": False,
        }

    return {
        "level_code": level_code,
        "level_label": level[1],
        "raw_level": raw_level or None,
        "source": source,
        "relative_to_city": compare_geographic_scope(level_code, "city"),
        "relative_to_district": compare_geographic_scope(level_code, "district"),
        "is_city_or_finer": level[0] >= _LEVELS["city"][0],
        "is_district_or_finer": level[0] >= _LEVELS["district"][0],
    }


def classify_first_geocode(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Classify geocodes[0]; place search results are intentionally excluded."""

    candidates = payload.get("geocodes") or []
    if not isinstance(candidates, list):
        return None
    first = next((item for item in candidates if isinstance(item, dict)), None)
    return classify_amap_location(first) if first else None
