"""Regex dialogue rules and bounded candidate retrieval for LLM fallback."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any


_RULES_PATH = Path(__file__).resolve().parent / "data" / "dialogue_rules.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())


def _search_text(text: str) -> str:
    return "".join(
        char
        for char in normalize_text(text)
        if unicodedata.category(char)[0] in {"L", "N"}
    )


def _ngrams(text: str, size: int = 2) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def iter_rules(data: dict[str, Any] | None = None):
    catalog = data if data is not None else load_dialogue_rules()
    for category in catalog.get("categories", []):
        for rule in category.get("rules", []):
            yield {**rule, "category": category["name"]}


@lru_cache(maxsize=1)
def load_dialogue_rules() -> dict[str, Any]:
    with _RULES_PATH.open(encoding="utf-8") as source:
        data = json.load(source)
    seen_ids: set[str] = set()
    for rule in iter_rules(data):
        rule_id = rule.get("id")
        if not rule_id or rule_id in seen_ids:
            raise RuntimeError(f"话术规则ID缺失或重复: {rule_id!r}")
        seen_ids.add(rule_id)
        patterns = rule.get("patterns") or []
        if not patterns:
            raise RuntimeError(f"话术规则 {rule_id} 没有正则表达式。")
        for pattern in patterns:
            re.compile(pattern, re.IGNORECASE)
    return data


def get_rule(rule_id: str) -> dict[str, Any] | None:
    return next((rule for rule in iter_rules() if rule["id"] == rule_id), None)


def match_dialogue_rule(
    user_text: str,
    exclude_rule_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    normalized = normalize_text(user_text)
    for rule in iter_rules():
        if exclude_rule_ids and rule["id"] in exclude_rule_ids:
            continue
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in rule["patterns"]):
            return rule
    return None


def retrieve_rule_candidates(
    user_text: str,
    limit: int = 3,
    exclude_rule_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a small candidate set; answers are intentionally excluded."""

    query = _search_text(user_text)
    query_grams = _ngrams(query)
    ranked = []
    for order, rule in enumerate(iter_rules()):
        if exclude_rule_ids and rule["id"] in exclude_rule_ids:
            continue
        keywords = [_search_text(item) for item in rule.get("keywords", [])]
        examples = [_search_text(item) for item in rule.get("examples", [])]
        keyword_score = sum(2.0 for keyword in keywords if keyword and keyword in query)
        similarity = max(
            (SequenceMatcher(None, query, example).ratio() for example in examples),
            default=0.0,
        )
        candidate_grams = _ngrams("".join(keywords + examples))
        union = query_grams | candidate_grams
        overlap = len(query_grams & candidate_grams) / len(union) if union else 0.0
        score = keyword_score + similarity * 3.0 + overlap * 2.0
        ranked.append((score, -order, rule))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        {
            "id": rule["id"],
            "category": rule["category"],
            "topic": rule["topic"],
            "examples": rule.get("examples", []),
            "score": round(score, 4),
        }
        for score, _, rule in ranked[: max(1, limit)]
    ]
