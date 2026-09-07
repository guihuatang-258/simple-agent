"""话术规则的确定性正则匹配，以及供 LLM 兜底使用的受限候选召回。"""

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
    """统一全半角和大小写并移除空白，降低输入格式对正则命中的影响。"""

    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())


def _search_text(text: str) -> str:
    """生成词法召回文本；标点不参与相似度计算，只保留字母和数字。"""

    return "".join(
        char
        for char in normalize_text(text)
        if unicodedata.category(char)[0] in {"L", "N"}
    )


def _ngrams(text: str, size: int = 2) -> set[str]:
    """切分字符 n-gram；中文没有天然空格，因此默认使用二元字符集合。"""

    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def iter_rules(data: dict[str, Any] | None = None):
    """按配置顺序展开分类下的规则，并把分类名附加到每条规则。"""

    catalog = data if data is not None else load_dialogue_rules()
    for category in catalog.get("categories", []):
        for rule in category.get("rules", []):
            yield {**rule, "category": category["name"]}


@lru_cache(maxsize=1)
def load_dialogue_rules() -> dict[str, Any]:
    """加载并校验规则库；缓存结果，避免每轮对话重复读取和编译配置。"""

    with _RULES_PATH.open(encoding="utf-8") as source:
        data = json.load(source)
    seen_ids: set[str] = set()
    for rule in iter_rules(data):
        # ID 是后续 LLM 复核和标准话术回读的唯一契约，必须存在且不可重复。
        rule_id = rule.get("id")
        if not rule_id or rule_id in seen_ids:
            raise RuntimeError(f"话术规则ID缺失或重复: {rule_id!r}")
        seen_ids.add(rule_id)
        patterns = rule.get("patterns") or []
        if not patterns:
            raise RuntimeError(f"话术规则 {rule_id} 没有正则表达式。")
        # 启动时提前发现非法正则，避免咨询过程中才触发配置错误。
        for pattern in patterns:
            re.compile(pattern, re.IGNORECASE)
    return data


def get_rule(rule_id: str) -> dict[str, Any] | None:
    """通过已校验的 ID 回读完整规则；标准答案只在这个阶段取出。"""

    return next((rule for rule in iter_rules() if rule["id"] == rule_id), None)


def match_dialogue_rule(
    user_text: str,
    exclude_rule_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """按配置顺序执行正则，返回第一条确定性命中的规则。"""

    normalized = normalize_text(user_text)
    for rule in iter_rules():
        # 网点查询等由专用 Agent 工具负责的意图不能被 FAQ 话术截获。
        if exclude_rule_ids and rule["id"] in exclude_rule_ids:
            continue
        # 一条用户问题可配置多条正则，任意一条命中即可直接返回标准话术。
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in rule["patterns"]):
            return rule
    return None


def retrieve_rule_candidates(
    user_text: str,
    limit: int = 3,
    exclude_rule_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """用本地词法特征召回少量候选；返回结果刻意排除标准答案。"""

    # 该函数不是向量检索：query 和候选都不会调用 embedding 模型或向量库。
    query = _search_text(user_text)
    query_grams = _ngrams(query)
    ranked = []
    for order, rule in enumerate(iter_rules()):
        # 与正则阶段保持相同的业务边界，防止专用工具意图混入 FAQ 候选。
        if exclude_rule_ids and rule["id"] in exclude_rule_ids:
            continue
        keywords = [_search_text(item) for item in rule.get("keywords", [])]
        examples = [_search_text(item) for item in rule.get("examples", [])]

        # 完整关键词命中是最强的词法信号，每命中一个固定加 2 分。
        keyword_score = sum(2.0 for keyword in keywords if keyword and keyword in query)

        # 取用户问题与所有示例中最高的字符序列相似度，避免示例数量影响分数。
        similarity = max(
            (SequenceMatcher(None, query, example).ratio() for example in examples),
            default=0.0,
        )

        # 二元字符 Jaccard 补充捕获中文短语局部重叠，例如“道路救援/需要救援”。
        candidate_grams = _ngrams("".join(keywords + examples))
        union = query_grams | candidate_grams
        overlap = len(query_grams & candidate_grams) / len(union) if union else 0.0

        # 将三类互补的词法信号加权合并，不在此阶段做最终语义判定。
        score = keyword_score + similarity * 3.0 + overlap * 2.0
        # -order 让同分规则保持 JSON 中的原始顺序，保证结果稳定可复现。
        ranked.append((score, -order, rule))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # 只暴露 ID、主题和示例给 LLM；answer 始终留在本地，由代码最终回读。
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
