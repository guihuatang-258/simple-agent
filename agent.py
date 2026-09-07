"""Build the explicit LangGraph customer-service workflow."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from branch_tools import (
    get_compliant_marketing_message,
    recommend_nearby_branch_data,
    request_human_handoff,
)
from dashscope_chat import build_dashscope_model
from dialogue_rules import (
    get_rule,
    match_dialogue_rule,
    normalize_text,
    retrieve_rule_candidates,
)
from mcp_tools import load_amap_store_tools, load_tencent_tools
from openai_chat import build_openai_model


# MCP加载函数不加括号；网点、营销和转人工由显式Graph节点确定性调用。
DEFAULT_TOOLS = [
    load_amap_store_tools,
    # load_tencent_tools,
]

_NATIVE_PROVIDERS = {"dashscope", "tongyi", "qwen"}
_BRANCH_PATTERNS = (
    r"(附近|周边|最近).*(网点|门店|服务点|租车点)",
    r"(网点|门店|服务点|租车点).*(地址|电话|联系方式|在哪|位置|营业时间|几点|开门|关门)",
    r"(有|查|找).*(网点|门店|服务点|租车点)",
    r"(哪里|哪儿|什么地方).*(取车|租车)",
    r"(取车|租车).*(地方|地点|哪里|哪儿)",
)
_HANDOFF_PATTERNS = (
    r"(转|接|找|要|需要|帮我).{0,5}(人工|真人客服|客服人员)",
    r"^(人工|人工客服|真人客服)$",
)
_GOODBYE_PATTERNS = (
    r"^(再见|拜拜|bye|goodbye|结束|没事了|不用了|就这样)(啊|吧|了|谢谢)?[。.!！]?$",
    r"^谢谢.*(再见|拜拜)[。.!！]?$",
)
_AFFIRMATIVE_PATTERN = re.compile(
    r"^(好|好的|可以|行|需要|要|转吧|帮我转|麻烦转|是|嗯|确认)[。.!！]?$"
)
_NEGATIVE_PATTERN = re.compile(r"^(不用|不需要|不要|否|算了|先不用)[。.!！]?$")


class CustomerServiceState(MessagesState):
    route: str
    matched_rule_id: str | None
    pending_intent: str | None
    conversation_ended: bool


def _build_model():
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider in _NATIVE_PROVIDERS:
        return build_dashscope_model()
    return build_openai_model()


def _message_text(message: BaseMessage | Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def _latest_user_text(state: CustomerServiceState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            return _message_text(message).strip()
    return ""


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _model_json(model, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    response = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        config={"tags": ["internal-routing"]},
    )
    return _parse_json_object(_message_text(response))


def _route_request(state: CustomerServiceState) -> dict[str, Any]:
    user_text = _latest_user_text(state)
    normalized = normalize_text(user_text)
    pending = state.get("pending_intent")

    if _matches_any(user_text, _GOODBYE_PATTERNS):
        return {
            "route": "goodbye",
            "matched_rule_id": None,
            "pending_intent": None,
            "conversation_ended": True,
        }

    if pending == "handoff_confirmation":
        if _AFFIRMATIVE_PATTERN.fullmatch(normalized):
            return {
                "route": "handoff",
                "matched_rule_id": None,
                "pending_intent": None,
                "conversation_ended": False,
            }
        if _NEGATIVE_PATTERN.fullmatch(normalized):
            return {
                "route": "handoff_declined",
                "matched_rule_id": None,
                "pending_intent": None,
                "conversation_ended": False,
            }

    if _matches_any(user_text, _HANDOFF_PATTERNS):
        return {
            "route": "handoff",
            "matched_rule_id": None,
            "pending_intent": None,
            "conversation_ended": False,
        }

    if pending == "branch_address" or _matches_any(user_text, _BRANCH_PATTERNS):
        return {
            "route": "branch",
            "matched_rule_id": None,
            "pending_intent": pending if pending == "branch_address" else None,
            "conversation_ended": False,
        }

    rule = match_dialogue_rule(user_text)
    if rule:
        return {
            "route": "rule",
            "matched_rule_id": rule["id"],
            "pending_intent": None,
            "conversation_ended": False,
        }
    return {
        "route": "fallback",
        "matched_rule_id": None,
        "pending_intent": None,
        "conversation_ended": False,
    }


def _goodbye_node(_: CustomerServiceState) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content="感谢您的咨询，再见。")],
        "conversation_ended": True,
        "pending_intent": None,
    }


def _handoff_declined_node(_: CustomerServiceState) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content="好的，您还可以继续咨询其他问题。")],
        "pending_intent": None,
    }


def _handoff_node(state: CustomerServiceState) -> dict[str, Any]:
    result = json.loads(
        request_human_handoff.invoke(
            {
                "user_confirmed": True,
                "reason": f"用户明确请求转人工：{_latest_user_text(state)}",
            }
        )
    )
    return {
        "messages": [
            AIMessage(
                content=result["message"],
                additional_kwargs={
                    "response_source": "handoff",
                    "handoff_request_id": result.get("request_id"),
                },
            )
        ],
        "pending_intent": None,
    }


def _rule_node(state: CustomerServiceState) -> dict[str, Any]:
    rule = get_rule(state.get("matched_rule_id") or "")
    if not rule:
        return {
            "messages": [AIMessage(content="暂时没有找到对应话术，需要我为您转接人工客服吗？")],
            "pending_intent": "handoff_confirmation",
        }
    return {
        "messages": [
            AIMessage(
                content=rule["answer"],
                additional_kwargs={
                    "response_source": "regex_rule",
                    "rule_id": rule["id"],
                    "rule_category": rule["category"],
                },
            )
        ],
        "pending_intent": None,
    }


def _heuristic_address(user_text: str, pending: bool) -> str:
    text = user_text.strip()
    if pending:
        return text
    patterns = (
        r"(?:我(?:现在)?在|当前位置(?:是|在)?|地址(?:是|在)?)(.+?)(?:附近|周边|[,，。?!！]|$)",
        r"^(.+?)(?:附近|周边).*(?:网点|门店|服务点)",
        r"^(.+?)(?:离|距离)(?:哪个|哪家|什么).*(?:网点|门店|服务点)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            address = re.sub(r"^(请问|麻烦问下|帮我查下)", "", match.group(1)).strip()
            if address:
                return address
    return ""


async def _extract_address(model, user_text: str) -> str:
    result = await _model_json(
        model,
        """你只负责从用户消息中提取用户当前所在的地址或地标，不回答问题。
如果消息中没有明确地址，address返回空字符串。不要把目的地、还车地或网点类型
当作当前地址。只输出JSON，例如 {\"address\":\"天津南站\"}。""",
        user_text,
    )
    address = result.get("address", "")
    return address.strip() if isinstance(address, str) else ""


def _tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in result
        )
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def _coordinates_from_geo_result(result: Any) -> tuple[float, float] | None:
    text = _tool_result_text(result)
    payload = _parse_json_object(text)
    candidates = payload.get("return") or payload.get("geocodes") or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            location = str(candidate.get("location", ""))
            match = re.fullmatch(
                r"\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*",
                location,
            )
            if match:
                return float(match.group(1)), float(match.group(2))
    match = re.search(
        r"(?<!\d)(-?\d{2,3}\.\d+)\s*[,，]\s*(-?\d{1,2}\.\d+)(?!\d)",
        text,
    )
    return (float(match.group(1)), float(match.group(2))) if match else None


def _city_hint(address: str) -> str | None:
    match = re.search(r"([\u4e00-\u9fff]{2,8}市)", address)
    if match:
        return match.group(1)
    for municipality in ("北京", "上海", "天津", "重庆"):
        if municipality in address:
            return municipality
    return None


def _branch_response(result: dict[str, Any]) -> tuple[str, str | None]:
    nearest = result["nearest_branch"]
    recommended = result.get("recommended_branch")
    lines: list[str] = []
    if not nearest["is_open"]:
        lines.append(
            f"距您最近的{nearest['name']}当前未营业，营业时间为{nearest['business_hours']}。"
        )
    if recommended:
        status = "当前营业" if recommended["is_open"] else "当前未营业"
        lines.extend(
            [
                f"为您推荐：{recommended['name']}（距您约{recommended['distance_km']:.2f}公里）",
                f"地址：{recommended['short_address']}",
                f"联系电话：{recommended['phone'] or '暂未配置'}",
                f"营业时间：{recommended['business_hours']}，{status}",
            ]
        )
        marketing = json.loads(
            get_compliant_marketing_message.invoke({"scenario": "branch_query"})
        )
        if marketing.get("should_speak") and marketing.get("message"):
            lines.extend(["", marketing["message"]])
        return "\n".join(lines), None

    lines.extend([result["message"], "需要我为您转接人工客服吗？"])
    return "\n".join(lines), "handoff_confirmation"


async def _branch_node(
    state: CustomerServiceState,
    *,
    model,
    tool_map: dict[str, BaseTool],
) -> dict[str, Any]:
    user_text = _latest_user_text(state)
    pending = state.get("pending_intent") == "branch_address"
    address = _heuristic_address(user_text, pending)
    if not address:
        try:
            address = await _extract_address(model, user_text)
        except Exception:
            address = ""
    if not address:
        return {
            "messages": [AIMessage(content="请告诉我您当前所在的详细地址或附近地标。")],
            "pending_intent": "branch_address",
        }

    geo_tool = tool_map.get("maps_geo")
    if geo_tool is None:
        return {
            "messages": [AIMessage(content="地址解析服务暂未配置，需要我为您转接人工客服吗？")],
            "pending_intent": "handoff_confirmation",
        }
    arguments = {"address": address}
    city = _city_hint(address)
    if city:
        arguments["city"] = city
    try:
        geo_result = await geo_tool.ainvoke(arguments)
        coordinates = _coordinates_from_geo_result(geo_result)
    except Exception:
        coordinates = None
    if coordinates is None:
        return {
            "messages": [AIMessage(content="暂时无法定位这个地址，请提供更详细的区、道路或地标信息。")],
            "pending_intent": "branch_address",
        }

    recommendation = recommend_nearby_branch_data(*coordinates)
    if recommendation.get("status") != "ok":
        return {
            "messages": [AIMessage(content="网点查询暂时不可用，需要我为您转接人工客服吗？")],
            "pending_intent": "handoff_confirmation",
        }
    response, next_intent = _branch_response(recommendation)
    return {
        "messages": [
            AIMessage(
                content=response,
                additional_kwargs={"response_source": "branch_workflow"},
            )
        ],
        "pending_intent": next_intent,
    }


async def _fallback_node(
    state: CustomerServiceState,
    *,
    model,
    tool_map: dict[str, BaseTool],
) -> dict[str, Any]:
    user_text = _latest_user_text(state)
    candidates = retrieve_rule_candidates(user_text, limit=3)
    prompt = json.dumps(
        {"user_question": user_text, "candidate_topics": candidates},
        ensure_ascii=False,
    )
    rule_id = None
    try:
        decision = await _model_json(
            model,
            """你是受限的客服话术语义匹配器，不负责回答用户问题。候选最多三条，
只在用户问题与某一候选的topic/examples语义明确一致时返回其id，否则返回null。
不得依据常识扩展候选，不得生成答案。只输出JSON：{\"rule_id\": null}或
{\"rule_id\": \"候选id\"}。""",
            prompt,
        )
        selected = decision.get("rule_id")
        allowed_ids = {candidate["id"] for candidate in candidates}
        if isinstance(selected, str) and selected in allowed_ids:
            rule_id = selected
    except Exception:
        rule_id = None

    if rule_id == "branch-location-or-phone":
        return await _branch_node(state, model=model, tool_map=tool_map)

    rule = get_rule(rule_id) if rule_id else None
    if rule:
        return {
            "messages": [
                AIMessage(
                    content=rule["answer"],
                    additional_kwargs={
                        "response_source": "bounded_semantic_fallback",
                        "rule_id": rule["id"],
                        "rule_category": rule["category"],
                    },
                )
            ],
            "pending_intent": None,
        }
    return {
        "messages": [
            AIMessage(
                content=(
                    "抱歉，目前我只能处理网点查询，以及租车条件、要素查询、"
                    "车况与服务相关问题。需要我为您转接人工客服吗？"
                ),
                additional_kwargs={"response_source": "fallback_reject"},
            )
        ],
        "pending_intent": "handoff_confirmation",
    }


def build_agent(tools: Sequence[BaseTool] = (), *, model=None):
    """Build one-turn routing graph; the checkpointer carries state across turns."""

    selected_model = model or _build_model()
    tool_map = {tool.name: tool for tool in tools}

    async def branch_node(state: CustomerServiceState):
        return await _branch_node(state, model=selected_model, tool_map=tool_map)

    async def fallback_node(state: CustomerServiceState):
        return await _fallback_node(
            state,
            model=selected_model,
            tool_map=tool_map,
        )

    workflow = StateGraph(CustomerServiceState)
    workflow.add_node("route", _route_request)
    workflow.add_node("goodbye", _goodbye_node)
    workflow.add_node("handoff", _handoff_node)
    workflow.add_node("handoff_declined", _handoff_declined_node)
    workflow.add_node("rule", _rule_node)
    workflow.add_node("branch", branch_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_edge(START, "route")
    workflow.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {
            "goodbye": "goodbye",
            "handoff": "handoff",
            "handoff_declined": "handoff_declined",
            "branch": "branch",
            "rule": "rule",
            "fallback": "fallback",
        },
    )
    for node in ("goodbye", "handoff", "handoff_declined", "branch", "rule", "fallback"):
        workflow.add_edge(node, END)
    return workflow.compile(checkpointer=InMemorySaver(), name="customer-service-graph")


def build_chat_agent(tools: Sequence[BaseTool] = ()):
    """Build the pure-chat comparison graph used by main_chat.py."""

    del tools
    model = _build_model()

    async def chat_node(state: MessagesState):
        messages = [
            SystemMessage(content="你是一个简洁、可靠的助手。用用户使用的语言回答。"),
            *state["messages"],
        ]
        response = await model.ainvoke(messages, config={"tags": ["user-facing"]})
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("chat", chat_node)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile(checkpointer=InMemorySaver(), name="pure-chat-graph")
