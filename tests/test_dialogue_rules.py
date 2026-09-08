from __future__ import annotations

import asyncio
import re
import unittest
import uuid

from langchain_core.messages import AIMessage

from agent import build_agent
from components.customer_service.dialogue_rules import (
    get_rule,
    iter_rules,
    load_dialogue_rules,
    match_dialogue_rule,
    retrieve_rule_candidates,
)


class FakeModel:
    def __init__(self, responses: list[str] | None = None):
        self.calls = 0
        self.responses = list(responses or [])
        self.structured_methods: list[str | None] = []

    async def ainvoke(self, messages, config=None):
        del messages, config
        self.calls += 1
        content = self.responses.pop(0) if self.responses else '{"rule_id": null}'
        return AIMessage(content=content)

    def with_structured_output(self, schema, **kwargs):
        self.structured_methods.append(kwargs.get("method"))
        parent = self

        class StructuredFakeModel:
            async def ainvoke(self, messages, config=None):
                response = await parent.ainvoke(messages, config=config)
                return schema.model_validate_json(response.content)

        return StructuredFakeModel()


class FakeAMapClient:
    def __init__(self, *, level: str = "poi", relative: str = "finer"):
        self.calls: list[tuple[str, str]] = []
        self.level = level
        self.relative = relative

    async def resolve_location(self, address: str, city: str = ""):
        self.calls.append((address, city))
        return {
            "status": "ok",
            "longitude": 117.050646,
            "latitude": 39.050010,
            "classification": {
                "level_code": self.level,
                "relative_to_district": self.relative,
            },
            "cache_hit": False,
            "elapsed_ms": 20,
        }


def invoke(graph, text: str, thread_id: str | None = None):
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    return asyncio.run(
        graph.ainvoke({"messages": [{"role": "user", "content": text}]}, config=config)
    )


class DialogueRuleTests(unittest.TestCase):
    def test_all_visible_workbook_marketing_rules_are_loaded(self):
        data = load_dialogue_rules()
        self.assertEqual(len(data["categories"]), 4)
        self.assertEqual(len(list(iter_rules())), 18)
        self.assertEqual(data["source"]["sheet"], "营销话术")
        self.assertEqual(data["source"]["range"], "A3:C20")
        self.assertTrue(all(len(rule["patterns"]) >= 3 for rule in iter_rules()))

    def test_every_pattern_has_a_bounded_full_utterance_guard(self):
        length_guard = re.compile(r"^\^\(\?=\.\{1,(\d+)\}\$\)")
        for rule in iter_rules():
            for pattern in rule["patterns"]:
                with self.subTest(rule=rule["id"], pattern=pattern):
                    match = length_guard.match(pattern)
                    self.assertIsNotNone(match)
                    self.assertLessEqual(int(match.group(1)), 80)
                    self.assertTrue(pattern.endswith("$"))
                    self.assertNotIn(".*", pattern)
                    self.assertNotIn(".+", pattern)

    def test_regex_matches_each_category(self):
        samples = {
            "第一次租车，流程会不会很麻烦还要排队": "rental-process-concern",
            "异地还车费为什么这么贵": "one-way-return-fee-high",
            "跑长途车坏了有没有道路救援": "long-distance-breakdown",
            "五一的价格怎么比平时高这么多": "holiday-price-increase",
        }
        for question, expected in samples.items():
            with self.subTest(question=question):
                self.assertEqual(match_dialogue_rule(question)["id"], expected)

    def test_every_configured_example_routes_to_its_own_rule(self):
        for rule in iter_rules():
            for example in rule.get("examples", []):
                with self.subTest(rule=rule["id"], example=example):
                    self.assertEqual(match_dialogue_rule(example)["id"], rule["id"])

    def test_price_rules_do_not_entangle(self):
        samples = {
            "别家比你们便宜": "competitor-price-cheaper",
            "新能源车怎么比油车还贵": "new-energy-price-higher",
            "租车价格能不能便宜点": "request-lower-price-or-discount",
            "这个价格我还是觉得有点高，能不能优惠一下啊": "price-high-needs-promotion",
            "还车时会不会乱扣款": "holiday-price-accuracy-or-hidden-fees",
            "我再想想": "price-hesitation",
        }
        for question, expected in samples.items():
            with self.subTest(question=question):
                self.assertEqual(match_dialogue_rule(question)["id"], expected)

    def test_long_multi_intent_utterance_does_not_match_a_single_rule(self):
        question = (
            "我想查一下附近网点和电话，再看看有哪些车型，还想问异地还车费为什么高，"
            "新能源车怎么比油车贵，最后确认取车和还车会不会额外收费或者乱扣款"
        )
        self.assertIsNone(match_dialogue_rule(question))

    def test_candidate_retrieval_never_exposes_answers(self):
        candidates = retrieve_rule_candidates("车在半路坏了怎么办", limit=3)
        self.assertLessEqual(len(candidates), 3)
        self.assertTrue(all("answer" not in candidate for candidate in candidates))


class CustomerServiceGraphTests(unittest.TestCase):
    def test_regex_rule_response_only_calls_entry_classifier(self):
        model = FakeModel(
            ['{"intent":"faq","address":"","handoff_decision":"unknown"}']
        )
        graph = build_agent(model=model)
        result = invoke(graph, "第一次租车，流程会不会很麻烦还要排队")
        final = result["messages"][-1]
        self.assertEqual(final.content, get_rule("rental-process-concern")["answer"])
        self.assertEqual(final.additional_kwargs["response_source"], "regex_rule")
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.structured_methods, ["json_mode"])

    def test_bounded_fallback_selects_configured_answer(self):
        model = FakeModel(
            [
                '{"intent":"faq","address":"","handoff_decision":"unknown"}',
                '{"rule_id":"long-distance-breakdown"}',
            ]
        )
        graph = build_agent(model=model)
        result = invoke(graph, "长距离自驾时车半路趴窝会有人处理吗")
        final = result["messages"][-1]
        self.assertEqual(final.content, get_rule("long-distance-breakdown")["answer"])
        self.assertEqual(
            final.additional_kwargs["response_source"], "bounded_semantic_fallback"
        )
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.structured_methods, ["json_mode", "json_mode"])

    def test_unmatched_fallback_then_confirmed_handoff(self):
        model = FakeModel(
            [
                '{"intent":"faq","address":"","handoff_decision":"unknown"}',
                '{"rule_id":null}',
                '{"intent":"human_handoff","address":"","handoff_decision":"confirmed"}',
            ]
        )
        graph = build_agent(model=model)
        thread_id = uuid.uuid4().hex
        first = invoke(graph, "我的发票什么时候开", thread_id)
        self.assertEqual(first["pending_intent"], "handoff_confirmation")
        second = invoke(graph, "好的", thread_id)
        final = second["messages"][-1]
        self.assertEqual(final.additional_kwargs["response_source"], "handoff")
        self.assertEqual(model.calls, 3)

    def test_goodbye_marks_conversation_ended(self):
        model = FakeModel(
            ['{"intent":"goodbye","address":"","handoff_decision":"unknown"}']
        )
        graph = build_agent(model=model)
        result = invoke(graph, "谢谢，再见")
        self.assertTrue(result["conversation_ended"])
        self.assertEqual(result["messages"][-1].content, "感谢您的咨询，再见。")
        self.assertEqual(model.calls, 1)

    def test_branch_address_is_collected_across_turns(self):
        amap_client = FakeAMapClient()
        model = FakeModel(
            [
                '{"intent":"branch_query","address":"","handoff_decision":"unknown"}',
                '{"intent":"branch_query","address":"天津南站","handoff_decision":"unknown"}',
            ]
        )
        graph = build_agent(model=model, amap_client=amap_client)
        thread_id = uuid.uuid4().hex
        first = invoke(graph, "帮我查一下最近的网点", thread_id)
        self.assertEqual(first["pending_intent"], "branch_address")
        second = invoke(graph, "天津南站", thread_id)
        final = second["messages"][-1]
        self.assertEqual(final.additional_kwargs["response_source"], "branch_workflow")
        self.assertIn("天津南站服务点", final.content)
        self.assertEqual(model.calls, 2)
        self.assertEqual(amap_client.calls, [("天津南站", "天津")])

    def test_entry_classifier_sends_branch_query_directly_to_tools(self):
        amap_client = FakeAMapClient()
        model = FakeModel(
            [
                '{"intent":"branch_query","address":"天津南站","handoff_decision":"unknown"}',
            ]
        )
        graph = build_agent(model=model, amap_client=amap_client)
        result = invoke(graph, "天津南站周围可以办理提车吗")
        final = result["messages"][-1]
        self.assertEqual(final.additional_kwargs["response_source"], "branch_workflow")
        self.assertIn("天津南站服务点", final.content)
        self.assertEqual(model.calls, 1)

    def test_branch_query_requests_detail_for_coarse_location(self):
        amap_client = FakeAMapClient(level="city", relative="same")
        model = FakeModel(
            [
                '{"intent":"branch_query","address":"天津市",'
                '"handoff_decision":"unknown"}',
            ]
        )
        graph = build_agent(model=model, amap_client=amap_client)

        result = invoke(graph, "天津市哪个网点最近")
        final = result["messages"][-1]

        self.assertEqual(result["pending_intent"], "branch_address")
        self.assertEqual(
            final.additional_kwargs["response_source"],
            "branch_address_refinement",
        )
        self.assertIn("位置范围较大", final.content)


if __name__ == "__main__":
    unittest.main()
