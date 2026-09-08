from __future__ import annotations

import unittest
from typing import Literal
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict

from dashscope_chat import ChatDashScope


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["faq", "goodbye"]


class DashScopeStructuredOutputTests(unittest.TestCase):
    @patch("dashscope_chat.Generation.call")
    @patch("requests.Session.get")
    def test_json_mode_reaches_native_dashscope_call(self, _get, generation_call):
        generation_call.return_value = {
            "status_code": 200,
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"intent":"faq"}',
                        }
                    }
                ]
            },
        }
        model = ChatDashScope(model="qwen-plus", api_key="test-key")

        structured = model.with_structured_output(_Decision, method="json_mode")
        result = structured.invoke([HumanMessage(content="请输出 JSON")])

        self.assertEqual(result.intent, "faq")
        self.assertEqual(
            generation_call.call_args.kwargs["response_format"],
            {"type": "json_object"},
        )


if __name__ == "__main__":
    unittest.main()
