from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from main import _append_execution_path


class CliExecutionPathTests(unittest.TestCase):
    def test_route_and_rule_updates_include_debug_details(self):
        path = ["START"]

        _append_execution_path(
            {
                "route": {
                    "route": "rule",
                    "matched_rule_id": "rental-process-concern",
                }
            },
            path,
        )
        _append_execution_path(
            {
                "rule": {
                    "messages": [
                        AIMessage(
                            content="configured answer",
                            additional_kwargs={
                                "response_source": "regex_rule",
                                "rule_id": "rental-process-concern",
                            },
                        )
                    ]
                }
            },
            path,
        )

        self.assertEqual(
            path,
            [
                "START",
                "route(decision=rule, rule_id=rental-process-concern)",
                "rule(source=regex_rule, rule_id=rental-process-concern)",
            ],
        )

    def test_internal_update_keys_are_not_printed_as_nodes(self):
        path = ["START"]
        _append_execution_path({"__interrupt__": ()}, path)
        self.assertEqual(path, ["START"])


if __name__ == "__main__":
    unittest.main()
