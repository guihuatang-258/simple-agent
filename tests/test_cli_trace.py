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

    def test_branch_path_includes_web_service_performance_details(self):
        path = ["START"]
        _append_execution_path(
            {
                "branch": {
                    "messages": [
                        AIMessage(
                            content="branch answer",
                            additional_kwargs={
                                "response_source": "branch_workflow",
                                "map_provider": "amap_web_service",
                                "map_level": "poi",
                                "map_elapsed_ms": 228,
                                "map_cache_hit": False,
                            },
                        )
                    ]
                }
            },
            path,
        )

        self.assertEqual(
            path[-1],
            "branch(source=branch_workflow, map=amap_web_service, "
            "level=poi, map_ms=228, cache=miss)",
        )


if __name__ == "__main__":
    unittest.main()
