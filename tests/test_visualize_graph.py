from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from visualize_graph import build_graph_structure, main


class VisualizeGraphTests(unittest.TestCase):
    def test_mermaid_contains_customer_service_nodes(self):
        mermaid = build_graph_structure().draw_mermaid()

        for node in ("route", "branch", "rule", "fallback", "handoff", "goodbye"):
            with self.subTest(node=node):
                self.assertIn(node, mermaid)

    def test_mermaid_can_be_written_without_model_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "graph.mmd"

            result = main(["--format", "mermaid", "--output", str(output)])

            self.assertEqual(result, 0)
            self.assertIn("graph TD", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
