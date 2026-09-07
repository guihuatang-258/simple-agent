from __future__ import annotations

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from branch_tools import (
    get_compliant_marketing_message,
    load_branch_catalog,
    recommend_nearby_branch_data,
    request_human_handoff,
)


class BranchRecommendationTests(unittest.TestCase):
    def test_catalog_contains_poc_branches(self):
        branches = load_branch_catalog()["branches"]
        self.assertEqual(
            [branch["name"] for branch in branches],
            ["天津南站服务点", "静海体育学院自助点"],
        )

    def test_nearest_open_branch_is_recommended(self):
        result = recommend_nearby_branch_data(
            117.055065,
            39.053851,
            "2026-09-07T10:00:00+08:00",
        )
        self.assertEqual(result["recommendation_reason"], "nearest_open")
        self.assertEqual(
            result["recommended_branch"]["name"], "天津南站服务点"
        )
        self.assertTrue(result["recommended_branch"]["is_open"])

    def test_closed_nearest_branch_falls_back_to_an_open_branch(self):
        result = recommend_nearby_branch_data(
            117.063901,
            38.959319,
            "2026-09-07T08:30:00+08:00",
        )
        self.assertEqual(result["nearest_branch"]["name"], "静海体育学院自助点")
        self.assertFalse(result["nearest_branch"]["is_open"])
        self.assertEqual(result["recommendation_reason"], "nearest_open_fallback")
        self.assertEqual(
            result["recommended_branch"]["name"], "天津南站服务点"
        )

    def test_no_open_or_24_hour_branch_after_closing(self):
        result = recommend_nearby_branch_data(
            117.055065,
            39.053851,
            "2026-09-07T21:00:00+08:00",
        )
        self.assertEqual(result["recommendation_reason"], "no_open_branch")
        self.assertIsNone(result["recommended_branch"])

    def test_closed_nearest_branch_prefers_24_hour_fallback(self):
        branches = deepcopy(load_branch_catalog()["branches"])
        always_open = deepcopy(branches[1])
        always_open.update(
            {
                "id": "poc-24-hour-branch",
                "name": "POC 24小时网点",
                "longitude": 117.07,
                "latitude": 39.06,
                "is_24_hours": True,
            }
        )
        with patch(
            "branch_tools.load_branch_catalog",
            return_value={"branches": [branches[0], always_open]},
        ):
            result = recommend_nearby_branch_data(
                117.055065,
                39.053851,
                "2026-09-07T22:00:00+08:00",
            )
        self.assertEqual(result["recommendation_reason"], "nearest_24h_fallback")
        self.assertEqual(result["recommended_branch"]["name"], "POC 24小时网点")


class PolicyAndHandoffTests(unittest.TestCase):
    def test_peak_season_message_is_compliant_and_enabled(self):
        result = json.loads(get_compliant_marketing_message.invoke({}))
        self.assertTrue(result["peak_season"])
        self.assertTrue(result["should_speak"])
        self.assertIn("以APP实时展示为准", result["message"])

    def test_handoff_requires_confirmation(self):
        result = json.loads(
            request_human_handoff.invoke(
                {"user_confirmed": False, "reason": "超出话术边界"}
            )
        )
        self.assertEqual(result["status"], "confirmation_required")

    def test_confirmed_handoff_creates_poc_event(self):
        result = json.loads(
            request_human_handoff.invoke(
                {"user_confirmed": True, "reason": "用户明确要求人工"}
            )
        )
        self.assertEqual(result["status"], "handoff_requested")
        self.assertEqual(result["mode"], "poc_mock")
        self.assertTrue(result["request_id"])


if __name__ == "__main__":
    unittest.main()
