from __future__ import annotations

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from components.customer_service.branch_tools import (
    get_compliant_marketing_message,
    load_branch_catalog,
    recommend_nearby_branch_data,
    request_human_handoff,
)


class BranchRecommendationTests(unittest.TestCase):
    def test_catalog_contains_all_unique_workbook_branches(self):
        branches = load_branch_catalog()["branches"]
        self.assertEqual(len(branches), 64)
        self.assertEqual(len({branch["id"] for branch in branches}), 64)
        self.assertEqual(len({branch["name"] for branch in branches}), 64)

    def test_duplicate_workbook_row_preserves_both_phone_numbers(self):
        branches = load_branch_catalog()["branches"]
        tianjin_south = next(
            branch for branch in branches if branch["name"] == "天津南站服务点"
        )
        self.assertEqual(
            tianjin_south["phones"],
            ["17822068818", "17822601703"],
        )

    def test_returns_the_two_nearest_branches(self):
        result = recommend_nearby_branch_data(
            117.055065,
            39.053851,
            "2026-09-07T10:00:00+08:00",
        )
        self.assertEqual(result["recommendation_reason"], "nearest_open")
        self.assertEqual(len(result["nearby_branches"]), 2)
        self.assertEqual(result["nearby_branches"][0]["name"], "天津南站服务点")
        self.assertEqual(result["nearby_branches"][1]["name"], "潮向城自助点")
        self.assertLessEqual(
            result["nearby_branches"][0]["distance_km"],
            result["nearby_branches"][1]["distance_km"],
        )
        self.assertTrue(result["recommended_branch"]["is_open"])

    def test_closed_nearest_branches_are_still_returned_with_status(self):
        result = recommend_nearby_branch_data(
            117.063901,
            38.959319,
            "2026-09-07T08:30:00+08:00",
        )
        self.assertEqual(result["nearest_branch"]["name"], "静海体育学院自助点")
        self.assertFalse(result["nearest_branch"]["is_open"])
        self.assertEqual(len(result["nearby_branches"]), 2)
        self.assertEqual(result["recommendation_reason"], "no_open_branch")
        self.assertIsNone(result["recommended_branch"])

    def test_no_open_or_24_hour_branch_after_closing(self):
        result = recommend_nearby_branch_data(
            117.055065,
            39.053851,
            "2026-09-07T21:00:00+08:00",
        )
        self.assertEqual(result["recommendation_reason"], "no_open_branch")
        self.assertIsNone(result["recommended_branch"])
        self.assertEqual(len(result["nearby_branches"]), 2)

    def test_does_not_return_branches_beyond_thirty_kilometers(self):
        result = recommend_nearby_branch_data(
            116.4074,
            39.9042,
            "2026-09-07T10:00:00+08:00",
        )
        self.assertEqual(
            result["recommendation_reason"], "no_branch_within_radius"
        )
        self.assertEqual(result["max_distance_km"], 30.0)
        self.assertEqual(result["nearby_branches"], [])
        self.assertEqual(result["all_candidates"], [])
        self.assertIsNone(result["nearest_branch"])

    def test_returns_one_branch_when_only_one_is_within_radius(self):
        result = recommend_nearby_branch_data(
            117.417873,
            40.023531,
            "2026-09-07T10:00:00+08:00",
        )
        self.assertEqual(len(result["nearby_branches"]), 1)
        self.assertEqual(
            result["nearby_branches"][0]["name"], "蓟州新城国际自助点"
        )
        self.assertLessEqual(result["nearby_branches"][0]["distance_km"], 30)

    def test_closed_nearest_branch_prefers_24_hour_fallback(self):
        branches = deepcopy(load_branch_catalog()["branches"][:2])
        closed_nearest = branches[0]
        closed_nearest.update(
            {
                "longitude": 117.055065,
                "latitude": 39.053851,
            }
        )
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
            "components.customer_service.branch_tools.load_branch_catalog",
            return_value={
                "result_limit": 2,
                "max_return_distance_km": 30,
                "branches": [closed_nearest, always_open],
            },
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
