from __future__ import annotations

import unittest

from scripts.geo_level import (
    classify_amap_location,
    classify_first_geocode,
    compare_geographic_scope,
)


class GeographicLevelTests(unittest.TestCase):
    def test_district_level_comparisons(self):
        result = classify_amap_location(
            {
                "province": "天津市",
                "city": "天津市",
                "district": "西青区",
                "level": "区县",
            }
        )

        self.assertEqual(result["level_code"], "district")
        self.assertEqual(result["relative_to_city"], "finer")
        self.assertEqual(result["relative_to_district"], "same")
        self.assertTrue(result["is_district_or_finer"])

    def test_province_is_broader_than_city_and_district(self):
        self.assertEqual(compare_geographic_scope(
            "province", "city"), "broader")
        self.assertEqual(compare_geographic_scope(
            "province", "district"), "broader")

    def test_municipality_is_normalized_to_city_for_business_scope(self):
        result = classify_amap_location(
            {
                "province": "天津市",
                "city": "天津市",
                "level": "省",
            }
        )

        self.assertEqual(result["level_code"], "city")
        self.assertEqual(result["source"],
                         "amap_level_municipality_normalized")
        self.assertEqual(result["relative_to_city"], "same")

    def test_raw_level_has_priority_over_empty_township_field(self):
        result = classify_amap_location(
            {
                "province": "天津市",
                "city": "天津市",
                "district": "西青区",
                "township": [],
                "level": "乡镇",
            }
        )

        self.assertEqual(result["level_code"], "township")
        self.assertEqual(result["relative_to_district"], "finer")

    def test_place_result_is_not_used_as_geocode_level(self):
        result = classify_first_geocode(
            {
                "pois": [
                    {
                        "id": "B001",
                        "name": "天津南站",
                        "location": "117.05,39.05",
                        "adname": "西青区",
                    }
                ]
            }
        )

        self.assertIsNone(result)

    def test_geocode_interest_point_level_is_supported(self):
        result = classify_first_geocode(
            {
                "geocodes": [
                    {
                        "formatted_address": "北京市海淀区颐和园路5号北京大学",
                        "level": "兴趣点",
                    }
                ]
            }
        )

        self.assertEqual(result["level_code"], "poi")
        self.assertEqual(result["source"], "amap_level")
        self.assertEqual(result["relative_to_district"], "finer")

    def test_geocode_door_address_level_is_supported(self):
        result = classify_first_geocode(
            {"geocodes": [{"formatted_address": "颐和园路5号", "level": "门址"}]}
        )

        self.assertEqual(result["level_code"], "address")

    def test_unknown_level_falls_back_to_deepest_known_field(self):
        result = classify_amap_location(
            {
                "province": "天津市",
                "city": "天津市",
                "district": "西青区",
                "level": "未知",
            }
        )

        self.assertEqual(result["level_code"], "district")
        self.assertEqual(result["source"], "field_inference")
        self.assertEqual(result["raw_level"], "未知")


if __name__ == "__main__":
    unittest.main()
