from __future__ import annotations

import unittest

from scripts.geo_level import (
    classify_amap_location,
    classify_amap_poi,
    classify_first_geocode,
    classify_first_poi,
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

    def test_place_result_is_classified_separately_from_geocode(self):
        payload = {
            "pois": [
                {
                    "id": "B001",
                    "name": "天津南站",
                    "location": "117.05,39.05",
                    "typecode": "150200",
                    "adname": "西青区",
                }
            ]
        }

        self.assertIsNone(classify_first_geocode(payload))
        result = classify_first_poi(payload)

        self.assertEqual(result["level_code"], "poi")
        self.assertEqual(result["source"], "poi_typecode")
        self.assertEqual(result["raw_typecode"], "150200")
        self.assertTrue(result["is_district_or_finer"])

    def test_administrative_place_typecodes_have_explicit_levels(self):
        cases = {
            "190102": "province",
            "190103": "city",
            "190104": "city",
            "190105": "district",
            "190106": "township",
            "190107": "subdistrict",
            "190108": "village",
            "190109": "village_group",
        }

        for typecode, expected in cases.items():
            with self.subTest(typecode=typecode):
                result = classify_amap_poi({"typecode": typecode})
                self.assertEqual(result["level_code"], expected)
                self.assertEqual(result["source"], "poi_typecode")

    def test_road_and_address_typecodes_are_finer_than_district(self):
        road = classify_amap_poi({"typecode": "190302"})
        address = classify_amap_poi({"typecode": "190403"})

        self.assertEqual(road["level_code"], "precise_location")
        self.assertEqual(address["level_code"], "address")
        self.assertEqual(road["relative_to_district"], "finer")
        self.assertEqual(address["relative_to_district"], "finer")

    def test_ambiguous_place_name_typecode_stays_unknown(self):
        result = classify_amap_poi(
            {
                "id": "B002",
                "name": "某自然地名",
                "location": "117.05,39.05",
                "typecode": "190203",
            }
        )

        self.assertEqual(result["level_code"], "unknown")
        self.assertEqual(result["source"], "poi_typecode_ambiguous")

    def test_exact_admin_query_overrides_rewritten_poi_result(self):
        result = classify_amap_poi(
            {
                "id": "B003",
                "name": "北京市人民政府(旧址)",
                "pname": "北京市",
                "cityname": "北京市",
                "adname": "东城区",
                "location": "116.407387,39.904179",
                "typecode": "130102",
            },
            query="北京市",
        )

        self.assertEqual(result["level_code"], "city")
        self.assertEqual(result["source"], "query_admin_field")
        self.assertEqual(result["relative_to_city"], "same")
        self.assertEqual(result["relative_to_district"], "broader")

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
