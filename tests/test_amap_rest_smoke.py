from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from scripts.amap_rest_smoke import (
    _compact_params,
    _detailed_address_from_first_poi,
    _first_poi_id,
    _location_from_geo,
    _request,
    _validate_text_search,
)


class AMapRestSmokeTests(unittest.TestCase):
    def test_optional_parameters_are_compacted(self):
        self.assertEqual(
            _compact_params(
                keywords="北京大学",
                types="",
                city_limit=True,
                page_size=None,
            ),
            {"keywords": "北京大学", "city_limit": "true"},
        )

    def test_text_search_requires_keywords_or_types(self):
        with self.assertRaises(ValueError):
            _validate_text_search({"region": "北京市"})

    def test_chained_fields_are_extracted(self):
        self.assertEqual(
            _location_from_geo({"geocodes": [{"location": "117.05,39.05"}]}),
            "117.05,39.05",
        )
        self.assertEqual(_first_poi_id({"pois": [{"id": "B001"}]}), "B001")

    def test_first_text_result_builds_detailed_geocode_address(self):
        address, city, poi = _detailed_address_from_first_poi(
            {
                "pois": [
                    {
                        "id": "B000A816R6",
                        "pname": "北京市",
                        "cityname": "北京市",
                        "adname": "海淀区",
                        "address": "颐和园路5号",
                        "name": "北京大学",
                    }
                ]
            }
        )

        self.assertEqual(address, "北京市海淀区颐和园路5号北京大学")
        self.assertEqual(city, "北京市")
        self.assertEqual(poi["id"], "B000A816R6")

    @patch("builtins.print")
    def test_request_adds_key_but_does_not_log_it(self, mocked_print):
        response = Mock(status_code=200)
        response.json.return_value = {"status": "1", "info": "OK"}
        session = Mock()
        session.get.return_value = response

        result = _request(
            session,
            "https://restapi.amap.com/v5/place/text",
            {"keywords": "北京大学"},
            key="secret-key",
            timeout=15,
        )

        self.assertEqual(result["status"], "1")
        self.assertEqual(
            session.get.call_args.kwargs["params"]["key"], "secret-key")
        output = " ".join(str(call) for call in mocked_print.call_args_list)
        self.assertNotIn("secret-key", output)


if __name__ == "__main__":
    unittest.main()
