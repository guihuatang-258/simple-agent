from __future__ import annotations

import json
import unittest

from amap_tool_smoke import _first_poi_id, _location_from_geo, _result_payload


class AMapToolSmokeTests(unittest.TestCase):
    def test_mcp_content_blocks_are_parsed(self):
        result = [{"type": "text", "text": json.dumps({"status": "ok"})}]
        self.assertEqual(_result_payload(result), {"status": "ok"})

    def test_geo_location_is_extracted(self):
        result = [
            {
                "type": "text",
                "text": json.dumps({"return": [{"location": "117.05,39.05"}]}),
            }
        ]
        self.assertEqual(_location_from_geo(result), "117.05,39.05")

    def test_first_poi_id_is_extracted(self):
        result = [
            {
                "type": "text",
                "text": json.dumps({"pois": [{"id": "B001", "name": "停车场"}]}),
            }
        ]
        self.assertEqual(_first_poi_id(result), "B001")


if __name__ == "__main__":
    unittest.main()
