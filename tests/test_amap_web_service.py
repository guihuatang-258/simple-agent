from __future__ import annotations

import unittest
from unittest.mock import Mock

from amap_web_service import AMapWebServiceClient, AMapWebServiceError


def _response(*, typecode: str = "150200") -> Mock:
    response = Mock(status_code=200)
    response.json.return_value = {
        "status": "1",
        "info": "OK",
        "pois": [
            {
                "id": "B0016113IJ",
                "name": "天津南站",
                "address": "柳静路",
                "pname": "天津市",
                "cityname": "天津市",
                "adname": "西青区",
                "location": "117.060808,39.056892",
                "typecode": typecode,
            }
        ],
    }
    return response


class AMapWebServiceClientTests(unittest.TestCase):
    def test_place_text_returns_coordinates_and_poi_level(self):
        session = Mock()
        session.get.return_value = _response()
        client = AMapWebServiceClient(api_key="secret", session=session)

        result = client.resolve_location_sync("天津南站", "天津")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["longitude"], 117.060808)
        self.assertEqual(result["latitude"], 39.056892)
        self.assertEqual(result["classification"]["level_code"], "poi")
        self.assertFalse(result["cache_hit"])
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["page_size"], "1")
        self.assertEqual(params["region"], "天津")
        self.assertEqual(params["city_limit"], "true")

    def test_successful_address_is_cached(self):
        session = Mock()
        session.get.return_value = _response()
        client = AMapWebServiceClient(api_key="secret", session=session)

        fresh = client.resolve_location_sync("天津南站", "天津")
        cached = client.resolve_location_sync("天津南站", "天津")

        self.assertFalse(fresh["cache_hit"])
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(cached["elapsed_ms"], 0)
        session.get.assert_called_once()

    def test_missing_key_fails_without_http_request(self):
        session = Mock()
        client = AMapWebServiceClient(api_key="", session=session)

        with self.assertRaises(AMapWebServiceError):
            client.resolve_location_sync("天津南站", "天津")

        session.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
