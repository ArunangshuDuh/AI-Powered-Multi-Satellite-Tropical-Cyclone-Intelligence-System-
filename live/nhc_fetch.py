"""
Fetches the real, current list of active tropical storms from NOAA's National
Hurricane Center. No API key required.

Source: https://www.nhc.noaa.gov/CurrentStorms.json — NHC's own documented,
machine-readable feed (used by NOAA's own tools and several open-source trackers).
Root shape is always `{"activeStorms": [...]}`; the array is empty (not missing)
when nothing is active, which is normal outside hurricane season — not an error.

Could not be tested from the sandbox this was built in (no outbound network
access there). Written defensively; run it once locally to confirm before a demo.
"""
import json
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

NHC_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
REQUEST_TIMEOUT_SEC = 10
_CACHE_TTL_SEC = 300
_cache = {"fetched_at": 0, "data": None}


class NhcFetchError(Exception):
    pass


def _http_get_json(url: str):
    try:
        req = Request(url, headers={"User-Agent": "tropical-cyclone-ai-backend/1.0"})
        with urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise NhcFetchError(f"NHC returned HTTP {e.code}") from e
    except URLError as e:
        raise NhcFetchError(f"could not reach NHC ({e.reason}) — check internet access") from e
    except json.JSONDecodeError as e:
        raise NhcFetchError(f"NHC response was not valid JSON: {e}") from e


def _normalize(raw_storm: dict) -> dict:
    """Maps NHC's camelCase fields to the same shape used elsewhere in this API."""
    lat = raw_storm.get("latitudeNumeric")
    lon = raw_storm.get("longitudeNumeric")
    return {
        "storm_id": raw_storm.get("id") or raw_storm.get("binNumber"),
        "name": raw_storm.get("name"),
        "classification": raw_storm.get("classification"),  # e.g. "HU", "TS", "TD"
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "intensity_kt": raw_storm.get("intensity"),
        "pressure_mb": raw_storm.get("pressure"),
        "last_update": raw_storm.get("lastUpdate"),
        "public_advisory_url": raw_storm.get("publicAdvisory", {}).get("url")
        if isinstance(raw_storm.get("publicAdvisory"), dict) else None,
    }


def get_active_storms():
    """Returns a list of real, currently-active storms (possibly empty). Raises
    NhcFetchError on upstream failure — callers should return a clean 502."""
    if _cache["data"] is not None and (time.time() - _cache["fetched_at"]) < _CACHE_TTL_SEC:
        return _cache["data"]

    raw = _http_get_json(NHC_URL)
    storms = [_normalize(s) for s in raw.get("activeStorms", [])]

    _cache["data"] = storms
    _cache["fetched_at"] = time.time()
    return storms
