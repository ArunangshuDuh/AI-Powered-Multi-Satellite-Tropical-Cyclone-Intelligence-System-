"""
Fetches real, current GOES satellite imagery from NOAA/NESDIS/STAR's public CDN.
No API key required — this is the same CDN NOAA's own public viewer pages
(star.nesdis.noaa.gov/GOES/...) and several open-source tools pull from.

IMPORTANT — could not be tested from the sandbox this was built in (no outbound
network access there at all). The logic below is written defensively (timeouts,
explicit exceptions, no silent fallback to fake data) and follows the same
directory-scrape approach several existing public tools use, but you should run
`python -c "from live.goes_fetch import get_latest_image; get_latest_image('east','full_disk')"`
once locally to confirm it works before relying on it for a demo.

How it works: cdn.star.nesdis.noaa.gov doesn't expose a stable "latest.jpg" URL —
each image file is named with a timestamp prefix (e.g.
"20260908083000_GOES19-ABI-FD-GEOCOLOR-1808x1808.jpg"), and the only way to find the
current one is to fetch the directory listing page and pick the newest filename.
That's what this module does.

NOAA occasionally re-designates which physical satellite serves as GOES-East vs.
GOES-West (GOES-16 -> GOES-19 as East happened in 2025; GOES-17 -> GOES-18 as West
happened earlier). If requests here start failing with 404s, check
https://www.star.nesdis.noaa.gov/GOES/ and update SATELLITES below.
"""
import re
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

CDN_BASE = "https://cdn.star.nesdis.noaa.gov"

# Update these if NOAA re-designates GOES-East / GOES-West again.
SATELLITES = {
    "east": "GOES19",
    "west": "GOES18",
}

# "full_disk" is the safest option — always valid regardless of where a storm is.
# Sector codes below were confirmed against NOAA's public sector viewer pages, but
# NOAA has changed sector naming before, so full_disk is the recommended default.
REGIONS = {
    "full_disk": "FD",
    "caribbean": "SECTOR/car",
    "gulf": "SECTOR/gm",
    "southeast_us": "SECTOR/se",
    "northeast_us": "SECTOR/ne",
    "pacific_southwest": "SECTOR/psw",
    "pacific_northwest": "SECTOR/pnw",
}

PRODUCT = "GEOCOLOR"  # true-color by day, multispectral IR by night — most legible for a demo
REQUEST_TIMEOUT_SEC = 10
_CACHE_TTL_SEC = 300  # NOAA only refreshes every 10-15 min anyway; avoid hammering it
_cache = {}  # (satellite_key, region_key) -> (fetched_at, filename, image_bytes)


class GoesFetchError(Exception):
    """Raised on any upstream failure (network, parsing, or NOAA-side change)."""


def _directory_url(satellite_key: str, region_key: str) -> str:
    if satellite_key not in SATELLITES:
        raise GoesFetchError(f"unknown satellite '{satellite_key}', expected one of {list(SATELLITES)}")
    if region_key not in REGIONS:
        raise GoesFetchError(f"unknown region '{region_key}', expected one of {list(REGIONS)}")
    sat = SATELLITES[satellite_key]
    region_path = REGIONS[region_key]
    return f"{CDN_BASE}/{sat}/ABI/{region_path}/{PRODUCT}/"


def _http_get(url: str) -> bytes:
    try:
        req = Request(url, headers={"User-Agent": "tropical-cyclone-ai-backend/1.0"})
        with urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return resp.read()
    except HTTPError as e:
        raise GoesFetchError(f"NOAA CDN returned HTTP {e.code} for {url}") from e
    except URLError as e:
        raise GoesFetchError(f"could not reach NOAA CDN ({e.reason}) — check internet access") from e


def _latest_filename(directory_html: bytes) -> str:
    """Directory listings link every timestamped image; pick the newest by timestamp prefix."""
    text = directory_html.decode("utf-8", errors="ignore")
    candidates = re.findall(r'href="(\d{10,14}_[^"]+\.jpg)"', text)
    if not candidates:
        raise GoesFetchError("no timestamped .jpg files found in NOAA directory listing (page format may have changed)")

    def ts(name):
        return int(re.match(r"(\d+)_", name).group(1))

    return max(candidates, key=ts)


def get_latest_image(satellite_key: str = "east", region_key: str = "full_disk"):
    """
    Returns (image_bytes, filename, cached: bool) for the most recent real GOES
    image for the given satellite ("east"/"west") and region (see REGIONS keys).
    Raises GoesFetchError on any failure — callers should catch this and return
    a clean 502, never fall back to fake image bytes.
    """
    cache_key = (satellite_key, region_key)
    cached = _cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SEC:
        return cached[2], cached[1], True

    directory_url = _directory_url(satellite_key, region_key)
    listing = _http_get(directory_url)
    filename = _latest_filename(listing)
    image_bytes = _http_get(directory_url + filename)

    _cache[cache_key] = (time.time(), filename, image_bytes)
    return image_bytes, filename, False
