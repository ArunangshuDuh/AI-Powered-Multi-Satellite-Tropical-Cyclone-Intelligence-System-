"""
Optional live-data endpoints. These are ADDITIVE — they don't touch or renumber
anything in the original 6-endpoint frozen contract in api/main.py. Everything
here is real, current data fetched from NOAA at request time (cached briefly),
never precomputed and never fed into the trained model — see README "Live
satellite feature" for why that distinction matters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response

from live.goes_fetch import get_latest_image, GoesFetchError, SATELLITES, REGIONS
from live.nhc_fetch import get_active_storms, NhcFetchError

router = APIRouter(prefix="/api/live")


def error_response(status_code: int, message: str):
    return JSONResponse(status_code=status_code, content={"error": True, "message": message})


@router.get("/storms")
def live_storms():
    """Real, currently-active storms from NOAA NHC. Empty list is normal (off-season)."""
    try:
        storms = get_active_storms()
        return {"storms": storms}
    except NhcFetchError as e:
        return error_response(502, f"could not fetch live storm data from NHC: {e}")


@router.get("/satellite-image")
def live_satellite_image(
    satellite: str = Query("east", description=f"one of {list(SATELLITES)}"),
    region: str = Query("full_disk", description=f"one of {list(REGIONS)}"),
):
    """
    Real, current GOES satellite image (JPEG), proxied from NOAA's public CDN.
    Defaults to full-disk GOES-East, which is always valid regardless of where a
    storm currently is. For display only — not consumed by the trained model.
    """
    try:
        image_bytes, filename, cached = get_latest_image(satellite, region)
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "X-Source-Filename": filename,
                "X-Cache-Hit": str(cached),
                "X-Data-Source": "NOAA/NESDIS/STAR GOES CDN (public, no API key)",
            },
        )
    except GoesFetchError as e:
        return error_response(502, f"could not fetch live satellite image: {e}")
