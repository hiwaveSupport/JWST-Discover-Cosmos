import io
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

from astropy.io import fits
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import httpx
import numpy as np
from PIL import Image

# Configure clean, highly structured logging output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("JWST-Chromatic-Backend")

# 1. Resolve absolute server execution directory coordinates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Bind paths explicitly to guarantee target files locate correctly on remote cloud servers
DB_NAME = os.path.join(BASE_DIR, "jwst_archive.db")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(
    title="JWST Chromatic Translation Discovery Engine",
    description="A non-blocking API featuring unified multi-column text lookups, category indexing trackers, representative color vector data streaming, and live on-the-fly FITS processing paths.",
    version="1.6.2",
)

# Initialize Jinja2 Engine using absolute path structures
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Enable CORS so your frontend layout can communicate smoothly with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection() -> sqlite3.Connection:
    """Creates a row-factory database connection for clear key-value dictionary mappings."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def apply_astronomical_stretch(image_data: np.ndarray) -> np.ndarray:
    """Transforms a high-dynamic-range scientific float matrix into a display-ready 8-bit integer array.

    Uses percentile clipping to drop noise spikes and log-scaling to enhance
    faint gas structure.
    """
    # Neutralize bad pixel dropouts, NaNs, and instrument infinity anomalies
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract boundaries eliminating top and bottom outliers (1st to 99.5th percentile)
    vmin, vmax = np.percentile(image_data, [1.0, 99.5])
    clipped = np.clip(image_data, vmin, vmax)

    # Execute Logarithmic non-linear stretching
    stretched = np.log1p(clipped - vmin)

    # Map cleanly across a standard 0-255 viewport scale
    span = stretched.max() - stretched.min()
    if span == 0:
        span = 1e-8

    normalized = (stretched - stretched.min()) / span
    return (normalized * 255).astype(np.uint8)


# --- CORE FRONTEND TEMPLATE ROUTE ---


@app.get("/")
async def get_index(request: Request):
    """Serves the user interface portal directly from the server route.

    Uses an defensive polymorphic fallback loop to maintain 100% backward and
    forward compatibility across changing Starlette versions.
    """
    logger.info(
        "Serving user portal matrix index template layout across root gateway."
    )
    try:
        # Modern Starlette 0.29.0+ / 1.0.0+ Signature Structure
        return templates.TemplateResponse(
            request=request, name="index.html", context={"request": request}
        )
    except TypeError:
        # Legacy Starlette Version Fallback Signature Structure
        logger.info(
            "Newer signature rejected. Applying legacy version template translation fallback wrapper loop."
        )
        return templates.TemplateResponse("index.html", {"request": request})


# --- NEW FRONTEND HARMONIZATION API ENDPOINT ---


@app.get("/api/previews", response_model=List[Dict[str, Any]])
def get_previews():
    """Serves a raw array containing the completely enriched chromatic variables.

    This ensures full out-of-the-box compatibility with the index.html
    rendering layer.
    """
    logger.info(
        "Broad chromatic query requested directly via frontend archive interface."
    )
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            obs_id, 
            filters, 
            filter_human_label, 
            impact_description, 
            primary_color_hex, 
            primary_color_name
        FROM jwst_previews
        WHERE filter_human_label IS NOT NULL
        ORDER BY target_name ASC
    """
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(
            f"Failed to extract unified preview collection sequence: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal SQL payload extraction pipeline error.",
        )
    finally:
        conn.close()


# --- API UTILITY INVENTORY ENDPOINTS ---


@app.get("/api/obs-inventory", response_model=List[Dict[str, Any]])
def get_obs_inventory():
    """Returns a list of all unique obs_ids paired with target names and titles to cleanly populate dropdown frames."""
    logger.info("Inventory lookup requested for picker selection dropdown maps.")
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT obs_id, target_name, observation_title 
        FROM jwst_previews 
        ORDER BY target_name ASC
    """
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Failed to compile inventory cache map: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal metadata inventory lookup failure."
        )
    finally:
        conn.close()


@app.get("/api/classifications", response_model=List[str])
def get_classifications():
    """Sweeps the local cache database repository and extracts a unique list of all non-empty scientific target classification entry tags."""
    logger.info(
        "Dynamic retrieval request launched for target classification strings."
    )
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT target_classification 
        FROM jwst_previews 
        WHERE target_classification IS NOT NULL 
          AND target_classification != ''
        ORDER BY target_classification ASC
    """
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        classifications = [row["target_classification"] for row in rows]
        return classifications
    except sqlite3.Error as e:
        logger.error(
            f"Failed to extract distinct target classifications from cache database: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal scientific taxonomy inventory database error.",
        )
    finally:
        conn.close()


# --- DATA ACQUISITION & SEARCH MATRIX ENDPOINT ---


@app.get("/api/observations", response_model=Dict[str, Any])
def get_observations(
    search: Optional[str] = Query(
        None,
        description="Fuzzy search term scanning name, classification, filter labels, and impact descriptions simultaneously",
    ),
    obs_id: Optional[List[str]] = Query(
        None,
        description="A multi-select array list of absolute observation identifiers to strictly isolate entries",
    ),
    classification: Optional[List[str]] = Query(
        None,
        description="A multi-select list of target classifications to isolate entries by celestial object taxonomy",
    ),
    instrument: Optional[str] = Query(
        None, description="Filter by instrument setup (NIRCAM, MIRI, NIRISS)"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset index"),
):
    """Queries the local database cache to supply your Material Design grid layout with dynamic filtering tracking multi-select keys, natural text keywords, or instrument parameters."""
    logger.info(
        f"Metadata fetch requested. Filters -> Search: {search}, Multi-Select IDs: {obs_id}, Classifications: {classification}, Instrument: {instrument}"
    )
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            obs_id, target_name, target_classification, observation_title, 
            ra, dec, instrument_name, filters, exposure_time,
            filter_human_label, impact_description,
            primary_color_hex, primary_color_name
        FROM jwst_previews 
        WHERE 1=1
    """
    params = []

    # 1. Evaluate Multi-Select Array IDs using a dynamically bound SQL IN clause configuration
    if obs_id and len(obs_id) > 0:
        clean_ids = [uid.strip() for uid in obs_id if uid.strip() != ""]
        if clean_ids:
            placeholders = ", ".join(["?"] * len(clean_ids))
            query += f" AND obs_id IN ({placeholders})"
            params.extend(clean_ids)

    # 2. Evaluate Multi-Select Target Classifications using a dynamic SQL IN clause framework
    if classification and len(classification) > 0:
        clean_classes = [c.strip() for c in classification if c.strip() != ""]
        if clean_classes:
            placeholders = ", ".join(["?"] * len(clean_classes))
            query += f" AND target_classification IN ({placeholders})"
            params.extend(clean_classes)

    # 3. Four-Dimensional Fuzzy Natural Language Text Search Mapping
    if search:
        query += """ AND (
            target_name LIKE ? 
            OR target_classification LIKE ? 
            OR filter_human_label LIKE ? 
            OR impact_description LIKE ?
        )"""
        wildcard_search = f"%{search.strip()}%"
        params.extend([
            wildcard_search,
            wildcard_search,
            wildcard_search,
            wildcard_search,
        ])

    # 4. Evaluate Instrument Filters
    if instrument:
        query += " AND instrument_name LIKE ?"
        params.append(f"%{instrument.upper().strip()}%")

    # Append sorting, limits, and pagination variables
    query += " ORDER BY target_name ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        records = [dict(row) for row in rows]

        # Calculate matching totals by mirroring exact parameter filters inside count engine
        count_query = "SELECT COUNT(*) FROM jwst_previews WHERE 1=1"
        count_params = []

        if obs_id and len(obs_id) > 0:
            clean_ids = [uid.strip() for uid in obs_id if uid.strip() != ""]
            if clean_ids:
                placeholders = ", ".join(["?"] * len(clean_ids))
                count_query += f" AND obs_id IN ({placeholders})"
                count_params.extend(clean_ids)
        if classification and len(classification) > 0:
            clean_classes = [c.strip() for c in classification if c.strip() != ""]
            if clean_classes:
                placeholders = ", ".join(["?"] * len(clean_classes))
                count_query += f" AND target_classification IN ({placeholders})"
                count_params.extend(clean_classes)
        if search:
            count_query += """ AND (
                target_name LIKE ? 
                OR target_classification LIKE ? 
                OR filter_human_label LIKE ? 
                OR impact_description LIKE ?
            )"""
            count_params.extend([
                wildcard_search,
                wildcard_search,
                wildcard_search,
                wildcard_search,
            ])
        if instrument:
            count_query += " AND instrument_name LIKE ?"
            count_params.append(f"%{instrument.upper().strip()}%")

        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]

        return {
            "total_records": total_count,
            "limit": limit,
            "offset": offset,
            "results": records,
        }
    except sqlite3.Error as e:
        logger.error(f"Database read failure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal metadata engine repository error.",
        )
    finally:
        conn.close()


# --- FITS LIVE ARRAY STREAMING ENGINE ---


@app.get("/api/stream/{obs_id}")
async def stream_observation_image(obs_id: str, download: bool = Query(False)):
    """Intercepts an observation ID, fetches the scientific FITS stream from MAST, processes it in RAM, and pipes back an ephemeral, high-contrast visual PNG stream."""
    logger.info(f"🌌 Processing stream conversion pipeline for asset: {obs_id}")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT preview_url, target_name, product_filename FROM jwst_previews WHERE obs_id = ?",
        (obs_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Requested observation ID mapping not found in database cache.",
        )

    mast_url = row["preview_url"]
    target_name = row["target_name"]
    filename = row["product_filename"]

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f" └── Fetching binary cluster from MAST cloud: {filename}")
            response = await client.get(mast_url, timeout=60.0)
            if response.status_code != 200:
                logger.error(
                    f"❌ MAST endpoint rejected image download hook. Status: {response.status_code}"
                )
                raise HTTPException(
                    status_code=502,
                    detail="Unable to retrieve data matrix from upstream space servers.",
                )
        except httpx.RequestError as exc:
            logger.error(f"❌ Network failure communicating with MAST: {str(exc)}")
            raise HTTPException(
                status_code=504,
                detail="Upstream space archive network request timeout.",
            )

    try:
        fits_buffer = io.BytesIO(response.content)
        with fits.open(fits_buffer) as hdul:
            if "SCI" in hdul:
                raw_pixels = hdul["SCI"].data
            else:
                raw_pixels = (
                    hdul[1].data if len(hdul) > 1 else hdul[0].data
                )

            if raw_pixels is None or len(raw_pixels.shape) != 2:
                raise ValueError(
                    "Target array data mapping structure does not match a recognizable 2D shape framework."
                )

            processed_pixels = apply_astronomical_stretch(raw_pixels)

            output_image_buffer = io.BytesIO()
            img = Image.fromarray(processed_pixels, mode="L")
            img.save(output_image_buffer, format="PNG")
            output_image_buffer.seek(0)

    except Exception as err:
        logger.error(f"❌ Scientific matrix processing extraction failure: {str(err)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate matrix transformations on space asset data.",
        )

    headers = {}
    if download:
        safe_target_name = "".join(
            c for c in target_name if c.isalnum() or c in (" ", "_", "-")
        ).rstrip()
        headers["Content-Disposition"] = (
            f'attachment; filename="JWST_{safe_target_name}_{obs_id}.png"'
        )

    return StreamingResponse(
        output_image_buffer, media_type="image/png", headers=headers
    )