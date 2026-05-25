import os
import io
import logging
import sqlite3
from typing import Optional, Dict, Any, List
import numpy as np
from astropy.io import fits
from PIL import Image
import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

# Configure clean, highly structured logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("JWST-Chromatic-Backend")

#DB_NAME = "jwst_archive.db"
# 1. Get the absolute path of the directory containing server.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Dynamically look for the database file in that same directory
DB_NAME = os.path.join(BASE_DIR, "jwst_archive.db")

app = FastAPI(
    title="JWST Chromatic Translation Discovery Engine",
    description="A non-blocking API featuring multi-column text lookups, explicit filter tracking, representative color vector data streaming, and live on-the-fly FITS processing paths.",
    version="1.5.1"
)

# FORCE it to look at your singular folder name
templates = Jinja2Templates(directory="template")

# Enable CORS so your frontend layout (React/Next.js/HTML5) can communicate smoothly with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains in production environments
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
    """
    Transforms a high-dynamic-range scientific float matrix into a display-ready 8-bit integer array.
    Uses percentile clipping to drop noise spikes and log-scaling to enhance faint gas structure.
    """
    # 1. Neutralize bad pixel dropouts, NaNs, and instrument infinity anomalies
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Extract boundaries eliminating top and bottom outliers (1st to 99.5th percentile)
    # This ensures a single ultra-bright star doesn't turn the rest of a nebula pitch black
    vmin, vmax = np.percentile(image_data, [1.0, 99.5])
    clipped = np.clip(image_data, vmin, vmax)
    
    # 3. Execute Logarithmic non-linear stretching 
    # This amplifies faint plasma clouds while smoothly compressing bright stellar cores
    stretched = np.log1p(clipped - vmin)
    
    # 4. Map cleanly across a standard 0-255 viewport scale
    span = stretched.max() - stretched.min()
    if span == 0:
        span = 1e-8  # Safe protection fallback against division by zero on blank frames
        
    normalized = (stretched - stretched.min()) / span
    return (normalized * 255).astype(np.uint8)


@app.get("/api/obs-inventory", response_model=List[Dict[str, Any]])
def get_obs_inventory():
    """
    A lightweight metadata-mapping utility endpoint.
    Returns a list of all unique obs_ids paired with target names and titles
    to cleanly populate multi-select dropdown picker frames on the frontend.
    """
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
        raise HTTPException(status_code=500, detail="Internal metadata inventory lookup failure.")
    finally:
        conn.close()


@app.get("/api/observations", response_model=Dict[str, Any])
def get_observations(
    search: Optional[str] = Query(None, description="Fuzzy search term scanning name, classification, filter labels, and impact descriptions simultaneously"),
    obs_id: Optional[List[str]] = Query(None, description="A multi-select array list of absolute observation identifiers to strictly isolate entries"),
    instrument: Optional[str] = Query(None, description="Filter by instrument setup (NIRCAM, MIRI, NIRISS)"),
    limit: int = Query(20, ge=1, le=100, description="Number of results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset index")
):
    """
    Queries the local SQLite database cache to supply your Material Design grid layout 
    with dynamic filtering tracking multi-select lists, cross-column search tokens, or instrument sets.
    Now includes color matrix indicator attributes.
    """
    logger.info(f"Metadata fetch requested. Filters -> Search: {search}, Multi-Select IDs: {obs_id}, Instrument: {instrument}")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Base query string extended to include chromatic translation truth variables
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
            query += f" AND obs_id IN ({placeholders})"  # <-- FIX: Typo successfully resolved here
            params.extend(clean_ids)
        
    # 2. Four-Dimensional Fuzzy Text Search Mapping
    if search:
        query += """ AND (
            target_name LIKE ? 
            OR target_classification LIKE ? 
            OR filter_human_label LIKE ? 
            OR impact_description LIKE ?
        )"""
        wildcard_search = f"%{search.strip()}%"
        params.extend([wildcard_search, wildcard_search, wildcard_search, wildcard_search])
        
    # 3. Evaluate Instrument Filters
    if instrument:
        query += " AND instrument_name LIKE ?"
        params.append(f"%{instrument.upper().strip()}%")
        
    # Append sorting, limits, and pagination variables
    query += " ORDER BY target_name ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Format database rows into native JSON-serializable list configurations
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
        if search:
            count_query += """ AND (
                target_name LIKE ? 
                OR target_classification LIKE ? 
                OR filter_human_label LIKE ? 
                OR impact_description LIKE ?
            )"""
            count_params.extend([wildcard_search, wildcard_search, wildcard_search, wildcard_search])
        if instrument:
            count_query += " AND instrument_name LIKE ?"
            count_params.append(f"%{instrument.upper().strip()}%")
            
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        return {
            "total_records": total_count,
            "limit": limit,
            "offset": offset,
            "results": records
        }
    except sqlite3.Error as e:
        logger.error(f"Database read failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal metadata engine repository error.")
    finally:
        conn.close()


@app.get("/api/stream/{obs_id}")
async def stream_observation_image(obs_id: str, download: bool = Query(False)):
    """
    The Core Live Rendering Engine. 
    Intercepts an observation ID, fetches the scientific FITS stream from MAST,
    processes it in RAM, and pipes back an ephemeral, high-contrast visual PNG stream.
    """
    logger.info(f"🌌 Processing stream conversion pipeline for asset: {obs_id}")
    
    # Step 1: Look up data path mappings from the local database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT preview_url, target_name, product_filename FROM jwst_previews WHERE obs_id = ?", (obs_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Requested observation ID mapping not found in database cache.")
        
    mast_url = row["preview_url"]
    target_name = row["target_name"]
    filename = row["product_filename"]
    
    # Step 2: Stream the scientific FITS binary payload from NASA MAST servers directly into memory
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f" └── Fetching binary cluster from MAST cloud: {filename}")
            # Request payload stream with an active 60 second network limit gateway
            response = await client.get(mast_url, timeout=60.0)
            
            if response.status_code != 200:
                logger.error(f"❌ MAST endpoint rejected image download hook. Status: {response.status_code}")
                raise HTTPException(status_code=502, detail="Unable to retrieve data matrix from upstream space servers.")
                
        except httpx.RequestError as exc:
            logger.error(f"❌ Network failure communicating with MAST: {str(exc)}")
            raise HTTPException(status_code=504, detail="Upstream space archive network request timeout.")

    # Step 3: Parse FITS structures out of the memory buffer
    try:
        fits_buffer = io.BytesIO(response.content)
        
        with fits.open(fits_buffer) as hdul:
            # Isolate the core scientific 2D matrix layer array
            if "SCI" in hdul:
                raw_pixels = hdul["SCI"].data
            else:
                raw_pixels = hdul[1].data if len(hdul) > 1 else hdul[0].data
                
            if raw_pixels is None or len(raw_pixels.shape) != 2:
                raise ValueError("Target array data mapping structure does not match a recognizable 2D shape framework.")
                
            # Step 4: Execute our matrix math contrast-stretching functions
            processed_pixels = apply_astronomical_stretch(raw_pixels)
            
            # Step 5: Serialize the processed NumPy grid array into a compressed PNG byte stream in RAM
            output_image_buffer = io.BytesIO()
            img = Image.fromarray(processed_pixels, mode="L")  # Outfitted as a single channel high-depth grayscale image
            img.save(output_image_buffer, format="PNG")
            output_image_buffer.seek(0)
            
    except Exception as err:
        logger.error(f"❌ Scientific matrix processing extraction failure: {str(err)}")
        raise HTTPException(status_code=500, detail="Failed to calculate matrix transformations on space asset data.")

    # Step 6: Formulate response payload headers
    headers = {}
    if download:
        # If frontend appends ?download=true, force the browser to execute a file save routine
        safe_target_name = "".join(c for c in target_name if c.isalnum() or c in (" ", "_", "-")).rstrip()
        headers["Content-Disposition"] = f'attachment; filename="JWST_{safe_target_name}_{obs_id}.png"'
    
    return StreamingResponse(
        output_image_buffer,
        media_type="image/png",
        headers=headers
    )