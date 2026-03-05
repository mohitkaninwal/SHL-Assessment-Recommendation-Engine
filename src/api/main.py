"""
FastAPI Application for SHL Assessment Recommendation System
"""

import os
import sys
import logging
import asyncio
import re
import json
import requests
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from bs4 import BeautifulSoup
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.recommendation.query_guardrails import QueryGuardrails

if TYPE_CHECKING:
    from src.recommendation.recommend import RecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="AI-powered recommendation system for SHL assessments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommendation engine instance
recommendation_engine: Optional["RecommendationEngine"] = None
engine_init_lock = asyncio.Lock()


# Request/Response Models
class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint"""
    query: str = Field(
        ...,
        description="Natural language query or job description",
        min_length=1,
        max_length=5000,
        example="I am hiring for Java developers who can also collaborate effectively"
    )
    top_k: int = Field(
        default=10,
        description="Number of recommendations to return",
        ge=1,
        le=20,
        example=10
    )
    balance_skills: bool = Field(
        default=True,
        description="Whether to balance hard and soft skills",
        example=True
    )
    include_explanation: bool = Field(
        default=False,
        description="Whether to include explanation for recommendations",
        example=False
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty or just whitespace"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class RecommendedAssessment(BaseModel):
    """Public model for one recommended assessment."""
    url: str = Field(..., description="URL to the assessment")
    name: str = Field(..., description="Assessment name")
    adaptive_support: str = Field(..., description="Whether adaptive testing is supported")
    description: str = Field(..., description="Assessment description")
    duration: int = Field(..., description="Approximate duration in minutes")
    remote_support: str = Field(..., description="Whether remote testing is supported")
    test_type: List[str] = Field(..., description="Assessment test type labels")


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint."""
    recommended_assessments: List[RecommendedAssessment] = Field(
        ...,
        description="List of recommended assessments"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Health status", example="healthy")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version", example="1.0.0")
    engine_status: str = Field(..., description="Recommendation engine status")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")


TEST_TYPE_LABELS = {
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "A": "Ability & Aptitude",
    "D": "Development & 360",
    "S": "Simulations",
    "E": "Assessment Exercises",
}


def _to_yes_no(value: Any, default: str) -> str:
    """Normalize different truthy/falsy representations to Yes/No."""
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return "Yes" if value else "No"
    value_str = str(value).strip().lower()
    if value_str in {"yes", "y", "true", "1"}:
        return "Yes"
    if value_str in {"no", "n", "false", "0"}:
        return "No"
    return default


def _normalize_test_type_labels(raw_value: Any) -> List[str]:
    """Convert test type codes/labels into a normalized list of readable labels."""
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        tokens = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        raw_text = str(raw_value).strip()
        if not raw_text:
            return []
        tokens = [token for token in re.split(r"[,\s/|]+", raw_text) if token]

    labels: List[str] = []
    for token in tokens:
        normalized_token = token.upper()
        label = TEST_TYPE_LABELS.get(normalized_token, token.strip())
        if label not in labels:
            labels.append(label)

    return labels


def _format_recommended_assessment(rec: Dict[str, Any]) -> RecommendedAssessment:
    """Map internal recommendation fields to public /recommend response schema."""
    rec = _merge_with_catalog_details(rec)
    url = rec.get("url") or rec.get("assessment_url") or ""
    name = rec.get("name") or rec.get("assessment_name") or "Unknown"
    description = (rec.get("description") or "").strip()
    duration = rec.get("duration")
    try:
        duration = int(duration) if duration is not None else 0
    except (TypeError, ValueError):
        duration = 0

    # Prefer full test-type coverage from catalog when available.
    test_type_raw = rec.get("all_test_types")
    if test_type_raw in (None, "", []):
        test_type_raw = rec.get("test_type")

    return RecommendedAssessment(
        url=url,
        name=name,
        adaptive_support=_to_yes_no(rec.get("adaptive_support"), default="No"),
        description=description,
        duration=duration,
        remote_support=_to_yes_no(rec.get("remote_support"), default="Yes"),
        test_type=_normalize_test_type_labels(test_type_raw),
    )


URL_PATTERN = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", flags=re.IGNORECASE)
JD_SECTION_PATTERNS = {
    "responsibilities": re.compile(
        r"^(what you will be doing|responsibilities|role summary|job responsibilities)\b",
        re.IGNORECASE,
    ),
    "essential": re.compile(r"^(essential|required|required skills|must have)\b", re.IGNORECASE),
    "desirable": re.compile(r"^(desirable|preferred|nice to have)\b", re.IGNORECASE),
}
JD_NOISE_PATTERNS = [
    re.compile(r"\bjoin a community\b", re.IGNORECASE),
    re.compile(r"\bshaping the future of work\b", re.IGNORECASE),
    re.compile(r"\bexcellent benefits package\b", re.IGNORECASE),
    re.compile(r"\bcareer development\b", re.IGNORECASE),
    re.compile(r"\bdiversity\b", re.IGNORECASE),
    re.compile(r"\binclusivity\b", re.IGNORECASE),
    re.compile(r"\bno better time to become a part\b", re.IGNORECASE),
]
JD_SIGNAL_PATTERN = re.compile(
    r"\b(ai|ml|machine learning|nlp|computer vision|llm|rag|python|tensorflow|pytorch|openai|"
    r"deployment|model|monitoring|scalability|performance|collaboration|teamwork|stakeholder|"
    r"communication|agile|research|engineering)\b",
    re.IGNORECASE,
)


def _extract_text_from_html(html: str) -> str:
    """Extract readable job description text from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header", "aside"]):
        tag.decompose()

    chunks: List[str] = []

    title = soup.find("title")
    if title:
        title_text = title.get_text(" ", strip=True)
        if title_text:
            chunks.append(title_text)

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        chunks.append(str(meta_desc.get("content")).strip())

    for node in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = node.get_text(" ", strip=True)
        if node.name in {"h1", "h2", "h3"} and len(text) >= 3:
            chunks.append(text)
        elif len(text) >= 20:
            chunks.append(text)

    text = " ".join(chunks)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_jd_text(text: str, max_chars: int = 3500) -> str:
    """
    Convert raw JD text into a concise, skill-focused query representation.
    Keeps role, responsibilities, essential/desirable skills; removes employer fluff.
    """
    if not text:
        return ""

    normalized = text.replace("\u202f", " ").replace("\xa0", " ")
    lines = [line.strip(" \t•-*") for line in normalized.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return " ".join(normalized.split())[:max_chars]

    role_line = ""
    section = None
    responsibilities: List[str] = []
    essential: List[str] = []
    desirable: List[str] = []
    signal_fallback: List[str] = []

    def _append_unique(container: List[str], value: str) -> None:
        value_clean = " ".join(value.split()).strip()
        if not value_clean:
            return
        if any(p.search(value_clean) for p in JD_NOISE_PATTERNS):
            return
        if value_clean.lower() not in {v.lower() for v in container}:
            container.append(value_clean)

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        lower = line.lower()

        if not role_line and (
            "engineer" in lower
            or "developer" in lower
            or "scientist" in lower
            or "analyst" in lower
            or "manager" in lower
            or lower.startswith("job description")
        ):
            role_line = line

        matched_section = None
        for sec_name, sec_pattern in JD_SECTION_PATTERNS.items():
            if sec_pattern.search(line):
                matched_section = sec_name
                break
        if matched_section:
            section = matched_section
            continue

        if any(p.search(line) for p in JD_NOISE_PATTERNS):
            continue

        if section == "responsibilities":
            _append_unique(responsibilities, line)
        elif section == "essential":
            _append_unique(essential, line)
        elif section == "desirable":
            _append_unique(desirable, line)
        elif JD_SIGNAL_PATTERN.search(line):
            _append_unique(signal_fallback, line)

    chunks: List[str] = []
    if role_line:
        chunks.append(f"Role: {role_line}")
    if responsibilities:
        chunks.append("Responsibilities: " + "; ".join(responsibilities[:12]))
    if essential:
        chunks.append("Required skills: " + "; ".join(essential[:12]))
    if desirable:
        chunks.append("Preferred skills: " + "; ".join(desirable[:8]))
    if not chunks and signal_fallback:
        chunks.append("Job requirements: " + "; ".join(signal_fallback[:16]))

    final_text = " ".join(chunks).strip()
    if not final_text:
        final_text = " ".join(normalized.split())

    return final_text[:max_chars]


def _fetch_job_description_from_url(url: str, timeout_seconds: int = 20) -> str:
    """Fetch and normalize JD text from a URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, timeout=timeout_seconds, headers=headers, allow_redirects=True)
    response.raise_for_status()

    content_type = (response.headers.get("content-type") or "").lower()
    if "text/html" in content_type or "<html" in response.text.lower():
        extracted = _extract_text_from_html(response.text)
    else:
        extracted = re.sub(r"\s+", " ", response.text).strip()

    if not extracted or len(extracted) < 40:
        raise ValueError("Could not extract enough job description content from the provided URL.")

    return extracted


def _normalize_query_input(raw_query: str) -> str:
    """
    Normalize recommendation input.

    Supports:
    - Natural-language queries
    - Raw JD text
    - URLs containing a JD (URL-only or mixed text + URL)
    """
    query = (raw_query or "").strip()
    if not query:
        return ""

    url = _extract_first_url(query)
    if not url:
        return _normalize_jd_text(query, max_chars=5000)

    url_context = query.replace(url, " ").strip()
    jd_text = _fetch_job_description_from_url(url)
    jd_text = _normalize_jd_text(jd_text, max_chars=4500)

    if url_context:
        url_context = _normalize_jd_text(url_context, max_chars=800)
        combined = f"{url_context}\n\nJob Description:\n{jd_text}"
    else:
        combined = jd_text

    # Keep payload bounded for downstream validation/model costs.
    return combined[:5000].strip()


def _extract_first_url(text: str) -> Optional[str]:
    """Extract first URL-like token and normalize scheme/punctuation."""
    if not text:
        return None

    match = URL_PATTERN.search(text)
    if not match:
        return None

    candidate = match.group(0).strip()
    candidate = candidate.rstrip(").,;:!?]}>\"'")

    if candidate.lower().startswith("www."):
        candidate = f"https://{candidate}"

    if not re.match(r"^https?://", candidate, flags=re.IGNORECASE):
        return None
    return candidate


@lru_cache(maxsize=1)
def _load_catalog_lookup() -> Dict[str, Dict[str, Any]]:
    """Load processed catalog once and build URL -> assessment lookup for backfills."""
    catalog_path = Path("data/processed_catalog.json")
    if not catalog_path.exists():
        return {}

    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lookup: Dict[str, Dict[str, Any]] = {}
        for item in data.get("assessments", []):
            url = str(item.get("url") or "").strip()
            if url:
                lookup[url] = item
                canonical = _canonicalize_url(url)
                if canonical:
                    lookup[canonical] = item
        return lookup
    except Exception:
        return {}


def _merge_with_catalog_details(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill missing recommendation fields from processed catalog by URL."""
    merged = dict(rec)
    url = str(merged.get("url") or merged.get("assessment_url") or "").strip()
    if not url:
        return merged

    catalog_item = _load_catalog_lookup().get(url)
    if not catalog_item:
        catalog_item = _load_catalog_lookup().get(_canonicalize_url(url))
    if not catalog_item:
        return merged

    for key in ["description", "duration", "remote_support", "adaptive_support", "all_test_types", "test_type"]:
        if merged.get(key) in (None, "", []):
            merged[key] = catalog_item.get(key)

    if not merged.get("name") and catalog_item.get("name"):
        merged["name"] = catalog_item.get("name")
    if not merged.get("assessment_name") and catalog_item.get("name"):
        merged["assessment_name"] = catalog_item.get("name")
    if not merged.get("url") and catalog_item.get("url"):
        merged["url"] = catalog_item.get("url")
    if not merged.get("assessment_url") and catalog_item.get("url"):
        merged["assessment_url"] = catalog_item.get("url")

    return merged


def _canonicalize_url(url: str) -> str:
    """Normalize URLs for stable catalog lookup (host/path only, no query/fragment)."""
    try:
        parsed = urlsplit(url.strip())
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        path = "/" + path.lstrip("/")
        path = path.rstrip("/") + "/"
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return url.strip()


# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Startup hook (engine is initialized lazily on first /recommend request)."""
    logger.info("Starting SHL Assessment Recommendation API...")


async def get_or_init_engine() -> Optional["RecommendationEngine"]:
    """Initialize recommendation engine once, lazily, to reduce cold-start memory spikes."""
    global recommendation_engine
    if recommendation_engine is not None:
        return recommendation_engine

    async with engine_init_lock:
        if recommendation_engine is not None:
            return recommendation_engine
        try:
            # Lazy import keeps API /health up even if recommendation deps are misconfigured.
            from src.recommendation.recommend import RecommendationEngine

            logger.info("Initializing recommendation engine (lazy)...")
            recommendation_engine = RecommendationEngine(
                use_rag=True,
                use_llm_reranking=True,
                use_query_expansion=False
            )
            logger.info("✓ Recommendation engine initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize recommendation engine: %s", e, exc_info=True)
            recommendation_engine = None
        return recommendation_engine


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down SHL Assessment Recommendation API...")


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"Status={response.status_code} Duration={duration:.3f}s"
    )
    
    return response


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "description": "AI-powered recommendation system for SHL assessments",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check if the API and recommendation engine are healthy"
)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status with timestamp
    """
    engine_status = "healthy" if recommendation_engine is not None else "not_initialized"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        engine_status=engine_status
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get Assessment Recommendations",
    description="Get personalized assessment recommendations based on a query or job description",
    responses={
        200: {"description": "Successful recommendation"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    }
)
async def get_recommendations(request: RecommendationRequest):
    """
    Get assessment recommendations
    
    Args:
        request: Recommendation request with query and parameters
        
    Returns:
        List of recommended assessments with metadata
        
    Raises:
        HTTPException: If recommendation engine is unavailable or request fails
    """
    engine = await get_or_init_engine()

    # Check if engine is available
    if engine is None:
        logger.error("Recommendation engine is not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine is not available. Please try again later."
        )

    try:
        normalized_query = _normalize_query_input(request.query)
    except requests.RequestException as e:
        logger.warning("Failed to fetch JD URL from query '%s': %s", request.query[:120], e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to fetch job description from the provided URL."
        )
    except ValueError as e:
        logger.warning("Invalid JD URL content for query '%s': %s", request.query[:120], e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Guardrail: block unclear/nonsensical prompts before LLM invocation.
    validation = QueryGuardrails.validate(normalized_query)
    if not validation.is_valid:
        logger.warning("Rejected unclear query via guardrails: '%s'", normalized_query[:80])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation.message
        )
    
    try:
        logger.info(f"Processing recommendation request: query='{normalized_query[:50]}...', top_k={request.top_k}")
        
        # Get recommendations
        result = engine.recommend(
            query=normalized_query,
            top_k=request.top_k,
            balance_skills=request.balance_skills,
            include_explanation=request.include_explanation
        )
        
        # Convert to externally expected response shape.
        recommendations = [
            _format_recommended_assessment(rec)
            for rec in result['recommendations']
        ]
        response = RecommendationResponse(recommended_assessments=recommendations)
        
        logger.info(f"Successfully generated {len(recommendations)} recommendations")
        
        return response
        
    except ValueError as e:
        # Validation error
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    except Exception as e:
        # Internal error
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating recommendations. Please try again."
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG") else None,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# Run server
def main():
    """Run the API server"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
