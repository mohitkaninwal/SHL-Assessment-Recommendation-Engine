"""
FastAPI Application for SHL Assessment Recommendation System
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
recommendation_engine: Optional[RecommendationEngine] = None


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


class AssessmentRecommendation(BaseModel):
    """Model for a single assessment recommendation"""
    assessment_name: str = Field(..., description="Name of the assessment")
    assessment_url: str = Field(..., description="URL to the assessment")
    test_type: str = Field(..., description="Type of test (K=Knowledge, P=Personality, etc.)")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    description: Optional[str] = Field(None, description="Assessment description")
    category: Optional[str] = Field(None, description="Assessment category")


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint"""
    query: str = Field(..., description="Original query")
    recommendations: List[AssessmentRecommendation] = Field(
        ...,
        description="List of recommended assessments"
    )
    total_found: int = Field(..., description="Total number of assessments found")
    returned: int = Field(..., description="Number of recommendations returned")
    explanation: Optional[str] = Field(None, description="Explanation for recommendations")
    query_analysis: Optional[Dict[str, Any]] = Field(None, description="Query analysis details")
    timestamp: str = Field(..., description="Response timestamp")


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


# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize recommendation engine on startup"""
    global recommendation_engine
    
    logger.info("Starting SHL Assessment Recommendation API...")
    
    try:
        # Initialize recommendation engine
        logger.info("Initializing recommendation engine...")
        recommendation_engine = RecommendationEngine(
            use_rag=True,
            use_llm_reranking=True,
            use_query_expansion=False
        )
        logger.info("✓ Recommendation engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}", exc_info=True)
        # Don't fail startup, but log the error
        recommendation_engine = None


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
    engine_status = "healthy" if recommendation_engine is not None else "unavailable"
    
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
    # Check if engine is available
    if recommendation_engine is None:
        logger.error("Recommendation engine is not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine is not available. Please try again later."
        )
    
    try:
        logger.info(f"Processing recommendation request: query='{request.query[:50]}...', top_k={request.top_k}")
        
        # Get recommendations
        result = recommendation_engine.recommend(
            query=request.query,
            top_k=request.top_k,
            balance_skills=request.balance_skills,
            include_explanation=request.include_explanation
        )
        
        # Convert to response model
        recommendations = [
            AssessmentRecommendation(**rec)
            for rec in result['recommendations']
        ]
        
        response = RecommendationResponse(
            query=result['query'],
            recommendations=recommendations,
            total_found=result['total_found'],
            returned=result['returned'],
            explanation=result.get('explanation'),
            query_analysis=result.get('query_analysis'),
            timestamp=datetime.now().isoformat()
        )
        
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
