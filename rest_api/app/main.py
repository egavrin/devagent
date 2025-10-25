"""Main FastAPI application."""
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from dotenv import load_dotenv

from app.routers import auth, users
from app.database import create_tables, get_db
from app.utils.security import get_password_hash
from app.models.user import User

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="REST API with Authentication",
    description="A complete REST API with user authentication, tests, and security checks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1", tags=["authentication"])
app.include_router(users.router, prefix="/api/v1", tags=["users"])

# Security scheme for OpenAPI
token_scheme = HTTPBearer()


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    create_tables()
    print("Database tables initialized")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "REST API with Authentication",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }


# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": {
            "message": exc.detail,
            "status_code": exc.status_code
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
