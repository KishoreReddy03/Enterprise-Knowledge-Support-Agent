import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

from config import settings
# These imports assume the rest of the structure is created
# from api.routes import chat, health, eval

# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stripe Knowledge Support Agent",
    description="Enterprise-grade AI support agent using RAG, Multi-Agent Orchestration, and Guardrails.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    settings.display_config()
    logger.info("Application starting up...")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app": "Stripe Knowledge Support Agent",
        "status": "online",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs"
    }

# Register routers (uncomment as files are created)
# app.include_router(health.router, prefix="/api/v1")
# app.include_router(chat.router, prefix="/api/v1")
# app.include_router(eval.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
