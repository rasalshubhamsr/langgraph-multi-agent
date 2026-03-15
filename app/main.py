from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.routers.analyse import router as analyse_router
from app.models.schemas import HealthResponse
from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LangGraph Multi-Agent API...")
    logger.info(f"Model: {settings.llm_model}")
    logger.info(f"Max reflection loops: {settings.max_reflection_loops}")
    logger.info(f"LangSmith tracing: {settings.langchain_tracing_v2}")
    yield
    logger.info("Shutdown complete.")


app = FastAPI(
    title="LangGraph Multi-Agent Credit Analyser",
    description=(
        "Autonomous multi-agent credit analysis pipeline using LangGraph. "
        "Features LLM self-reflection loops for hallucination reduction, "
        "LangSmith observability, and structured risk assessment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyse_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy",
        langsmith_enabled=settings.langchain_tracing_v2,
        model=settings.llm_model,
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "langgraph-multi-agent",
        "docs": "/docs",
        "health": "/health",
        "github": "https://github.com/rasalshubhamsr/langgraph-multi-agent",
    }
