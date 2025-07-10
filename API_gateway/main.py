# API Gateway Module - Modular Implementation

## 1. `main.py`
import logging
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseSettings
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import httpx
from .utils import settings, verify_jwt
from .middlewares import RateLimitMiddleware, REQUEST_COUNT, REQUEST_LATENCY

# Initialize tracer and metrics
trace.set_tracer_provider(trace.TracerProvider())
tracer = trace.get_tracer(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-gateway")

# FastAPI app
app = FastAPI(title="DeepSeek API Gateway", version="1.0.0")
FastAPIInstrumentor.instrument_app(app)
app.add_middleware(RateLimitMiddleware)

# Health-check endpoints
@app.get('/health/live')
async def liveness():
    return PlainTextResponse("OK")

@app.get('/health/ready')
async def readiness():
    return PlainTextResponse("OK")

# Metrics endpoint
@app.get('/metrics')
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Core chat endpoint
@app.post('/chat', dependencies=[Depends(verify_jwt)])
async def chat_gateway(payload: dict, request: Request):
    text = payload.get('message') or payload.get('text')
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'message' or 'text'")

    async with httpx.AsyncClient(timeout=10.0) as client:
        with tracer.start_as_current_span("nlu_request"):
            nlu_resp = await client.post(f"{settings.SERVICE_MAP['nlu']}/parse", json={"text": text})
            if nlu_resp.status_code != 200:
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="NLU service error")
            parsed = nlu_resp.json()

        with tracer.start_as_current_span("rag_request"):
            rag_resp = await client.post(f"{settings.SERVICE_MAP['rag']}/query", json=parsed)
            if rag_resp.status_code != 200:
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="RAG service error")
            answer = rag_resp.json().get('answer')

    return JSONResponse({"reply": answer})

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


## 2. `utils.py`

import os
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseSettings
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

class Settings(BaseSettings):
    SERVICE_MAP: dict = {
        'auth': os.getenv('AUTH_URL', 'http://auth-service'),
        'conversation': os.getenv('CONVERSATION_URL', 'http://conversation-service'),
        'nlu': os.getenv('NLU_URL', 'http://nlu-service'),
        'rag': os.getenv('RAG_URL', 'http://rag-service'),
        'metadata': os.getenv('METADATA_URL', 'http://metadata-service'),
    }
    JWT_SECRET: str = os.getenv('JWT_SECRET', 'supersecret')
    RATE_LIMIT_CALLS: int = int(os.getenv('RATE_LIMIT_CALLS', 100))
    RATE_LIMIT_PERIOD: int = int(os.getenv('RATE_LIMIT_PERIOD', 60))

settings = Settings()

# JWT Bearer scheme
bearer_scheme = HTTPBearer()

async def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    # JWT decode validation omitted for brevity
    return token

# Prometheus metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_gateway_request_latency_seconds', 'Latency of requests', ['method', 'endpoint'])


## 3. `middlewares.py`

import time
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ratelimit import limits, RateLimitException
from .utils import settings, REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger("api-gateway.middleware")

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            limits(calls=settings.RATE_LIMIT_CALLS, period=settings.RATE_LIMIT_PERIOD)(lambda: None)()
        except RateLimitException:
            return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                 content={"detail": "Rate limit exceeded"})

        start = time.time()
        response = await call_next(request)
        latency = time.time() - start
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(latency)
        REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
        logger.info(f"{request.method} {request.url.path} {response.status_code} in {latency:.3f}s")
        return response

