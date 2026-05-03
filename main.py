from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.health import router as health_router
from api.routes.analyze import router as analyze_router
from utils.error_handlers import register_exception_handlers

app = FastAPI(
    title="PlantCare AI",
    description="AI-powered plant identification and care guidance",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(analyze_router, prefix="/api/v1")
