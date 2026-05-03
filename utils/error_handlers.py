from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai


class ImageValidationError(Exception):
    pass


class PlantAnalysisError(Exception):
    pass


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ImageValidationError)
    async def image_validation_handler(request: Request, exc: ImageValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_image", "detail": str(exc)},
        )

    @app.exception_handler(openai.APIError)
    async def openai_api_error_handler(request: Request, exc: openai.APIError):
        return JSONResponse(
            status_code=502,
            content={"error": "llm_unavailable", "detail": "OpenAI API error. Please try again."},
        )

    @app.exception_handler(openai.RateLimitError)
    async def rate_limit_handler(request: Request, exc: openai.RateLimitError):
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "detail": "Too many requests. Please wait a moment."},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "An unexpected error occurred."},
        )
