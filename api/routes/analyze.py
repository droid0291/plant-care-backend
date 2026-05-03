import logging
from fastapi import APIRouter, Depends
from openai import OpenAI

from models.plant_response import AnalyzeRequest, PlantAnalysisResponse
from services import cache_service, rag_service, llm_service
from utils.image_utils import resize_and_encode
from utils.error_handlers import ImageValidationError
from config import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()


def get_openai_client(settings: Settings = Depends(get_settings)) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@router.post("/analyze", response_model=PlantAnalysisResponse)
async def analyze_plant(
    request: AnalyzeRequest,
    openai_client: OpenAI = Depends(get_openai_client)
) -> PlantAnalysisResponse:
    # Validate and resize image
    try:
        processed_image = resize_and_encode(request.image_base64)
    except ValueError as e:
        raise ImageValidationError(str(e))

    # Check cache first
    cached = cache_service.get_cached(processed_image)
    if cached:
        logger.info("Cache hit — returning cached result")
        return cached

    # Step 1: Quick plant name identification for RAG query
    logger.info("Identifying plant for RAG query...")
    plant_name = llm_service.identify_plant_name(processed_image, openai_client)
    rag_query = f"Care requirements, health issues, and diseases for {plant_name}"
    logger.info(f"RAG query: {rag_query}")

    # Step 2: Retrieve relevant knowledge base chunks
    rag_chunks = rag_service.query_plant_knowledge(rag_query, openai_client)
    rag_sources = rag_service.get_rag_source_labels(rag_query, openai_client)
    logger.info(f"RAG returned {len(rag_chunks)} relevant chunks")

    # Step 3: Full structured analysis with GPT-4o vision
    logger.info("Running full plant analysis with GPT-4o...")
    result = llm_service.analyze_plant(
        image_base64=processed_image,
        rag_context=rag_chunks,
        rag_sources=rag_sources,
        user_note=request.user_note,
        openai_client=openai_client
    )
    logger.info(f"Analysis complete: {result.identification.common_name} ({result.identification.confidence_score:.0%} confidence)")

    # Step 4: Cache result
    cache_service.store_cached(processed_image, result)

    return result
