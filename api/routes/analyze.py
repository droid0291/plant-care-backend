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
    try:
        processed_image = resize_and_encode(request.image_base64)
    except ValueError as e:
        raise ImageValidationError(str(e))

    cached = cache_service.get_cached(processed_image)
    if cached:
        logger.info("Cache hit — returning cached result")
        return cached

    # Step 1: Identify plant name from image (lightweight vision call)
    logger.info("Identifying plant...")
    plant_name = llm_service.identify_plant_name(processed_image, openai_client)
    logger.info(f"Identified: {plant_name}")

    # Step 2: Look up plant in local knowledge base (no API call)
    kb_plant = rag_service.lookup_plant_by_name(plant_name)

    try:
        if kb_plant:
            # Fast path: plant is in KB — only ask GPT-4o for health + fun facts
            logger.info(f"KB hit for '{plant_name}' — using KB care data, assessing health with GPT-4o")
            health_result = llm_service.assess_health_only(
                image_base64=processed_image,
                plant_name=plant_name,
                user_note=request.user_note,
                openai_client=openai_client,
            )
            result = rag_service.build_response_from_kb(kb_plant, health_result)
        else:
            # Fallback: unknown plant — GPT-4o generates everything, RAG provides context
            logger.info(f"KB miss for '{plant_name}' — running full GPT-4o analysis")
            rag_chunks, rag_sources = rag_service.query_knowledge_base(
                f"Care requirements, health issues, and diseases for {plant_name}",
                openai_client,
            )
            logger.info(f"RAG returned {len(rag_chunks)} relevant chunks")
            result = llm_service.analyze_plant(
                image_base64=processed_image,
                rag_context=rag_chunks,
                rag_sources=rag_sources,
                user_note=request.user_note,
                openai_client=openai_client,
            )
    except ValueError as e:
        raise ImageValidationError(str(e))

    logger.info(
        f"Analysis complete: {result.identification.common_name} "
        f"({result.identification.confidence_score:.0%} confidence) | "
        f"{'KB' if kb_plant else 'GPT-4o'} care data"
    )

    cache_service.store_cached(processed_image, result)
    return result