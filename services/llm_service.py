from __future__ import annotations
from typing import Optional
from openai import OpenAI
from models.plant_response import PlantAnalysisResponse
from config import get_settings

SYSTEM_PROMPT = """You are PlantCare AI, an expert botanist and plant health specialist with decades of experience.

Your role is to analyze plant images and provide accurate, actionable care guidance that helps plant owners keep their plants thriving.

EXPERTISE:
- Plant identification across thousands of species worldwide
- Diagnosing plant health issues from visual symptoms (color, texture, shape, spots, wilting)
- Providing practical, beginner-friendly care instructions
- Recommending urgent interventions when plants are in distress

ANALYSIS RULES:
1. Identify the plant species from the image. Provide your best identification with a confidence score (0.0–1.0).
2. If confidence is below 0.5, still provide your best guess — note the uncertainty with a lower score.
3. Never fabricate scientific names — use "Unknown species" if you genuinely cannot identify.
4. Assess visible health indicators: leaf color, texture, wilting, spots, pest signs, root exposure, pot condition.
5. Use the provided knowledge base context to enrich and validate your response.
6. Urgency levels:
   - low: plant appears healthy, minor improvements only
   - medium: noticeable issues, attention needed within 1-2 weeks
   - high: significant problems, act within a few days
   - critical: emergency care needed immediately to save the plant

RESPONSE QUALITY:
- Be specific and actionable — "water every 7-10 days" not just "water regularly"
- Fun facts should be genuinely interesting, not generic
- Care tips should be tailored to what you observe in the image
- If the user provided an observation note, incorporate it into your health assessment
"""

# Few-shot examples to guide the model on expected output format and tone
FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """Analyze this plant image and provide a complete assessment.

KNOWLEDGE BASE CONTEXT (use this to enrich your response):
Pothos (Epipremnum aureum) — Family: Araceae. Sunlight: Low to bright indirect light; avoid direct sun. Water: Every 1-2 weeks; allow top inch of soil to dry. Difficulty: Easy. Notes: Extremely hardy and fast-growing; propagates easily in water; trailing or climbing habit.
Overwatering is the most common cause of houseplant death. Signs include yellowing leaves (especially lower leaves), soft or mushy stems, soggy soil.

Respond strictly in the required JSON format."""
            }
        ]
    },
    {
        "role": "assistant",
        "content": """{
  "identification": {
    "common_name": "Pothos",
    "scientific_name": "Epipremnum aureum",
    "family": "Araceae",
    "confidence_score": 0.96
  },
  "sunlight": {
    "level": "Low to Bright Indirect Light",
    "hours_per_day": "2-8 hours indirect",
    "tips": "Keep near a window but out of direct sun — a north or east-facing windowsill is ideal."
  },
  "water": {
    "frequency": "Every 7-10 days",
    "amount": "Water until it flows from drainage holes, then discard excess",
    "tips": "Stick your finger 1 inch into the soil — only water when that inch is dry."
  },
  "health": {
    "status": "Healthy",
    "issues_detected": [],
    "improvement_tips": [
      "Rotate the pot 90° every two weeks for even, balanced growth",
      "Wipe leaves with a damp cloth monthly to keep them dust-free and photosynthesizing efficiently"
    ],
    "urgency": "low"
  },
  "fun_facts": [
    "Pothos can survive in water alone — simply place cuttings in a vase and they'll grow indefinitely",
    "NASA's Clean Air Study found Pothos effectively removes formaldehyde, xylene, and benzene from indoor air",
    "The name 'Devil's Ivy' comes from its ability to stay green even in near-darkness"
  ],
  "care_tips": [
    "Prune long trailing vines to encourage bushier, fuller growth",
    "Propagate easily by cutting below a node and placing in water for 2-3 weeks",
    "Yellowing lower leaves are usually natural aging — only worry if multiple leaves yellow at once"
  ],
  "rag_sources_used": ["Plant profile: Pothos", "Care guide: OVERWATERING — THE #1 KILLER OF HOUSEPLANTS"]
}"""
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """Analyze this plant image and provide a complete assessment.

KNOWLEDGE BASE CONTEXT (use this to enrich your response):
Monstera deliciosa — Family: Araceae. Sunlight: Bright indirect light. Water: Every 1-2 weeks; allow top 2 inches of soil to dry.
Symptom: Yellow leaves, especially lower leaves, with soggy or wet soil. Cause: Overwatering. Treatment: Stop watering immediately. Allow soil to dry completely. Check for root rot. Ensure pot has drainage holes. Urgency: medium.
Root rot: Remove from pot, trim all brown mushy roots, dust with cinnamon or sulfur, repot in fresh well-draining soil. Urgency: high.

User observation: The lower leaves are turning yellow and the soil feels very wet.

Respond strictly in the required JSON format."""
            }
        ]
    },
    {
        "role": "assistant",
        "content": """{
  "identification": {
    "common_name": "Monstera",
    "scientific_name": "Monstera deliciosa",
    "family": "Araceae",
    "confidence_score": 0.94
  },
  "sunlight": {
    "level": "Bright Indirect Light",
    "hours_per_day": "4-6 hours indirect",
    "tips": "Place 3-5 feet from a south or west-facing window — bright room but no direct rays on the leaves."
  },
  "water": {
    "frequency": "Every 10-14 days",
    "amount": "Water thoroughly until drainage occurs, then allow top 2 inches to dry before watering again",
    "tips": "Lift the pot — a lightweight pot means it needs water; heavy pot means wait. Always check before watering."
  },
  "health": {
    "status": "Overwatered / At Risk of Root Rot",
    "issues_detected": [
      "Multiple lower leaves yellowing — classic overwatering symptom",
      "Soil is reported as very wet, indicating waterlogging",
      "Risk of root rot if soggy conditions persist"
    ],
    "improvement_tips": [
      "Stop watering immediately — do not water again until the top 2 inches of soil are completely dry",
      "Move to a brighter, well-ventilated spot to help the soil dry faster",
      "Check drainage holes are clear and not blocked",
      "If yellowing spreads or stem base feels soft, unpot and inspect roots for rot"
    ],
    "urgency": "medium"
  },
  "fun_facts": [
    "Monstera leaves develop their iconic holes (fenestrations) naturally as the plant matures — young plants have solid leaves",
    "In the wild, Monstera uses these holes to allow wind and rainwater to pass through without damaging the leaf",
    "Monstera means 'monstrous' or 'abnormal' in Latin, referring to its dramatic perforated leaves"
  ],
  "care_tips": [
    "Give it a moss pole or trellis to climb — plants that climb produce larger, more fenestrated leaves",
    "Wipe large leaves with a damp cloth every few weeks to keep them clean and maximize light absorption",
    "Aerial roots are natural and healthy — you can guide them into the soil or simply leave them"
  ],
  "rag_sources_used": ["Plant profile: Monstera", "Disease info: Overwatering", "Disease info: Root rot caused by Pythium or Phytophthora fungi"]
}"""
    }
]


def identify_plant_name(image_base64: str, openai_client: OpenAI) -> str:
    """Quick lightweight call to get plant name for RAG query — no structured output needed."""
    settings = get_settings()
    response = openai_client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What plant is in this image? Reply with just the common name, nothing else. If unknown, reply 'unknown plant'."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()


def build_user_message(image_base64: str, rag_context: list[str], user_note: Optional[str]) -> dict:
    """Construct the user message with RAG context injected."""
    context_block = "\n".join(rag_context) if rag_context else "No additional context available."
    note_block = f"\nUser observation: {user_note}" if user_note else ""

    text_content = f"""Analyze this plant image and provide a complete assessment.

KNOWLEDGE BASE CONTEXT (use this to enrich your response):
{context_block}
{note_block}

Respond strictly in the required JSON format."""

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text_content},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "high"
                }
            }
        ]
    }


def analyze_plant(
    image_base64: str,
    rag_context: list[str],
    rag_sources: list[str],
    user_note: Optional[str],
    openai_client: OpenAI
) -> PlantAnalysisResponse:
    """Main structured output call — returns a fully parsed PlantAnalysisResponse."""
    settings = get_settings()

    user_message = build_user_message(image_base64, rag_context, user_note)

    response = openai_client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_MESSAGES,
            user_message
        ],
        response_format=PlantAnalysisResponse,
        max_tokens=1500
    )

    result: PlantAnalysisResponse = response.choices[0].message.parsed
    result.rag_sources_used = rag_sources
    return result
