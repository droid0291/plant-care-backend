from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SunlightRequirement(BaseModel):
    level: str = Field(description="e.g. 'Full Sun', 'Partial Shade', 'Low Light'")
    hours_per_day: str = Field(description="e.g. '6-8 hours' or 'Indirect only'")
    tips: str = Field(description="One actionable placement tip")


class WaterRequirement(BaseModel):
    frequency: str = Field(description="e.g. 'Every 7-10 days' or 'Keep soil moist'")
    amount: str = Field(description="e.g. 'Until water drains from pot'")
    tips: str = Field(description="One actionable watering tip")


class HealthAssessment(BaseModel):
    status: str = Field(description="Current health state: Healthy, Stressed, Diseased, Overwatered, etc.")
    issues_detected: list[str] = Field(description="List of observed problems, empty list if healthy")
    improvement_tips: list[str] = Field(description="Concrete steps to improve plant health")
    urgency: UrgencyLevel = Field(description="How urgently care is needed")


class PlantIdentification(BaseModel):
    common_name: str = Field(description="Common plant name, e.g. 'Snake Plant'")
    scientific_name: str = Field(description="Latin binomial, e.g. 'Sansevieria trifasciata'")
    family: str = Field(description="Plant family, e.g. 'Asparagaceae'")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in identification, 0.0 to 1.0"
    )


class PlantAnalysisResponse(BaseModel):
    identification: PlantIdentification
    sunlight: SunlightRequirement
    water: WaterRequirement
    health: HealthAssessment
    fun_facts: list[str] = Field(
        description="2-5 interesting facts about this plant species"
    )
    care_tips: list[str] = Field(
        description="2-5 general care tips specific to this plant"
    )
    rag_sources_used: list[str] = Field(
        default=[],
        description="Knowledge base entries that informed this response"
    )


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(description="Base64-encoded JPEG image")
    user_note: Optional[str] = Field(
        default=None,
        description="Optional user observation, e.g. 'leaves are turning yellow'"
    )
