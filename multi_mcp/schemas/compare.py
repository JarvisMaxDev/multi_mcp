"""Compare tool schema models."""

from typing import Annotated

from pydantic import BeforeValidator, Field

from multi_mcp.schemas.base import MultiToolRequest, MultiToolResponse, coerce_stringified_list
from multi_mcp.settings import settings


class CompareRequest(MultiToolRequest):
    """Chat request for side-by-side model compare."""

    models: Annotated[list[str], BeforeValidator(coerce_stringified_list)] = Field(
        default_factory=lambda: settings.default_model_list,
        min_length=2,
        description=f"List of LLM models to run in parallel (minimum 2) (will use default models ({settings.default_model_list}) if not specified)",
    )


class CompareResponse(MultiToolResponse):
    """Response with side-by-side compare results."""

    # Inherits results: list[ModelResponse] from MultiToolResponse
