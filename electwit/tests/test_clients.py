import pytest
from electwit.clients import (
    OpenRouterClient,
    GeminiClient,
)


@pytest.mark.asyncio
async def test_openrouter_client():
    model_name = "openrouter/cypher-alpha:free"
    client = OpenRouterClient(
        model_name=model_name,
    )
    response = await client.generate_response("Hi. This is a test prompt.")

    assert isinstance(response, str)
    assert len(response) > 0, "Response should not be empty"


@pytest.mark.asyncio
async def test_gemini_client():
    model_name = "gemini-1.5-flash"
    client = GeminiClient(
        model_name=model_name,
    )
    response = await client.generate_response("Hi. This is a test prompt.")

    assert isinstance(response, str)
    assert len(response) > 0, "Response should not be empty"
