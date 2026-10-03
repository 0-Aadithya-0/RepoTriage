import asyncio
import os

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class TestAnalysis(BaseModel):
    category: str = Field(description="Bug, Feature, or Question")
    priority: int = Field(description="Priority from 1 to 5")
    summary: str = Field(description="Short summary")


async def test_gemini() -> None:
    # Use the test key from the env or fallback to a dummy if we're just checking imports
    # Actually, we can't make a real call without a real key, but we can verify
    # the client initialization and types construction.
    api_key = os.environ.get("GEMINI_API_KEY", "dummy")
    
    print(f"genai version: {genai.__version__}")
    
    # Initialize async client
    client = genai.Client(api_key=api_key)
    
    print("Client initialized successfully.")
    
    # We won't make a real call, just verifying the schema configuration syntax
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TestAnalysis,
        temperature=0.0,
    )
    
    print(f"Config created: {config}")
    print("All SDK imports and types verify correctly.")

if __name__ == "__main__":
    asyncio.run(test_gemini())
