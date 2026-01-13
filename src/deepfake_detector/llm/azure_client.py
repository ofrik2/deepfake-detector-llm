from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from .client_base import LLMClient, LLMResponse
import time
from openai import RateLimitError
import base64
from pathlib import Path




@dataclass
class AzureOpenAIClient(LLMClient):
    """
    Azure OpenAI client implementation.

    Uses text + optional image inputs.
    Assumes environment variables are set (see env.example).
    """
    model_name: str = "azure-openai"

    def __post_init__(self):
        load_dotenv()

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([endpoint, api_key, deployment, api_version]):
            raise RuntimeError(
                "Missing Azure OpenAI environment variables. "
                "Check AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION."
            )

        self._deployment = deployment

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        """
        Convert a local image file to a data URL (base64),
        which Azure OpenAI accepts for vision inputs.
        """
        path = Path(image_path)
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:image/jpeg;base64,{b64}"


    def generate(
        self,
        *,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Send prompt + optional images to Azure OpenAI.

        Images are attached as vision inputs if provided.
        """
        messages = []

        # User message content (text + images)
        content = [{"type": "text", "text": prompt}]

        if image_paths:
            for p in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_data_url(p)
                        },
                    }
                )

        messages.append({"role": "user", "content": content})

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=messages,
                    temperature=0.2,
                )
                break
            except RateLimitError as e:
                if attempt == 2:
                    raise
                wait = 60
                print(f"[AzureOpenAI] Rate limit hit. Sleeping {wait}s and retrying...")
                time.sleep(wait)


        text = resp.choices[0].message.content

        return LLMResponse(
            raw_text=text,
            model_name=self.model_name,
            usage=getattr(resp, "usage", None),
        )
