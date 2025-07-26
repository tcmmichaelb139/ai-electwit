import os
import json
import logging
import re
from typing import Optional, List


from openai import AsyncOpenAI
import json_repair


from electwit.utils import load_prompt


logger = logging.getLogger(__name__)


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)


class BaseModelClient:
    """
    Base interface for LLM clients
    Each must provide:
     - generate_response(prompt: str, **kwargs) -> str
     - TODO
    """

    def __init__(self, model_name: str, prompt_dir: Optional[str] = None):
        self.model_name = model_name
        self.client = None
        self.max_tokens = 16384
        self.prompt_dir = prompt_dir
        self.system_prompt = load_prompt("system_prompt.txt", self.prompt_dir)

    def set_system_prompt(self, system_prompt: str):
        """
        Sets the system prompt
        """

        if not system_prompt:
            raise ValueError("System prompt cannot be empty.")

        self.system_prompt = system_prompt

    async def generate_response(self, prompt: str) -> str:
        """
        Returns the raw response generated from the LLM
        Subclasses override this
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def generate_response_json_list(self, prompt: str) -> Optional[List[dict]]:
        """
        Returns the response in JSON format
        """
        response = await self.generate_response(prompt)

        response = response.strip()

        if not response:
            logger.warning(f"[{self.model_name}] Empty response generated.")
            return []

        response_json = json_repair.loads(response)

        if not isinstance(response_json, list):
            logger.warning(
                f"[{self.model_name}] Response is not a list: {response_json}. "
            )
            return [response_json]

        return response_json


class OpenRouterClient(BaseModelClient):
    """
    Client interface for OpenRouter API
    """

    def __init__(
        self,
        model_name: str = "openrouter/cypher-alpha:free",
        prompt_dir: Optional[str] = None,
    ):
        super().__init__(model_name, prompt_dir)

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Generates a response from the OpenRouter API using the specified model.
        """

        for _ in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )

                if not response.choices:
                    logger.warning(
                        f"[{self.model_name}] No choices returned in response."
                    )
                    return ""

                logger.debug(f"[{self.model_name}] Response: {response}")

                content = response.choices[0].message.content.strip()

                if not content:
                    logger.warning(
                        f"[{self.model_name}] Empty content returned in response."
                    )
                    return ""

                return content

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                continue

        logger.error(f"[{self.model_name}] Failed to generate response after retries.")
        return ""


def load_model_client(
    model_name: str, prompt_dir: Optional[str] = None
) -> BaseModelClient:
    """
    Loads a model client based on the model name.
    """

    if (
        "openrouter" in model_name
        or "deepseek" in model_name
        or "mistral" in model_name
        or "llama" in model_name
        or "openai" in model_name
        or "grok" in model_name
        or "gemini" in model_name
        or "qwen" in model_name
        or "claude" in model_name
        or "moonshotai" in model_name
    ):
        return OpenRouterClient(model_name=model_name, prompt_dir=prompt_dir)

    raise ValueError(f"Unsupported model type: {model_name}.")
