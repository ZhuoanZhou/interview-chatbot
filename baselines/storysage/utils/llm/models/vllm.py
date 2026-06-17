from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from utils.llm.models.data import ModelResponse

load_dotenv(override=True)


class VLLMEngine:
    """
    A wrapper class for vLLM models served via OpenAI-compatible API.

    This engine allows you to use locally hosted vLLM instances by pointing
    to their base URL. The vLLM server must be started with --api-key or
    without authentication.

    Environment variables:
        VLLM_BASE_URL: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        VLLM_API_KEY: Optional API key if the vLLM server requires authentication
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

        # Extract base_url from kwargs or environment
        base_url = kwargs.pop("base_url", None) or os.getenv("VLLM_BASE_URL")
        api_key = kwargs.pop("api_key", None) or os.getenv("VLLM_API_KEY", "EMPTY")

        self.kwargs = kwargs

        if not base_url:
            raise ValueError(
                "VLLM_BASE_URL environment variable must be set or base_url must be passed. "
                "Example: VLLM_BASE_URL='http://localhost:8000/v1'"
            )

        # Initialize the OpenAI client pointing to vLLM
        self.client = ChatOpenAI(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    def invoke(self, prompt, **kwargs) -> ModelResponse:
        """
        Invoke the vLLM model with the given prompt.

        Args:
            prompt: The input prompt as a string
            **kwargs: Additional keyword arguments for the model invocation

        Returns:
            A ModelResponse object with the model's response and usage metadata
        """
        response = self.client.invoke(prompt, **kwargs)

        # Create ModelResponse with content and usage metadata
        model_response = ModelResponse(response.content)

        # Extract token usage information if available
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            model_response.response_metadata = {
                'token_usage': response.response_metadata['token_usage']
            }

        return model_response
