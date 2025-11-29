"""LLM client for calling OpenAI and Anthropic APIs."""

import asyncio
import logging
import os
from typing import Optional, Dict
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available")

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic library not available")

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq library not available")

from .models import LLMRequest, LLMResponse, AgentType

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class LLMClientError(Exception):
    """Custom exception for LLM client errors."""
    pass


class LLMClient:
    """
    Client for calling LLM APIs (OpenAI or Anthropic).
    
    Supports:
    - OpenAI: gpt-4-turbo, gpt-4
    - Anthropic: claude-sonnet-4, claude-opus-4
    
    Features:
    - Async API calls
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Error handling and logging
    """
    
    # Model to provider mapping
    MODEL_PROVIDERS = {
        "gpt-4-turbo": LLMProvider.OPENAI,
        "gpt-4": LLMProvider.OPENAI,
        "gpt-4-turbo-preview": LLMProvider.OPENAI,
        "gpt-4o": LLMProvider.OPENAI,
        "claude-sonnet-4": LLMProvider.ANTHROPIC,
        "claude-opus-4": LLMProvider.ANTHROPIC,
        "claude-3-opus-20240229": LLMProvider.ANTHROPIC,
        "claude-3-sonnet-20240229": LLMProvider.ANTHROPIC,
        "claude-3-5-sonnet-20240620": LLMProvider.ANTHROPIC,
        # Groq - LLaMA 3.3 (CURRENT)
        "llama-3.3-70b-versatile": LLMProvider.GROQ,
        "llama-3.3-70b-specdec": LLMProvider.GROQ,
        # LLaMA 3.2
        "llama-3.2-90b-text-preview": LLMProvider.GROQ,
        "llama-3.2-11b-text-preview": LLMProvider.GROQ,
        "llama-3.2-3b-preview": LLMProvider.GROQ,
        "llama-3.2-1b-preview": LLMProvider.GROQ,
        # Deprecated LLaMA 3.1 (kept for backwards compatibility)
        "llama-3.1-70b-versatile": LLMProvider.GROQ,
        "llama-3.1-8b-instant": LLMProvider.GROQ,
        "llama3-70b-8192": LLMProvider.GROQ,
        "llama3-8b-8192": LLMProvider.GROQ,
    }
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LLM client.
        
        Args:
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            groq_api_key: Groq API key (or use GROQ_API_KEY env var)
            retry_attempts: Number of retry attempts on failure
            retry_delay: Initial delay between retries (exponential backoff)
        """
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not provided")
        
        # Initialize Anthropic client
        self.anthropic_client = None
        if ANTHROPIC_AVAILABLE:
            api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = AsyncAnthropic(api_key=api_key)
                logger.info("Anthropic client initialized")
            else:
                logger.warning("Anthropic API key not provided")
        
        # Initialize Groq client
        self.groq_client = None
        if GROQ_AVAILABLE:
            api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = AsyncGroq(api_key=api_key)
                logger.info("Groq client initialized")
            else:
                logger.warning("Groq API key not provided")
        
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
    
    def _get_provider(self, model: str) -> LLMProvider:
        """
        Determine provider from model name.
        
        Args:
            model: Model name
        
        Returns:
            LLMProvider enum
        
        Raises:
            LLMClientError: If model is not supported
        """
        provider = self.MODEL_PROVIDERS.get(model)
        if not provider:
            raise LLMClientError(f"Unsupported model: {model}")
        return provider
    
    async def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, int, str]:
        """
        Call OpenAI API.
        
        Returns:
            Tuple of (response_text, tokens_used, finish_reason)
        """
        if not self.openai_client:
            raise LLMClientError("OpenAI client not initialized")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            finish_reason = response.choices[0].finish_reason
            
            return text, tokens, finish_reason
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMClientError(f"OpenAI API call failed: {e}")
    
    async def _call_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, int, str]:
        """
        Call Anthropic API.
        
        Returns:
            Tuple of (response_text, tokens_used, finish_reason)
        """
        if not self.anthropic_client:
            raise LLMClientError("Anthropic client not initialized")
        
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            finish_reason = response.stop_reason
            
            return text, tokens, finish_reason
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise LLMClientError(f"Anthropic API call failed: {e}")
    
    async def _call_groq(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, int, str]:
        """
        Call Groq API (LLaMA models).
        
        Returns:
            Tuple of (response_text, tokens_used, finish_reason)
        """
        if not self.groq_client:
            raise LLMClientError("Groq client not initialized")
        
        try:
            response = await self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            finish_reason = response.choices[0].finish_reason
            
            return text, tokens, finish_reason
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise LLMClientError(f"Groq API call failed: {e}")
    
    async def complete(
        self,
        prompt: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        agent_type: Optional[AgentType] = None
    ) -> LLMResponse:
        """
        Call LLM API and return completion.
        
        Supports:
        - OpenAI: gpt-4-turbo, gpt-4
        - Anthropic: claude-sonnet-4, claude-opus-4
        - Groq: llama-3.1-70b-versatile, llama-3.1-8b-instant
        
        Handles rate limits, retries, errors.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature (0-2)
            max_tokens: Maximum tokens to generate
            agent_type: Agent type for tracking
        
        Returns:
            LLMResponse with completion text and metadata
        
        Raises:
            LLMClientError: If all retry attempts fail
        """
        provider = self._get_provider(model)
        
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"LLM call attempt {attempt + 1}/{self.retry_attempts} "
                           f"(model={model}, temp={temperature})")
                
                # Call appropriate provider
                if provider == LLMProvider.OPENAI:
                    text, tokens, finish_reason = await self._call_openai(
                        prompt, model, temperature, max_tokens
                    )
                elif provider == LLMProvider.ANTHROPIC:
                    text, tokens, finish_reason = await self._call_anthropic(
                        prompt, model, temperature, max_tokens
                    )
                else:  # GROQ
                    text, tokens, finish_reason = await self._call_groq(
                        prompt, model, temperature, max_tokens
                    )
                
                logger.info(f"LLM call successful: {tokens} tokens, finish_reason={finish_reason}")
                
                return LLMResponse(
                    text=text,
                    agent_type=agent_type or AgentType.CONTEXT_AWARE,
                    model=model,
                    tokens_used=tokens,
                    finish_reason=finish_reason
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
        
        # All retries failed
        error_msg = f"All {self.retry_attempts} attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        raise LLMClientError(error_msg)
    
    async def complete_multiple(
        self,
        requests: Dict[AgentType, LLMRequest]
    ) -> Dict[AgentType, LLMResponse]:
        """
        Execute multiple LLM requests in parallel.
        
        Args:
            requests: Dict mapping AgentType to LLMRequest
        
        Returns:
            Dict mapping AgentType to LLMResponse
        """
        logger.info(f"Executing {len(requests)} parallel LLM requests")
        
        # Create tasks for parallel execution
        tasks = {}
        for agent_type, request in requests.items():
            task = self.complete(
                prompt=request.prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                agent_type=agent_type
            )
            tasks[agent_type] = task
        
        # Execute in parallel
        results = {}
        errors = {}
        
        for agent_type, task in tasks.items():
            try:
                response = await task
                results[agent_type] = response
            except Exception as e:
                logger.error(f"Request for {agent_type} failed: {e}")
                errors[agent_type] = str(e)
        
        if not results:
            raise LLMClientError(f"All parallel requests failed: {errors}")
        
        if errors:
            logger.warning(f"Some requests failed: {errors}")
        
        logger.info(f"Completed {len(results)}/{len(requests)} parallel requests")
        return results


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API calls.
    
    Returns pre-defined mock translations.
    """
    
    def __init__(self):
        """Initialize mock client (no API keys needed)."""
        super().__init__()
        self.call_count = 0
    
    async def complete(
        self,
        prompt: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        agent_type: Optional[AgentType] = None
    ) -> LLMResponse:
        """
        Return mock translation based on agent type.
        
        Returns different mock translations for each agent type.
        """
        self.call_count += 1
        
        # Generate mock translation based on agent type
        if agent_type == AgentType.CONTEXT_AWARE:
            text = "يجب إدخال المرضى الذين يعانون من أعراض شديدة إلى وحدات العناية المركزة."
        elif agent_type == AgentType.TERMINOLOGY_OPTIMIZED:
            text = "يجب أن يتم إدخالهم المرضى مع الأعراض الشديدة إلى وحدات العناية المركزة."
        elif agent_type == AgentType.CONSERVATIVE:
            text = "المرضى مع الأعراض الشديدة يجب يتم إدخالهم إلى وحدات العناية المركزة."
        else:
            text = "ترجمة تجريبية للنص المطلوب."
        
        await asyncio.sleep(0.1)  # Simulate API delay
        
        logger.info(f"Mock LLM call #{self.call_count} for {agent_type}")
        
        return LLMResponse(
            text=text,
            agent_type=agent_type or AgentType.CONTEXT_AWARE,
            model=model,
            tokens_used=100,
            finish_reason="stop"
        )


if __name__ == "__main__":
    # Test LLM client
    import asyncio
    
    async def test_client():
        logging.basicConfig(level=logging.INFO)
        
        # Test with mock client
        client = MockLLMClient()
        
        response = await client.complete(
            prompt="Translate: Hello world",
            agent_type=AgentType.CONTEXT_AWARE
        )
        
        print(f"Response: {response.text}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Model: {response.model}")
        
        # Test parallel requests
        from .models import LLMRequest
        
        requests = {
            AgentType.CONTEXT_AWARE: LLMRequest(
                prompt="Test prompt 1",
                agent_type=AgentType.CONTEXT_AWARE,
                temperature=0.3,
                model="gpt-4-turbo",
                max_tokens=2048
            ),
            AgentType.TERMINOLOGY_OPTIMIZED: LLMRequest(
                prompt="Test prompt 2",
                agent_type=AgentType.TERMINOLOGY_OPTIMIZED,
                temperature=0.2,
                model="gpt-4-turbo",
                max_tokens=2048
            ),
        }
        
        results = await client.complete_multiple(requests)
        print(f"\nParallel results: {len(results)} responses")
        for agent_type, response in results.items():
            print(f"  {agent_type}: {response.text[:50]}...")
    
    asyncio.run(test_client())
