"""Unit tests for LiteLLM client wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_mcp.models.config import ModelConfig, ModelConstraints, ModelsConfiguration
from multi_mcp.models.litellm_client import LiteLLMClient
from multi_mcp.models.resolver import ModelResolver
from multi_mcp.schemas.base import ModelResponse


class TestLiteLLMClient:
    """Tests for LiteLLMClient class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return ModelsConfiguration(
            version="1.0",
            models={
                "gpt-5-mini": ModelConfig(
                    litellm_model="openai/gpt-5-mini", aliases=["mini"], constraints=ModelConstraints(temperature=1.0)
                ),
                "gpt-5-pro": ModelConfig(litellm_model="openai/gpt-5-pro", aliases=["pro"]),
            },
        )

    @pytest.fixture
    def client(self, sample_config):
        """Create LiteLLM client with sample config."""
        resolver = ModelResolver(config=sample_config)
        return LiteLLMClient(resolver=resolver)

    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LiteLLM responses API response."""
        mock_response = MagicMock()

        # Responses API format
        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.role = "assistant"

        mock_content_item = MagicMock()
        mock_content_item.text = "Test response"
        mock_message.content = [mock_content_item]

        mock_response.output = [mock_message]

        # Usage stats (only total_tokens available in responses API)
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 150

        return mock_response

    @pytest.mark.asyncio
    async def test_call_async_success(self, client, mock_llm_response):
        """Test successful LLM call."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert isinstance(result, ModelResponse)
            assert result.status == "success"
            assert result.content == "Test response"
            assert result.metadata.model == "gpt-5-mini"
            assert result.metadata.total_tokens == 150
            assert result.metadata.latency_ms >= 0
            mock_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_async_with_model_resolution(self, client, mock_llm_response):
        """Test that model aliases are resolved correctly."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("mini")  # Alias
            result = await client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result.metadata.model == "gpt-5-mini"
            # Check that litellm_model was passed to acompletion
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["model"] == "openai/gpt-5-mini"

    @pytest.mark.asyncio
    async def test_call_async_uses_default_model_when_none_specified(self, client, mock_llm_response):
        """Test that default model is used when no model specified."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            default_model = client.resolver.get_default()
            canonical_name, model_config = client.resolver.resolve(default_model)
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            # Verify default model was used (dynamically, not hardcoded)
            assert result.metadata.model == canonical_name
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["model"] == model_config.litellm_model

    @pytest.mark.asyncio
    async def test_call_async_temperature_uses_default(self, client, mock_llm_response):
        """Test that default temperature is used when no constraint."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("gpt-5-pro")  # No temperature constraint
            await client.execute(canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}])

            call_kwargs = mock_completion.call_args[1]
            # Should use default temperature from settings (0.2)
            assert call_kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_call_async_temperature_constraint_enforced(self, client, mock_llm_response):
        """Test that model temperature constraints override default temperature."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")  # Has temperature=1.0 constraint
            await client.execute(canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}])

            call_kwargs = mock_completion.call_args[1]
            # Constraint should override default temperature
            assert call_kwargs["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_call_async_logging(self, client, mock_llm_response):
        """Test that LLM interactions are logged."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction") as mock_log,
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")
            await client.execute(canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}])

            mock_log.assert_called_once()
            call_args = mock_log.call_args[1]
            # Request data now contains the kwargs passed to litellm
            assert call_args["request_data"]["model"] == "openai/gpt-5-mini"
            assert call_args["request_data"]["input"] == [{"role": "user", "content": "Hello"}]  # Changed from "messages" to "input"
            # Response data is the ModelResponse dumped to dict
            assert call_args["response_data"]["content"] == "Test response"
            assert call_args["response_data"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_call_async_api_error_handling(self, client):
        """Test that API errors return error response."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.side_effect = Exception("API error")

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert isinstance(result, ModelResponse)
            assert result.status == "error"
            assert "API error" in result.error
            assert result.metadata.model == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_call_async_timeout_handling(self, client):
        """Test that timeout errors return error response."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.side_effect = TimeoutError()

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert isinstance(result, ModelResponse)
            assert result.status == "error"
            assert "timed out" in result.error
            assert result.metadata.model == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_call_async_messages_parameter(self, client, mock_llm_response):
        """Test that messages are passed correctly to LiteLLM as 'input'."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]

        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")
            await client.execute(canonical_name=canonical_name, model_config=model_config, messages=messages)

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["input"] == messages  # Changed from "messages" to "input"
            assert len(call_kwargs["input"]) == 2

    @pytest.mark.asyncio
    async def test_call_async_includes_model_params(self, sample_config, mock_llm_response):
        """Test that model-specific params are included in LLM call."""
        # Create config with custom params and explicit max_tokens
        config = ModelsConfiguration(
            version="1.0",
            models={
                "custom-model": ModelConfig(
                    litellm_model="provider/custom-model",
                    max_tokens=2000,  # Use explicit max_tokens field, not params
                    params={"top_p": 0.9},
                ),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("custom-model")
            await client.execute(canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}])

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["top_p"] == 0.9
            # Responses API uses `max_output_tokens`, not `max_tokens` (LiteLLM Responses API contract)
            assert call_kwargs["max_output_tokens"] == 2000
            assert "max_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_call_async_uses_default_max_tokens(self, sample_config, mock_llm_response):
        """Test that default max_tokens (32768) is used when not configured.

        For Responses API path, the parameter is named `max_output_tokens` (not `max_tokens`).
        """
        # Use sample_config which doesn't have max_tokens set
        resolver = ModelResolver(config=sample_config)
        client = LiteLLMClient(resolver=resolver)

        with (
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_completion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_completion.return_value = mock_llm_response

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")
            await client.execute(canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}])

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["max_output_tokens"] == 32768  # Default value (Responses API param)
            assert "max_tokens" not in call_kwargs

    def test_lazy_resolver_loading(self):
        """Test that resolver is lazy-loaded."""
        client = LiteLLMClient()

        # Resolver should not be created yet
        assert client._resolver is None

        # Access resolver property
        resolver = client.resolver

        # Now resolver should be created
        assert resolver is not None
        assert isinstance(resolver, ModelResolver)

        # Second access should return same instance
        resolver2 = client.resolver
        assert resolver is resolver2

    @pytest.mark.asyncio
    async def test_credential_validation_azure_missing_key(self, sample_config):
        """Test that Azure models fail with explicit error when AZURE_API_KEY is missing."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "azure-gpt-5-mini": ModelConfig(litellm_model="azure/gpt-5-mini", aliases=["azure-mini"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.azure_api_key = None
            mock_settings.azure_api_base = "https://example.azure.com"

            canonical_name, model_config = client.resolver.resolve("azure-gpt-5-mini")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "AZURE_API_KEY" in result.error
            assert "environment or .env file" in result.error
            assert "already set: AZURE_API_BASE" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_azure_missing_base(self, sample_config):
        """Test that Azure models fail with explicit error when AZURE_API_BASE is missing."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "azure-gpt-5-mini": ModelConfig(litellm_model="azure/gpt-5-mini", aliases=["azure-mini"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.azure_api_key = "test-key"
            mock_settings.azure_api_base = None

            canonical_name, model_config = client.resolver.resolve("azure-gpt-5-mini")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "AZURE_API_BASE" in result.error
            assert "environment or .env file" in result.error
            assert "already set: AZURE_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_azure_missing_both(self, sample_config):
        """Test that Azure models show both missing credentials."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "azure-gpt-5-mini": ModelConfig(litellm_model="azure/gpt-5-mini", aliases=["azure-mini"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.azure_api_key = None
            mock_settings.azure_api_base = None

            canonical_name, model_config = client.resolver.resolve("azure-gpt-5-mini")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "AZURE_API_KEY" in result.error
            assert "AZURE_API_BASE" in result.error
            assert "environment or .env file" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_gemini_missing(self, sample_config):
        """Test that Gemini models fail with explicit error when GEMINI_API_KEY is missing."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "gemini-2.5-flash": ModelConfig(litellm_model="gemini/gemini-2.5-flash", aliases=["flash"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.gemini_api_key = None

            canonical_name, model_config = client.resolver.resolve("gemini-2.5-flash")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "GEMINI_API_KEY" in result.error
            assert "environment or .env file" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_anthropic_missing(self, sample_config):
        """Test that Anthropic models fail with explicit error when ANTHROPIC_API_KEY is missing."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "claude-sonnet-4.5": ModelConfig(litellm_model="anthropic/claude-sonnet-4-5-20250929", aliases=["sonnet"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = None

            canonical_name, model_config = client.resolver.resolve("claude-sonnet-4.5")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "ANTHROPIC_API_KEY" in result.error
            assert "environment or .env file" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_openrouter_missing(self, sample_config):
        """Test that OpenRouter models fail with explicit error when OPENROUTER_API_KEY is missing."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "openrouter-model": ModelConfig(litellm_model="openrouter/some-model", aliases=["or"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.openrouter_api_key = None

            canonical_name, model_config = client.resolver.resolve("openrouter-model")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "OPENROUTER_API_KEY" in result.error
            assert "environment or .env file" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_openai_missing(self, sample_config):
        """Test that OpenAI models fail with explicit error when OPENAI_API_KEY is missing."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "gpt-5-mini": ModelConfig(litellm_model="openai/gpt-5-mini", aliases=["mini"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.openai_api_key = None

            canonical_name, model_config = client.resolver.resolve("gpt-5-mini")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "OPENAI_API_KEY" in result.error
            assert "environment or .env file" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_bedrock_missing_all(self, sample_config):
        """Test that Bedrock models fail with explicit error when all AWS credentials are missing."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "bedrock-model": ModelConfig(litellm_model="bedrock/anthropic.claude-sonnet-4-5-v2", aliases=["bedrock"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.aws_access_key_id = None
            mock_settings.aws_secret_access_key = None
            mock_settings.aws_region_name = None

            canonical_name, model_config = client.resolver.resolve("bedrock-model")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "AWS_ACCESS_KEY_ID" in result.error
            assert "AWS_SECRET_ACCESS_KEY" in result.error
            assert "AWS_REGION_NAME" in result.error
            assert "environment or .env file" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_bedrock_partial(self, sample_config):
        """Test that Bedrock shows which AWS credentials are already set."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "bedrock-model": ModelConfig(litellm_model="bedrock/anthropic.claude-sonnet-4-5-v2", aliases=["bedrock"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        with patch("multi_mcp.models.litellm_client.settings") as mock_settings:
            mock_settings.aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
            mock_settings.aws_secret_access_key = None
            mock_settings.aws_region_name = "us-east-1"

            canonical_name, model_config = client.resolver.resolve("bedrock-model")
            result = await client.execute(
                canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "Hello"}]
            )

            assert result.status == "error"
            assert "AWS_SECRET_ACCESS_KEY" in result.error
            assert "already set: AWS_ACCESS_KEY_ID, AWS_REGION_NAME" in result.error

    @pytest.mark.asyncio
    async def test_credential_validation_ollama_no_key_required(self, sample_config):
        """Test that Ollama models pass validation without any API key."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "ollama-llama": ModelConfig(litellm_model="ollama_chat/llama3.2", aliases=["llama"]),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        _, model_config = client.resolver.resolve("ollama-llama")
        error = client._validate_provider_credentials(model_config.litellm_model)
        assert error is None

    @pytest.mark.asyncio
    async def test_credential_validation_ollama_generate_prefix(self, sample_config):
        """Test that ollama/ prefix also passes validation without credentials."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "ollama-model": ModelConfig(litellm_model="ollama/llama3.2"),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        _, model_config = client.resolver.resolve("ollama-model")
        error = client._validate_provider_credentials(model_config.litellm_model)
        assert error is None

    @pytest.mark.asyncio
    async def test_call_async_rejects_cli_models(self, sample_config):
        """Test that call_async rejects CLI models."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "gemini-cli": ModelConfig(provider="cli", cli_command="gemini", cli_args=["chat"], cli_parser="json"),
            },
        )
        resolver = ModelResolver(config=config)
        client = LiteLLMClient(resolver=resolver)

        canonical_name, model_config = client.resolver.resolve("gemini-cli")
        result = await client.execute(
            canonical_name=canonical_name, model_config=model_config, messages=[{"role": "user", "content": "test"}]
        )

        assert result.status == "error"
        assert "CLI model" in result.error
        assert "Use CLIExecutor" in result.error
        assert result.metadata.model == "gemini-cli"


class TestOllamaChatCompletionPath:
    """Tests for the dual-path routing: Ollama models go through litellm.acompletion (not aresponses).

    Ollama (and similar providers) only expose Chat Completions API. Calling litellm.aresponses()
    on them raises BadRequestError ("LLM Provider NOT provided"). These tests verify that:
      - ollama_chat/ and ollama/ prefixes route to acompletion
      - The "messages" key is used (not "input")
      - Web search is skipped (not supported by Ollama)
      - api_base from settings is passed through
      - Content is extracted from response.choices[0].message.content
    """

    @pytest.fixture
    def ollama_client(self):
        """Create LiteLLM client with an Ollama model config."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "ollama-glm": ModelConfig(
                    litellm_model="ollama_chat/glm-5.1:cloud",
                    aliases=["glm-5.1", "glm"],
                ),
                "ollama-llama": ModelConfig(
                    litellm_model="ollama/llama3.2",
                    aliases=["llama"],
                ),
            },
        )
        return LiteLLMClient(resolver=ModelResolver(config=config))

    @pytest.fixture
    def mock_chat_completion_response(self):
        """Create mock Chat Completions API response (Ollama-shaped)."""
        mock = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hello from Ollama"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock.choices = [mock_choice]

        mock.usage = MagicMock()
        mock.usage.total_tokens = 42
        return mock

    @pytest.mark.asyncio
    async def test_ollama_routes_to_acompletion_not_aresponses(self, ollama_client, mock_chat_completion_response):
        """Ollama models must call litellm.acompletion, NOT litellm.aresponses."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_aresponses,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(ollama_client, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = mock_chat_completion_response

            canonical_name, model_config = ollama_client.resolver.resolve("glm-5.1")
            result = await ollama_client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert result.status == "success"
            assert result.content == "Hello from Ollama"
            assert result.metadata.total_tokens == 42

            # The critical assertion: acompletion was used, NOT aresponses
            mock_acompletion.assert_called_once()
            mock_aresponses.assert_not_called()

    @pytest.mark.asyncio
    async def test_ollama_generate_prefix_also_routes_to_acompletion(self, ollama_client, mock_chat_completion_response):
        """The bare 'ollama/' prefix (not just 'ollama_chat/') also routes to acompletion."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_aresponses,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(ollama_client, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = mock_chat_completion_response

            canonical_name, model_config = ollama_client.resolver.resolve("llama")
            await ollama_client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            mock_acompletion.assert_called_once()
            mock_aresponses.assert_not_called()

    @pytest.mark.asyncio
    async def test_ollama_uses_messages_key_not_input(self, ollama_client, mock_chat_completion_response):
        """Ollama path must pass 'messages' to acompletion, not 'input' (which is Responses API)."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(ollama_client, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = mock_chat_completion_response

            canonical_name, model_config = ollama_client.resolver.resolve("glm-5.1")
            await ollama_client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=messages,
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert "messages" in call_kwargs
            assert "input" not in call_kwargs
            assert call_kwargs["messages"] == messages

    @pytest.mark.asyncio
    async def test_ollama_skips_web_search_tool(self, mock_chat_completion_response):
        """Even with enable_web_search=True and provider_web_search=True, Ollama path drops the tool.

        Ollama doesn't understand OpenAI's `tools=[{"type": "web_search"}]`. We must NOT pass it.
        """
        config = ModelsConfiguration(
            version="1.0",
            models={
                "ollama-glm": ModelConfig(
                    litellm_model="ollama_chat/glm-5.1:cloud",
                    aliases=["glm"],
                    provider_web_search=True,  # Artificially enabled to verify it's still skipped
                ),
            },
        )
        client = LiteLLMClient(resolver=ModelResolver(config=config))

        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = mock_chat_completion_response

            canonical_name, model_config = client.resolver.resolve("glm")
            await client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
                enable_web_search=True,
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_ollama_passes_api_base_when_configured(self, ollama_client, mock_chat_completion_response):
        """When settings.ollama_api_base is set, it must be passed to acompletion as api_base."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(ollama_client, "_validate_provider_credentials", return_value=None),
            patch("multi_mcp.models.litellm_client.settings") as mock_settings,
        ):
            mock_settings.ollama_api_base = "http://my-ollama:11434"
            mock_settings.default_temperature = 0.2
            mock_settings.max_retries = 3
            mock_settings.model_timeout_seconds = 300
            mock_acompletion.return_value = mock_chat_completion_response

            canonical_name, model_config = ollama_client.resolver.resolve("glm-5.1")
            await ollama_client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert call_kwargs["api_base"] == "http://my-ollama:11434"

    @pytest.mark.asyncio
    async def test_ollama_uses_max_tokens_param_not_max_output_tokens(self, ollama_client, mock_chat_completion_response):
        """Chat Completions API uses `max_tokens` (not `max_output_tokens` which is Responses API)."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(ollama_client, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = mock_chat_completion_response

            canonical_name, model_config = ollama_client.resolver.resolve("glm-5.1")
            await ollama_client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert "max_tokens" in call_kwargs
            assert "max_output_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_ollama_omits_api_base_when_not_configured(self, ollama_client, mock_chat_completion_response):
        """When settings.ollama_api_base is None, no api_base key is passed (LiteLLM uses its default)."""
        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(ollama_client, "_validate_provider_credentials", return_value=None),
            patch("multi_mcp.models.litellm_client.settings") as mock_settings,
        ):
            mock_settings.ollama_api_base = None
            mock_settings.default_temperature = 0.2
            mock_settings.max_retries = 3
            mock_settings.model_timeout_seconds = 300
            mock_acompletion.return_value = mock_chat_completion_response

            canonical_name, model_config = ollama_client.resolver.resolve("glm-5.1")
            await ollama_client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert "api_base" not in call_kwargs

    @pytest.mark.asyncio
    async def test_non_ollama_models_still_use_aresponses(self, mock_chat_completion_response):
        """Sanity check: non-Ollama models (OpenAI, Anthropic, etc.) continue using aresponses."""
        config = ModelsConfiguration(
            version="1.0",
            models={
                "gpt-5-mini": ModelConfig(litellm_model="openai/gpt-5-mini", aliases=["mini"]),
            },
        )
        client = LiteLLMClient(resolver=ModelResolver(config=config))

        # Use the responses-API-shaped mock from the parent class
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.type = "message"
        mock_content = MagicMock()
        mock_content.text = "openai response"
        mock_message.content = [mock_content]
        mock_response.output = [mock_message]
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 10

        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.litellm.aresponses", new_callable=AsyncMock) as mock_aresponses,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client, "_validate_provider_credentials", return_value=None),
        ):
            mock_aresponses.return_value = mock_response

            canonical_name, model_config = client.resolver.resolve("mini")
            await client.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            mock_aresponses.assert_called_once()
            mock_acompletion.assert_not_called()


class TestExtractContentFromChatCompletion:
    """Tests for the chat-completion content extractor (used by Ollama path)."""

    def test_extracts_content_from_object_format(self):
        """Standard MagicMock-style response (LiteLLM ModelResponse object)."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        mock = MagicMock()
        mock.choices[0].message.content = "hello"
        assert _extract_content_from_chat_completion(mock) == "hello"

    def test_extracts_content_from_dict_format(self):
        """When LiteLLM returns a raw dict (some providers do this)."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {"choices": [{"message": {"content": "hello dict"}}]}
        assert _extract_content_from_chat_completion(response) == "hello dict"

    def test_returns_empty_string_when_choices_empty(self):
        """No choices → empty string (don't crash, let caller decide)."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {"choices": []}
        assert _extract_content_from_chat_completion(response) == ""

    def test_returns_empty_string_when_choices_missing(self):
        """No 'choices' key at all → empty string."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {}
        assert _extract_content_from_chat_completion(response) == ""

    def test_returns_empty_string_when_message_missing(self):
        """First choice has no message → empty string."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {"choices": [{}]}
        assert _extract_content_from_chat_completion(response) == ""

    def test_returns_empty_string_when_content_is_none(self):
        """Message exists but content is None (some Ollama responses do this)."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {"choices": [{"message": {"content": None}}]}
        assert _extract_content_from_chat_completion(response) == ""

    def test_normalizes_list_content_to_string(self):
        """Structured content as list of {type, text} parts is joined into a string.

        Some providers return content as `[{"type": "text", "text": "hello"}, ...]`.
        Without normalization, ModelResponse(content=...) would raise because content
        is annotated as str.
        """
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "hello "},
                            {"type": "text", "text": "world"},
                        ]
                    }
                }
            ]
        }
        assert _extract_content_from_chat_completion(response) == "hello world"

    def test_returns_empty_string_for_unexpected_content_type(self):
        """Defensive: int/dict/anything else → "" (don't crash, don't return non-str)."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {"choices": [{"message": {"content": 42}}]}
        assert _extract_content_from_chat_completion(response) == ""

    def test_handles_none_text_in_list_parts_defensively(self):
        """`{"text": null}` part used to crash "".join() — defensive `or ""` fix."""
        from multi_mcp.models.litellm_client import _extract_content_from_chat_completion

        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": None},
                            {"type": "text", "text": "after null"},
                        ]
                    }
                }
            ]
        }
        # Should not raise TypeError; null part contributes empty string
        assert _extract_content_from_chat_completion(response) == "after null"


class TestExtractContentFromResponsesApi:
    """Tests for the Responses API content extractor (used by OpenAI, Azure, Anthropic, Gemini).

    Previously this function was only tested indirectly via mocked execute() calls. Direct tests
    pin down dict/object handling, edge cases (None content, empty output), and the bug where
    `hasattr(response, 'output')` silently failed for dict-shaped responses.
    """

    def test_extracts_content_from_object_format(self):
        """Standard MagicMock-style response (LiteLLM ModelResponse object)."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        mock = MagicMock()
        mock_message = MagicMock()
        mock_message.type = "message"
        mock_content = MagicMock()
        mock_content.text = "hello"
        mock_message.content = [mock_content]
        mock.output = [mock_message]
        assert _extract_content_from_responses_api(mock) == "hello"

    def test_extracts_content_from_dict_format(self):
        """The bug fix: hasattr() didn't work for dicts; now uses isinstance check."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "hello dict"}]}]}
        assert _extract_content_from_responses_api(response) == "hello dict"

    def test_returns_empty_string_when_no_output_key(self):
        """Dict without 'output' key → "" (used to silently fail via hasattr)."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        assert _extract_content_from_responses_api({}) == ""

    def test_returns_empty_string_when_output_empty(self):
        """Empty output array → "" (no message to extract)."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        assert _extract_content_from_responses_api({"output": []}) == ""

    def test_returns_empty_string_when_no_message_item(self):
        """Output has items but none are message type (e.g. only reasoning/web_search_call)."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        response = {"output": [{"type": "reasoning", "content": "thinking..."}]}
        assert _extract_content_from_responses_api(response) == ""

    def test_skips_message_with_none_content_and_continues(self):
        """A message with content=None is skipped; next message with content wins."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        response = {
            "output": [
                {"type": "message", "content": None},
                {"type": "message", "content": [{"type": "output_text", "text": "second"}]},
            ]
        }
        assert _extract_content_from_responses_api(response) == "second"

    def test_handles_string_content_fallback(self):
        """Some providers put plain string content directly (not a list of parts)."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        response = {"output": [{"type": "message", "content": "plain string"}]}
        assert _extract_content_from_responses_api(response) == "plain string"

    def test_handles_none_text_in_list_parts_defensively(self):
        """Defensive: `{"text": null}` in content list doesn't crash "".join()."""
        from multi_mcp.models.litellm_client import _extract_content_from_responses_api

        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": None}, {"type": "output_text", "text": "after"}],
                }
            ]
        }
        assert _extract_content_from_responses_api(response) == "after"


class TestEmptyContentAndErrorHandling:
    """Tests for the post-fix safety nets in execute(): empty content → error, traceback logging."""

    @pytest.fixture
    def client_for_ollama(self):
        config = ModelsConfiguration(
            version="1.0",
            models={
                "ollama-glm": ModelConfig(
                    litellm_model="ollama_chat/glm-5.1:cloud",
                    aliases=["glm-5.1"],
                ),
            },
        )
        return LiteLLMClient(resolver=ModelResolver(config=config))

    @pytest.mark.asyncio
    async def test_empty_content_returns_error_not_success(self, client_for_ollama):
        """When extractor returns empty content, execute() must return status:error.

        Previously this silently returned status:success with empty content — a
        nasty failure mode where downstream code received "successful" empty reviews.
        """
        empty_response = MagicMock()
        empty_response.choices = [MagicMock(message=MagicMock(content=""))]
        empty_response.usage = MagicMock(total_tokens=0)

        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client_for_ollama, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = empty_response

            canonical_name, model_config = client_for_ollama.resolver.resolve("glm-5.1")
            result = await client_for_ollama.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert result.status == "error"
            assert "empty content" in result.error
            assert "acompletion" in result.error  # api_path is mentioned

    @pytest.mark.asyncio
    async def test_whitespace_only_content_returns_error(self, client_for_ollama):
        """Whitespace-only content also counts as empty (would break consumers expecting real text)."""
        ws_response = MagicMock()
        ws_response.choices = [MagicMock(message=MagicMock(content="   \n  \t  "))]
        ws_response.usage = MagicMock(total_tokens=5)

        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client_for_ollama, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = ws_response

            canonical_name, model_config = client_for_ollama.resolver.resolve("glm-5.1")
            result = await client_for_ollama.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert result.status == "error"
            assert "empty content" in result.error

    @pytest.mark.asyncio
    async def test_litellm_timeout_caught_separately(self, client_for_ollama):
        """litellm.Timeout (HTTP-layer) must produce the same clean timeout message as TimeoutError."""
        import litellm as litellm_module

        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client_for_ollama, "_validate_provider_credentials", return_value=None),
        ):
            # litellm.Timeout inherits from openai.APITimeoutError; we just need an instance
            mock_acompletion.side_effect = litellm_module.Timeout(
                message="LiteLLM HTTP timeout", model="ollama_chat/glm-5.1:cloud", llm_provider="ollama_chat"
            )

            canonical_name, model_config = client_for_ollama.resolver.resolve("glm-5.1")
            result = await client_for_ollama.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert result.status == "error"
            assert "timed out" in result.error
            assert "Request timed out after" in result.error

    @pytest.mark.asyncio
    async def test_usage_falls_back_to_prompt_plus_completion(self, client_for_ollama):
        """When usage.total_tokens is None/0 but prompt+completion exist, sum them.

        Some chat-completion providers don't compute total_tokens; without this
        fallback, latency_ms/cost metadata silently shows 0 tokens.
        """
        response_with_split = MagicMock()
        response_with_split.choices = [MagicMock(message=MagicMock(content="hello"))]
        # Simulate provider that doesn't set total_tokens
        usage = MagicMock()
        usage.total_tokens = None
        usage.prompt_tokens = 10
        usage.completion_tokens = 25
        response_with_split.usage = usage

        with (
            patch("multi_mcp.models.litellm_client.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("multi_mcp.models.litellm_client.log_llm_interaction"),
            patch.object(client_for_ollama, "_validate_provider_credentials", return_value=None),
        ):
            mock_acompletion.return_value = response_with_split

            canonical_name, model_config = client_for_ollama.resolver.resolve("glm-5.1")
            result = await client_for_ollama.execute(
                canonical_name=canonical_name,
                model_config=model_config,
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert result.status == "success"
            assert result.metadata.total_tokens == 35  # 10 + 25
