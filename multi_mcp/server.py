"""FastMCP server with MCP tool wrappers."""

import logging

from fastmcp import FastMCP

from multi_mcp.schemas.chat import ChatRequest
from multi_mcp.schemas.codereview import CodeReviewRequest
from multi_mcp.schemas.compare import CompareRequest
from multi_mcp.schemas.debate import DebateRequest
from multi_mcp.settings import settings
from multi_mcp.tools.chat import chat_impl
from multi_mcp.tools.codereview import codereview_impl
from multi_mcp.tools.compare import compare_impl
from multi_mcp.tools.debate import debate_impl
from multi_mcp.tools.models import models_impl
from multi_mcp.utils.helpers import get_version
from multi_mcp.utils.mcp_decorator import mcp_monitor
from multi_mcp.utils.mcp_factory import create_mcp_wrapper
from multi_mcp.utils.paths import LOGS_DIR, ensure_logs_dir

logger = logging.getLogger(__name__)

mcp = FastMCP(settings.server_name)

codereview = create_mcp_wrapper(
    CodeReviewRequest,
    codereview_impl,
    """Systematic code review using external models.
Covers quality, security, performance, and architecture.""",
)
codereview = mcp.tool()(mcp_monitor(codereview))

chat = create_mcp_wrapper(
    ChatRequest,
    chat_impl,
    """General chat with AI assistant.
Supports multi-turn conversations with project context and file inclusion.""",
)
chat = mcp.tool()(mcp_monitor(chat))

compare = create_mcp_wrapper(
    CompareRequest,
    compare_impl,
    """Compare responses from multiple AI models.
Runs the same content against all specified models in parallel.
Supports multi-turn conversations with project context and file inclusion.""",
)
compare = mcp.tool()(mcp_monitor(compare))

debate = create_mcp_wrapper(
    DebateRequest,
    debate_impl,
    """Multi-model debate: Step 1 (independent answers) + Step 2 (debate/critique).
Each model provides independent answer, then reviews all responses and votes.""",
)
debate = mcp.tool()(mcp_monitor(debate))


@mcp.tool()
@mcp_monitor
async def version() -> dict:
    """
    Get server version, configuration details, and list of available tools.
    """
    tool_names: list[str] = []
    if hasattr(mcp, "_tools"):
        tools_dict = mcp._tools  # type: ignore[attr-defined]
        tool_names = [tool.name for tool in tools_dict.values()]

    return {
        "name": settings.server_name,
        "version": get_version(),
        "tools": sorted(tool_names) if tool_names else ["chat", "codereview", "compare", "debate", "models", "version"],
    }


@mcp.tool()
@mcp_monitor
async def models() -> dict:
    """
    List available AI models.
    Returns model names, aliases, provider, and configuration.
    """
    return await models_impl()


# Prompts are surfaced as slash commands; the host model only sees this text plus the user's
# request, so it must be explicit enough that the host does not substitute its own work.
EXTERNAL_MODELS_RULE = """The work must be done by the external models this server runs. Do not do it yourself, and do not spawn subagents, load skills or start other sessions as a substitute.
Model names in the user's request refer to this server's `models` tool (names or aliases such as `claude`, `codex`, `gemini`), not to the host's own model list.
If a requested model is missing or reports invalid credentials, stop and report it instead of substituting another model."""

USER_REQUEST_NOTE = "Treat any text the user added after this command as their request."


@mcp.prompt(name="codereview")
async def codereview_prompt() -> str:
    """Perform systematic code review"""
    return f"""Run a multi-model code review with this server's `codereview` tool.

{EXTERNAL_MODELS_RULE}

1. Call `codereview` with step_number=1 and no thread_id to start a new review; keep the returned thread_id.
2. Call step_number=2 with that thread_id, the absolute `base_path`, absolute paths of the files under review in `relevant_files`, and the review scope and focus in `content`.
3. Pass `models` only when the user names reviewers; otherwise omit it to use the configured defaults.
4. Report the findings attributed to each model, and name any model that failed.

{USER_REQUEST_NOTE}"""


@mcp.prompt(name="chat")
async def chat_prompt() -> str:
    """Chat with AI assistant"""
    return f"""Ask an external model through this server's `chat` tool.

{EXTERNAL_MODELS_RULE}

Pass `model` only when the user names one; otherwise omit it to use the configured default. Relay the model's answer attributed to it.

{USER_REQUEST_NOTE}"""


@mcp.prompt(name="compare")
async def compare_prompt() -> str:
    """Compare responses from multiple AI models"""
    return f"""Run the same request against several external models in parallel with this server's `compare` tool.

{EXTERNAL_MODELS_RULE}

Pass `models` only when the user names them; otherwise omit it to use the configured defaults. Present each model's answer attributed to it, then compare them.

{USER_REQUEST_NOTE}"""


@mcp.prompt(name="debate")
async def debate_prompt() -> str:
    """Multi-model debate with critique and voting"""
    return f"""Run a multi-model debate with this server's `debate` tool: the models answer independently, then critique each other and vote.

{EXTERNAL_MODELS_RULE}

Pass `models` only when the user names them; otherwise omit it to use the configured defaults. Report each model's position and the vote attributed to the models.

{USER_REQUEST_NOTE}"""


@mcp.prompt(name="models")
async def models_prompt() -> str:
    """List available AI models"""
    return "Use the models tool to see all available AI models, their aliases, and configuration."


@mcp.prompt(name="version")
async def version_prompt() -> str:
    """Get server version and info"""
    return "Use the version tool to see server version, configuration details, and available tools."


def main() -> None:
    """Entry point for multi-server CLI command."""
    ensure_logs_dir()  # Create logs directory on first use, not on import
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "server.log")],
    )
    logger.info(f"[SERVER] Starting {settings.server_name} on stdio")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
