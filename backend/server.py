"""duck-e-bugger backend — Phase 2.

Hosts a persistent Claude Agent SDK session and bridges it to the Tauri
shell over a local WebSocket. The shell sends transcripts (later: real STT
output, currently: whatever the shell forwards); this service streams back
assistant text and routes tool-permission checks through the shell instead
of auto-approving or auto-denying them.

Wire protocol over WebSocket, one connection per shell instance. Most
messages are JSON text frames; audio is sent as raw binary frames alongside
a JSON header pair so the shell knows how to decode/play them.

Shell -> backend (JSON):
    {"type": "transcript", "text": "..."}  # manual test path (no STT yet in the shell)
    {"type": "permission_response", "request_id": "...", "allow": bool, "message": "..."}

Backend -> shell (JSON):
    {"type": "speaking_start"}
    {"type": "assistant_text", "text": "..."}
    {"type": "audio_start", "sample_rate": int, "sample_width": 2, "channels": 1}
    {"type": "audio_end"}
    {"type": "speaking_end"}
    {"type": "turn_done"}
    {"type": "permission_request", "request_id": "...", "tool": "...", "input": {...}}
    {"type": "permission_resolved", "request_id": "..."}
    {"type": "error", "message": "..."}

Backend -> shell (binary): one or more raw 16-bit PCM chunks, sent between
an audio_start/audio_end pair — the synthesized audio for the assistant_text
that immediately preceded them.

Phase 1: transcripts also arrive from the local active-listening pipeline
(mic -> VAD -> speaker filter -> STT, see audio_pipeline.py).

Phase 3: incoming text (from either source) is routed through
resolve_or_queue() — if a permission confirmation is pending, it's checked
for a yes/no answer (voice_commands.interpret_yes_no) and consumed as the
answer instead of becoming a new agent turn. Access-level scoping (which
tools are auto-approved) is loaded from access_config.json, not hardcoded,
so it can be edited without touching code.

Phase 4: each assistant_text block is synthesized to speech (tts.py, Piper)
and streamed right after it as an audio_start/binary/audio_end sequence. A
single background worker does this, one block at a time in order, so audio
playback order always matches text order even though synthesis happens
after the text (and therefore the subtitle) is already on screen. If no
voice model is installed, tts.py's synthesize_pcm() returns None and no
audio messages are sent at all — text-only turns keep working as before.
After each audio block is sent, the listening pipeline is muted for its
playback duration (ActiveListeningPipeline.mute_for_playback) so the
backend mic doesn't transcribe the assistant's own voice.

Voice setup (2026-11-12): the agent gets Claude Code's system prompt plus
VOICE_PROMPT (spoken replies, no markdown, ask instead of guessing, known
locations), starts in access_config.json's working_dir, and loads no MCP
servers — see build_agent_options(). Every assistant text block goes through
speech_text.to_speakable() before it's shown or spoken. Per-turn timing and
each tool call's duration are logged at INFO ("timing: ..." / "tool ...").

Auth: ClaudeSDKClient shells out to a bundled Claude Code CLI, which
resolves credentials the same way the CLI itself does — an explicit
ANTHROPIC_API_KEY (metered billing) if set, otherwise the machine's
existing Claude Code/claude.ai subscription login
(~/.claude/.credentials.json), with no code-level branching needed here.
See describe_auth() below for the startup log line.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
import websockets

from audio_pipeline import ActiveListeningPipeline
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from speech_text import to_speakable
from tts import SpeechSynthesizer
from voice_commands import interpret_yes_no

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("duck-e-bugger-backend")

HOST = "127.0.0.1"
PORT = 8765

# Conservative default: read-only tools auto-approved. Everything else
# (Write, Edit, Bash, ...) falls through to can_use_tool and gets a
# confirmation round-trip through the shell before it runs. Overridden by
# access_config.json when present — see load_allowed_tools().
DEFAULT_ALLOWED_TOOLS = ["Read", "Glob", "Grep"]
ACCESS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "access_config.json")

PERMISSION_TIMEOUT_SECONDS = 120
CLAUDE_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"


def describe_auth() -> str:
    """Logged once at startup — which credential source the bundled Claude
    Code CLI will use. Mirrors the CLI's own precedence: an explicit
    ANTHROPIC_API_KEY always wins (metered billing) even if a subscription
    login is also present; otherwise it falls back to the machine's
    existing Claude Code/claude.ai login, if any."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "using ANTHROPIC_API_KEY (metered billing)"
    if CLAUDE_CREDENTIALS_PATH.exists():
        return f"using Claude Code subscription login ({CLAUDE_CREDENTIALS_PATH})"
    return (
        "no ANTHROPIC_API_KEY and no Claude Code login found — agent turns "
        "will fail until one is set up (run `claude` to log in, or set "
        "ANTHROPIC_API_KEY in .env)"
    )


VOICE_PROMPT = """\
You are Duck-e-bugger, a voice assistant. The user talks to you through a \
microphone: their words reach you as speech-to-text transcripts. Everything \
you write is read aloud by a text-to-speech voice and shown as a one-line \
subtitle.

How to reply:
- Write plain spoken sentences. No markdown: no asterisks, headings, bullet \
lists, tables or code blocks. Don't read out full file paths or URLs; name the \
file, folder or site instead.
- Keep it short: one to three sentences, unless the user asks for more detail.
- Before a tool call that may take a while, say in a few words what you're \
about to do.
- Transcripts can mishear names (for example "Duckley Debugger" for \
"Duck-e-bugger"). If a name doesn't match anything, try the closest match in \
the known locations before searching further.
- If it's unclear which project, folder or file the user means, ask one short \
question instead of guessing.
- Never search the whole home directory or the whole OneDrive folder; those \
searches take minutes. Start from the known locations below and keep searches \
narrow.
"""


def load_access_config() -> dict:
    """access_config.json, or {} if it's missing or unreadable (every
    setting has a default)."""
    if os.path.exists(ACCESS_CONFIG_PATH):
        try:
            with open(ACCESS_CONFIG_PATH, encoding="utf-8") as f:
                config = json.load(f)
            if isinstance(config, dict):
                return config
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("failed to read %s (%s), using defaults", ACCESS_CONFIG_PATH, exc)
    return {}


def load_allowed_tools(config: dict) -> list[str]:
    tools = config.get("allowed_tools")
    return tools if isinstance(tools, list) else DEFAULT_ALLOWED_TOOLS


def _existing_dir(raw: str) -> Path | None:
    path = Path(raw).expanduser()
    if path.is_dir():
        return path
    log.warning("access_config.json: %s is not a folder, skipping it", path)
    return None


def build_agent_options(config: dict, can_use_tool) -> ClaudeAgentOptions:
    """Options for the voice agent: Claude Code's own system prompt plus
    VOICE_PROMPT and the known locations, starting in working_dir, and with
    no MCP servers. Without strict_mcp_config the bundled CLI also loads
    every claude.ai connector on the account (Gmail, Claude Docs, ... — 18 of
    them when checked on 2026-11-12), which sent a request about "the
    project" off to Claude Docs and added tokens to every turn."""
    locations = {
        label: path
        for label, raw in (config.get("known_locations") or {}).items()
        if (path := _existing_dir(raw)) is not None
    }
    prompt = VOICE_PROMPT
    if locations:
        prompt += "\nKnown locations:\n" + "".join(
            f"- {label}: {path}\n" for label, path in locations.items()
        )

    working_dir = _existing_dir(config["working_dir"]) if config.get("working_dir") else None
    return ClaudeAgentOptions(
        allowed_tools=load_allowed_tools(config),
        permission_mode="default",
        can_use_tool=can_use_tool,
        system_prompt={"type": "preset", "preset": "claude_code", "append": prompt},
        cwd=working_dir,
        add_dirs=[p for p in locations.values() if p != working_dir],
        mcp_servers={},
        strict_mcp_config=True,
        env={"ENABLE_CLAUDEAI_MCP_SERVERS": "false"},
    )


def summarize_tool_input(input_data: dict) -> str:
    """One short line for the log — never a whole file's contents."""
    for key in ("command", "file_path", "path", "pattern", "url", "query"):
        if key in input_data:
            text = str(input_data[key])
            return text if len(text) <= 120 else text[:117] + "..."
    return ", ".join(input_data) or "(no input)"


async def handle_connection(
    websocket,
    pipeline: ActiveListeningPipeline,
    synthesizer: SpeechSynthesizer,
    config: dict,
) -> None:
    log.info("shell connected")
    pending_permissions: dict[str, asyncio.Future] = {}
    pending_confirmation_request_id: str | None = None
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    # (text to speak, monotonic start time of the turn it belongs to)
    tts_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()

    async def resolve_permission(request_id: str, allow: bool, message: str | None = None) -> bool:
        """Resolves a pending can_use_tool future and notifies the shell.
        Returns False if there was nothing pending under that id."""
        nonlocal pending_confirmation_request_id
        future = pending_permissions.pop(request_id, None)
        if not future or future.done():
            return False
        future.set_result((allow, message))
        if pending_confirmation_request_id == request_id:
            pending_confirmation_request_id = None
        await websocket.send(json.dumps({"type": "permission_resolved", "request_id": request_id}))
        return True

    async def resolve_or_queue(text: str) -> None:
        """Incoming text (voice or typed) either answers a pending
        confirmation or becomes a new agent turn — never both."""
        if pending_confirmation_request_id is not None:
            decision = interpret_yes_no(text)
            if decision is not None:
                await resolve_permission(pending_confirmation_request_id, decision)
            else:
                log.info("heard %r while awaiting confirmation — no clear yes/no, ignoring", text)
            return
        await transcript_queue.put(text)

    async def run_active_listening() -> None:
        async for text in pipeline.listen():
            log.info("active listening transcript: %s", text)
            await resolve_or_queue(text)

    async def run_tts() -> None:
        """Synthesizes and streams one queued text block at a time, in
        order — keeps audio playback order matching text order even though
        synthesis (and therefore each block's audio) always lags a bit
        behind its already-displayed subtitle."""
        last_timed_turn = None
        while True:
            text, turn_started_at = await tts_queue.get()
            pcm = await asyncio.to_thread(synthesizer.synthesize_pcm, text)
            if not pcm:
                continue
            if turn_started_at != last_timed_turn:
                last_timed_turn = turn_started_at
                log.info(
                    "timing: first audio sent %.1fs after the request",
                    time.monotonic() - turn_started_at,
                )
            await websocket.send(json.dumps({
                "type": "audio_start",
                "sample_rate": synthesizer.sample_rate,
                "sample_width": 2,
                "channels": 1,
            }))
            await websocket.send(pcm)
            await websocket.send(json.dumps({"type": "audio_end"}))
            # The shell starts playing on audio_end; stop listening until it's done.
            pipeline.mute_for_playback(len(pcm) / 2 / synthesizer.sample_rate)

    async def can_use_tool(tool_name, input_data, context):
        nonlocal pending_confirmation_request_id
        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        pending_permissions[request_id] = future
        pending_confirmation_request_id = request_id
        await websocket.send(json.dumps({
            "type": "permission_request",
            "request_id": request_id,
            "tool": tool_name,
            "input": input_data,
        }))
        try:
            allow, message = await asyncio.wait_for(future, timeout=PERMISSION_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            pending_permissions.pop(request_id, None)
            if pending_confirmation_request_id == request_id:
                pending_confirmation_request_id = None
            await websocket.send(json.dumps({"type": "permission_resolved", "request_id": request_id}))
            return PermissionResultDeny(message="No response from shell — denied by timeout")
        if allow:
            return PermissionResultAllow(updated_input=input_data)
        return PermissionResultDeny(message=message or "Denied by user")

    options = build_agent_options(config, can_use_tool)

    async def run_turns(client: ClaudeSDKClient) -> None:
        while True:
            text = await transcript_queue.get()
            await websocket.send(json.dumps({"type": "speaking_start"}))
            started_at = time.monotonic()
            first_text_s: float | None = None
            running_tools: dict[str, tuple[str, float]] = {}  # tool_use_id -> (name, start)
            tool_times: list[str] = []
            try:
                await client.query(text)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                running_tools[block.id] = (block.name, time.monotonic())
                                log.info(
                                    "tool %s started: %s",
                                    block.name,
                                    summarize_tool_input(block.input),
                                )
                            elif hasattr(block, "text"):
                                spoken = to_speakable(block.text)
                                if not spoken:
                                    continue
                                if first_text_s is None:
                                    first_text_s = time.monotonic() - started_at
                                await websocket.send(json.dumps({
                                    "type": "assistant_text",
                                    "text": spoken,
                                }))
                                if synthesizer.available:
                                    await tts_queue.put((spoken, started_at))
                    elif isinstance(message, UserMessage) and isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock) and block.tool_use_id in running_tools:
                                name, tool_start = running_tools.pop(block.tool_use_id)
                                took = time.monotonic() - tool_start
                                tool_times.append(f"{name} {took:.1f}s")
                                log.info(
                                    "tool %s finished in %.1fs%s",
                                    name,
                                    took,
                                    " (error)" if block.is_error else "",
                                )
                    elif isinstance(message, ResultMessage) and message.subtype != "success":
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"agent turn ended with an error: {message.subtype}",
                        }))
            except Exception as exc:  # noqa: BLE001 — surface any turn failure to the shell
                log.exception("turn failed")
                await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
            finally:
                log.info(
                    "timing: turn took %.1fs (first text %s; tools: %s)",
                    time.monotonic() - started_at,
                    f"{first_text_s:.1f}s" if first_text_s is not None else "none",
                    ", ".join(tool_times) or "none",
                )
                await websocket.send(json.dumps({"type": "speaking_end"}))
                await websocket.send(json.dumps({"type": "turn_done"}))

    pipeline.start()
    async with ClaudeSDKClient(options=options) as client:
        try:
            mcp = await client.get_mcp_status()
            log.info(
                "agent session ready: %d MCP servers, working dir %s",
                len(mcp.get("mcpServers", [])),
                options.cwd or os.getcwd(),
            )
        except Exception as exc:  # noqa: BLE001 — diagnostics only
            log.warning("couldn't read MCP status: %s", exc)
        turn_task = asyncio.create_task(run_turns(client))
        listening_task = asyncio.create_task(run_active_listening())
        tts_task = asyncio.create_task(run_tts())
        try:
            async for raw in websocket:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "transcript":
                    await resolve_or_queue(msg["text"])
                elif msg_type == "permission_response":
                    await resolve_permission(
                        msg.get("request_id"), msg.get("allow", False), msg.get("message")
                    )
                else:
                    log.warning("unknown message type: %s", msg_type)
        finally:
            turn_task.cancel()
            listening_task.cancel()
            tts_task.cancel()
            pipeline.stop()
            log.info("shell disconnected")


async def main() -> None:
    log.info("Claude auth: %s", describe_auth())
    config = load_access_config()

    log.info("loading active-listening pipeline (VAD + speaker filter + STT)...")
    pipeline = ActiveListeningPipeline(vocabulary=config.get("vocabulary"))
    log.info("active-listening pipeline ready")

    log.info("loading TTS voice...")
    synthesizer = SpeechSynthesizer()
    if not synthesizer.available:
        log.warning("TTS running text-only — no voice model installed (see download_voice.py)")

    async def handler(websocket):
        await handle_connection(websocket, pipeline, synthesizer, config)

    async with websockets.serve(handler, HOST, PORT):
        log.info("listening on ws://%s:%s", HOST, PORT)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
