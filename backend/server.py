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
)
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


def load_allowed_tools() -> list[str]:
    if os.path.exists(ACCESS_CONFIG_PATH):
        try:
            with open(ACCESS_CONFIG_PATH, encoding="utf-8") as f:
                config = json.load(f)
            tools = config.get("allowed_tools")
            if isinstance(tools, list):
                return tools
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("failed to read %s (%s), using defaults", ACCESS_CONFIG_PATH, exc)
    return DEFAULT_ALLOWED_TOOLS


async def handle_connection(
    websocket, pipeline: ActiveListeningPipeline, synthesizer: SpeechSynthesizer
) -> None:
    log.info("shell connected")
    pending_permissions: dict[str, asyncio.Future] = {}
    pending_confirmation_request_id: str | None = None
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    tts_queue: asyncio.Queue[str] = asyncio.Queue()

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
        while True:
            text = await tts_queue.get()
            pcm = await asyncio.to_thread(synthesizer.synthesize_pcm, text)
            if not pcm:
                continue
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

    options = ClaudeAgentOptions(
        allowed_tools=load_allowed_tools(),
        permission_mode="default",
        can_use_tool=can_use_tool,
    )

    async def run_turns(client: ClaudeSDKClient) -> None:
        while True:
            text = await transcript_queue.get()
            await websocket.send(json.dumps({"type": "speaking_start"}))
            try:
                await client.query(text)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if hasattr(block, "text"):
                                await websocket.send(json.dumps({
                                    "type": "assistant_text",
                                    "text": block.text,
                                }))
                                if synthesizer.available:
                                    await tts_queue.put(block.text)
                    elif isinstance(message, ResultMessage) and message.subtype != "success":
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"agent turn ended with an error: {message.subtype}",
                        }))
            except Exception as exc:  # noqa: BLE001 — surface any turn failure to the shell
                log.exception("turn failed")
                await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
            finally:
                await websocket.send(json.dumps({"type": "speaking_end"}))
                await websocket.send(json.dumps({"type": "turn_done"}))

    pipeline.start()
    async with ClaudeSDKClient(options=options) as client:
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

    log.info("loading active-listening pipeline (VAD + speaker filter + STT)...")
    pipeline = ActiveListeningPipeline()
    log.info("active-listening pipeline ready")

    log.info("loading TTS voice...")
    synthesizer = SpeechSynthesizer()
    if not synthesizer.available:
        log.warning("TTS running text-only — no voice model installed (see download_voice.py)")

    async def handler(websocket):
        await handle_connection(websocket, pipeline, synthesizer)

    async with websockets.serve(handler, HOST, PORT):
        log.info("listening on ws://%s:%s", HOST, PORT)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
