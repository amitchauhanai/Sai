# AGENTS.md

Guide for AI coding agents and future contributors working on this project.

## Project Snapshot

Bro is a voice-native macOS OS co-pilot. The local client listens for a wake word or manual start, captures microphone audio and screenshots, executes mouse/keyboard actions, and shows a macOS overlay. The server receives the client WebSocket stream, uses ElevenLabs for speech-to-text, Amazon Nova/OpenRouter models for routing and vision reasoning, then sends executable commands back to the client.

Primary flow:

1. `client/wake_word.py` detects activation and connects to `ws://localhost:8080/ws/agent`.
2. The server in `server/main.py` requests screenshots, streams/handles audio, interprets intent, and chooses simple command execution or the vision agent loop.
3. The client executes returned actions through PyAutoGUI, macOS APIs, and AppleScript helpers.

## Repository Map

- `README.md` - main project overview, setup, architecture, permissions, and demo workflow.
- `setup_mac.sh` - one-shot macOS setup script for client/server virtual environments.
- `client/wake_word.py` - local macOS client entry point: wake mode, mic streaming, screenshots, overlay, app context, and action execution.
- `client/url_utils.py` - URL normalization for spoken/browser commands.
- `client/HeyBro_mac.ppn` - custom Picovoice Porcupine wake-word model.
- `client/requirements.txt` - client dependencies.
- `client/.env.example` - client environment template.
- `server/main.py` - FastAPI app, WebSocket endpoint, STT integration, model calls, deterministic simple actions, vision loop, memory, and guardrails.
- `server/test_client.py` - lightweight WebSocket protocol tester.
- `server/requirements.txt` - server dependencies.
- `server/.env.example` - server environment template.
- `tests/test_smoke.py` - unittest smoke tests for deterministic routing and URL normalization.
- `current_status.md`, `DEVPOST.md`, `presentation.html` - project/storytelling artifacts.

Do not treat `client/venv/` or `server/venv/` as source code. They are local virtual environments.

## Environment

Use Python 3.11+ on macOS.

Recommended setup from the repo root:

```bash
bash setup_mac.sh
```

Manual setup:

```bash
cd server
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env

cd ../client
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Required server keys in `server/.env`:

```bash
AMAZON_NOVA_API_KEY=...
OPENROUTER_API_KEY=...
ELEVENLABS_API_KEY=...
DEEPGRAM_API_KEY=...
GEMINI_API_KEY=...
BRO_STT_PROVIDER=deepgram
```

Client defaults to manual activation:

```bash
BRO_WAKE_MODE=manual
```

For real wake-word mode, set:

```bash
BRO_WAKE_MODE=auto
PICOVOICE_ACCESS_KEY=...
```

Never commit real `.env` files or API keys.

## Run Commands

Start the server:

```bash
cd server
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8080
```

Start the client in a second terminal:

```bash
cd client
./venv/bin/python wake_word.py
```

Run smoke tests from the repo root:

```bash
python3 -m unittest tests/test_smoke.py
```

Run the WebSocket test client while the server is running:

```bash
cd server
./venv/bin/python test_client.py
```

## macOS Permissions

The client needs these permissions for the terminal or IDE used to run it:

- Accessibility - PyAutoGUI mouse/keyboard actions.
- Screen Recording - screenshot capture through `screencapture`/`mss`.
- Microphone - PyAudio wake/audio streaming.

If permission prompts do not appear, reset them:

```bash
tccutil reset Accessibility
tccutil reset ScreenCapture
tccutil reset Microphone
```

Then fully quit and reopen the terminal or IDE.

## Development Notes

- Keep the client and server contract explicit. WebSocket messages should remain JSON-serializable and easy to inspect in logs.
- `AgentAction` and related Pydantic models in `server/main.py` are the server-side command schema. Update validation when adding commands.
- Coordinate-based actions use normalized `[0, 1000]` values server-side and are mapped to real screen dimensions client-side.
- For simple commands, prefer deterministic handlers before LLM calls when behavior is stable and testable.
- For advanced UI tasks, preserve the plan-act-verify loop and screenshot refresh after actions.
- Avoid hardcoded screen sizes. The client already has dynamic logical screen-size helpers.
- Be careful with sleeps in the client. Existing code prefers polling/readiness checks where possible.
- Keep macOS-specific imports optional so basic imports/tests can run outside a full macOS UI session where possible.
- Do not edit generated virtual environment files under `client/venv/` or `server/venv/`.

## Testing Guidance

Before finishing changes, run the smoke tests when possible:

```bash
python3 -m unittest tests/test_smoke.py
```

For server changes, at minimum verify that `server/main.py` imports cleanly and the FastAPI app starts.

For client changes, prefer syntax/import checks first because full behavior depends on macOS permissions and hardware:

```bash
cd client
./venv/bin/python -m py_compile wake_word.py url_utils.py
```

For command-routing changes, add or update tests in `tests/test_smoke.py`.

## Common Change Areas

- Add a new simple command:
  Update deterministic routing/action logic in `server/main.py`, then add a smoke test.

- Add a new executable client action:
  Update the server action schema, command generation prompts/handlers, and the client execution path in `client/wake_word.py`.

- Improve URL handling:
  Update `client/url_utils.py` and the URL normalization tests.

- Adjust model/provider settings:
  Keep environment-driven values near the top of `server/main.py` and document new `.env` keys in `server/.env.example`.

- Change wake-word behavior:
  Work in `client/wake_word.py` and keep `BRO_WAKE_MODE=manual` usable for development without Picovoice.

## Safety Rules

- Do not store or print API keys in logs.
- Do not make destructive OS actions the default for vague voice commands.
- Keep confirmation or clear intent checks for risky workflows.
- Preserve the client single-instance lock so two clients do not fight over mic, stdin, or the overlay.
- Treat screenshots and app context as sensitive user data.
