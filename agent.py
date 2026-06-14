#!/usr/bin/env python3
"""
Autonomous B3 Explorer improvement agent.
Runs until Anthropic credits are exhausted.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import anthropic

APP_DIR = Path(__file__).parent

# ── Tools ──────────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Read a file relative to the app directory."""
    target = (APP_DIR / path).resolve()
    if not str(target).startswith(str(APP_DIR)):
        return "Error: path outside app directory"
    try:
        return target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {path}: {e}"


def write_file(path: str, content: str) -> str:
    """Write a file relative to the app directory."""
    target = (APP_DIR / path).resolve()
    if not str(target).startswith(str(APP_DIR)):
        return "Error: path outside app directory"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def list_files(directory: str = ".") -> str:
    """List files in a directory relative to the app directory."""
    target = (APP_DIR / directory).resolve()
    if not str(target).startswith(str(APP_DIR)):
        return "Error: path outside app directory"
    try:
        entries = []
        for p in sorted(target.iterdir()):
            rel = p.relative_to(APP_DIR)
            entries.append(f"{'[dir] ' if p.is_dir() else '      '}{rel}")
        return "\n".join(entries) if entries else "(empty)"
    except Exception as e:
        return f"Error: {e}"


def run_python(code: str) -> str:
    """Run Python code and return stdout + stderr (timeout 30s)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
            cwd=str(APP_DIR),
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        if result.returncode != 0:
            parts.append(f"exit code: {result.returncode}")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


def search_in_files(pattern: str, file_glob: str = "*.py") -> str:
    """Search for a pattern in files matching a glob."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include", file_glob, pattern, str(APP_DIR)],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip()
        # Make paths relative
        lines = lines.replace(str(APP_DIR) + "/", "")
        return lines[:3000] if lines else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# ── Tool registry ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a file from the B3 Explorer app directory. "
            "Call this to inspect source code before making changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to app root (e.g. 'Main_Page.py', 'pages/1_Portfolio.py', 'utils/db.py')"}
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write or overwrite a file in the B3 Explorer app directory. "
            "Use this to apply improvements to source files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to app root"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory of the B3 Explorer app.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path relative to app root (default: '.')"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": "Run Python code to validate logic, test functions, or check syntax.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_in_files",
        "description": "Search for a text pattern across source files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Grep pattern to search for"},
                "file_glob": {"type": "string", "description": "File glob pattern (default: '*.py')"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
]

TOOL_FN = {
    "read_file": lambda inp: read_file(inp["path"]),
    "write_file": lambda inp: write_file(inp["path"], inp["content"]),
    "list_files": lambda inp: list_files(inp.get("directory", ".")),
    "run_python": lambda inp: run_python(inp["code"]),
    "search_in_files": lambda inp: search_in_files(inp["pattern"], inp.get("file_glob", "*.py")),
}

# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM = """You are an autonomous software engineer improving the B3 Explorer app — \
a Brazilian stock portfolio analysis platform built with Streamlit.

Your mission: continuously improve the app's code quality, features, UX, \
error handling, and performance. Work autonomously across multiple turns until your \
credits run out.

App structure:
- Main_Page.py — landing page, watchlist, fundamental screening (1642 lines)
- pages/1_Portfolio.py — portfolio builder, HRP optimization, risk metrics
- pages/2_Simulação.py — Monte Carlo simulations
- pages/3_Notícias.py — news aggregation and sentiment analysis
- pages/4_Valuation.py — DCF valuation, WACC, EV/EBITDA
- utils/db.py — SQLite cache and watchlist persistence
- style.css — dark theme (Obsidian Neo-Financial)

Priority improvement areas (tackle in order of impact):
1. Error handling — add try/except around yfinance/Fundamentus API calls that crash silently
2. Input validation — validate ticker symbols and date ranges before API calls
3. Cache pruning — prevent http_cache.sqlite from growing unboundedly (add size limits/TTL cleanup)
4. Performance — cache expensive computations with @st.cache_data where missing
5. UX polish — improve error messages shown to users, add loading spinners
6. Code quality — extract repeated logic into utils/, add type hints

Rules:
- Always read a file before modifying it
- Make focused, surgical changes — don't rewrite entire files
- Keep the Streamlit structure and the dark theme intact
- After each improvement, start the next one immediately
- Be explicit about what you changed and why in your text output
"""

# ── Agentic loop ────────────────────────────────────────────────────────────

def run_agent():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": (
                "Start improving the B3 Explorer app. "
                "Begin by listing the files, then pick the highest-impact improvement, "
                "implement it, and immediately move on to the next one. "
                "Keep going until your credits are exhausted."
            ),
        }
    ]

    turn = 0
    total_input_tokens = 0
    total_output_tokens = 0

    print("=" * 60)
    print("B3 Explorer Autonomous Improvement Agent")
    print("Runs until Anthropic credits are exhausted.")
    print("=" * 60)

    while True:
        turn += 1
        print(f"\n[Turn {turn}] Calling Claude...")

        try:
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                response = stream.get_final_message()

        except anthropic.AuthenticationError:
            print("\n[STOP] Authentication error — check ANTHROPIC_API_KEY.")
            break
        except anthropic.PermissionDeniedError as e:
            # Anthropic returns 403 when prepaid credits are exhausted
            print(f"\n[STOP] Credits exhausted (HTTP 403): {e}")
            break
        except anthropic.RateLimitError as e:
            print(f"\n[WAIT] Rate limited: {e}. Sleeping 60s...")
            time.sleep(60)
            continue
        except anthropic.APIStatusError as e:
            if e.status_code == 402:
                # Some billing systems return 402 Payment Required
                print(f"\n[STOP] Payment required — credits exhausted (HTTP 402): {e.message}")
                break
            print(f"\n[ERROR] API error {e.status_code}: {e.message}. Retrying in 10s...")
            time.sleep(10)
            continue
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        # Track usage
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        print(f"  Input: {response.usage.input_tokens} | Output: {response.usage.output_tokens} | "
              f"Total so far: {total_input_tokens}in / {total_output_tokens}out")

        # Print text content
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude]\n{block.text}")

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = TOOL_FN.get(block.name)
                if fn is None:
                    result = f"Unknown tool: {block.name}"
                else:
                    print(f"\n  [Tool] {block.name}({json.dumps(block.input)[:120]}...)")
                    result = fn(block.input)
                    preview = result[:200].replace("\n", " ")
                    print(f"  [Result] {preview}{'...' if len(result) > 200 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Agent finished its thought — nudge it to keep going
            print("\n[Nudge] Agent finished a turn. Prompting to continue...")
            messages.append({
                "role": "user",
                "content": "Good work. Keep going — implement the next improvement now.",
            })

        elif response.stop_reason == "max_tokens":
            # Hit output limit — just continue
            messages.append({
                "role": "user",
                "content": "Continue from where you left off.",
            })

        else:
            print(f"\n[STOP] Unexpected stop reason: {response.stop_reason}")
            break

        # Safety: cap context to avoid unbounded growth (keep last 30 messages)
        if len(messages) > 60:
            # Keep system + first user message + last 30
            messages = messages[:1] + messages[-30:]

    print(f"\n{'='*60}")
    print(f"Agent stopped. Total tokens used: {total_input_tokens} input / {total_output_tokens} output")
    print(f"Turns completed: {turn}")


if __name__ == "__main__":
    run_agent()
