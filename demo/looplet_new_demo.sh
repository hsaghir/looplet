#!/usr/bin/env bash
# Reproducible CLI demo for ``looplet new`` + ``looplet run-workspace``.
#
# Run as:
#   bash demo/looplet_new_demo.sh
#
# To record an asciinema cast:
#   asciinema rec --idle-time-limit 2 demo/looplet_new.cast \
#       --command "bash demo/looplet_new_demo.sh"
#
# To convert the cast to a GIF for embedding in READMEs / blog posts:
#   pip install --user agg
#   agg demo/looplet_new.cast demo/looplet_new.gif --speed 2 --theme monokai
#
# The script uses a few short ``sleep`` calls so the recording reads
# naturally (humans need a beat to absorb each command); set
# ``LOOPLET_DEMO_FAST=1`` to disable them when running for real.

set -euo pipefail

PAUSE() {
    if [[ -z "${LOOPLET_DEMO_FAST:-}" ]]; then
        sleep "${1:-1}"
    fi
}

TYPE() {
    # Pseudo-typed echo so the cast looks like a human typing.
    if [[ -n "${LOOPLET_DEMO_FAST:-}" ]]; then
        echo "$ $1"
        return
    fi
    printf "$ "
    local s="$1"
    for ((i=0; i<${#s}; i++)); do
        printf "%s" "${s:$i:1}"
        sleep 0.02
    done
    printf "\n"
}

# Tunables — caller may override via env.
: "${OPENAI_BASE_URL:=http://127.0.0.1:19823/v1}"
: "${OPENAI_API_KEY:=copilot}"
: "${OPENAI_MODEL:=claude-sonnet-4.6}"
: "${LOOPLET_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL LOOPLET_REPO

DEMO_DIR="${LOOPLET_DEMO_DIR:-/tmp/looplet_demo_$$}"
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

clear

echo "# looplet — agents from a paragraph, in one command"
PAUSE 2
echo
echo "# Step 1: configure any OpenAI-compatible endpoint."
PAUSE 1
TYPE "export OPENAI_BASE_URL=$OPENAI_BASE_URL"
TYPE "export OPENAI_API_KEY=***"
TYPE "export OPENAI_MODEL=$OPENAI_MODEL"
PAUSE 1
echo
echo "# Step 2: describe the agent we want."
PAUSE 1

BRIEF="An agent that takes a URL and returns the page title and a 2-sentence summary of the content. Tools: fetch_url(url) using stdlib urllib, extract_title(html), summarize_text(text) using ctx.llm."

TYPE "looplet new \"$BRIEF\" ./url_summarizer.workspace"
PAUSE 1
echo

# Real factory build. ``uv run --project ...`` keeps the demo working
# even when the user hasn't yet ``pip install``ed looplet onto PATH.
uv run --project "$LOOPLET_REPO" looplet new "$BRIEF" ./url_summarizer.workspace --quiet

echo
PAUSE 2
echo "# Step 3: give the produced agent a real task."
PAUSE 1
TYPE 'looplet run-workspace ./url_summarizer.workspace "Summarize https://example.com"'
PAUSE 1
echo

# Real run.
uv run --project "$LOOPLET_REPO" looplet run-workspace ./url_summarizer.workspace "Summarize https://example.com" --quiet

echo
PAUSE 2
echo "# 6 minutes from blank dir to a working agent that fetches, parses, summarizes."
echo "# That's the whole pitch."
PAUSE 3
