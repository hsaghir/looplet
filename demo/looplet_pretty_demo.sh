#!/usr/bin/env bash
# Reproducible README hero demo for the CLI pretty trace.
#
# Run as:
#   bash demo/looplet_pretty_demo.sh
#
# To record an asciinema cast:
#   asciinema rec demo/looplet_pretty.cast --overwrite --cols 88 --rows 24 \
#       --command "uv run --quiet --active python -m looplet.examples.pretty_demo"
#
# To convert the cast to a GIF for embedding in READMEs / docs:
#   agg demo/looplet_pretty.cast demo/looplet_pretty.gif \
#       --theme monokai --cols 88 --rows 24 --font-size 15 --fps-cap 12 \
#       --idle-time-limit 1 --last-frame-duration 2

set -euo pipefail

LOOPLET_REPO="${LOOPLET_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$LOOPLET_REPO"
uv run --quiet --active python -m looplet.examples.pretty_demo "$@"