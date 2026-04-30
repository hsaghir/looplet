"""fetch_feed — re-import the original top-level function so its
typing imports + helper closures stay intact."""

from examples.threat_intel.agent import fetch_feed as _spec

execute = _spec.execute
