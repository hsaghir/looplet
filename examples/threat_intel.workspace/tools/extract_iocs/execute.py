"""extract_iocs — re-import. This tool takes ctx: ToolContext to call
``ctx.llm.generate(...)`` for severity classification."""

from examples.threat_intel.agent import extract_iocs as _spec

execute = _spec.execute
