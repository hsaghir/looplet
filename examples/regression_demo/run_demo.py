"""Compatibility entry point for the packaged regression proof."""

from looplet.examples.regression_demo import DemoResult, main, render, run_demo

__all__ = ["DemoResult", "main", "render", "run_demo"]


if __name__ == "__main__":
    raise SystemExit(main())
