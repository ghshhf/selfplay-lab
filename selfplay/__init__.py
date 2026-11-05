"""Shared infrastructure for the self-play lab.

The old domains are script-style and each carries its own loop.  This package
holds the pieces that should be common: run manifests, evaluator health,
checkpointed recordings, and (in ``selfplay.nn``) the NumPy neural stack.
"""

__version__ = "0.1.0"
