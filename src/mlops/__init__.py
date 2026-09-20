"""Shared library for the Airflow pipeline and the serving API.

Every module here is plain Python: nothing imports Airflow, so each pipeline
step can be tested on its own without a scheduler.
"""

__all__ = [
    "config",
    "drift",
    "extract",
    "registry",
    "report",
    "scale",
    "split",
    "train",
    "validate",
]
