"""
ecm_digital_thread
===================

An intelligent Digital Thread framework for Engineering Change Management
(ECM) in heavy machinery manufacturing, integrating Digital Thread
traceability, NLP-based change-request classification, and
KPI/bottleneck analytics.

This package provides:
    * a synthetic demonstration dataset generator (6-month window),
    * a Digital Thread integration / traceability layer,
    * an NLP change-request classification pipeline with NER,
    * KPI / DAX-style analytics and a Bottleneck Severity Score,
    * an open-source analytics dashboard,
    * scripts that compute the demonstration tables and figures, and
    * the paper's published Tables 1-4 (``published``), kept separate.

See README.md for usage, and docs/methodology.md for the underlying
equations and design decisions.
"""

__version__ = "1.1.0"

from . import config  # noqa: F401
