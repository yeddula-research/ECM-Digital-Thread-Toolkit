"""
ecm_digital_thread
===================

An intelligent Digital Thread framework for Engineering Change Management
(ECM) in heavy machinery manufacturing, integrating Digital Thread
traceability, NLP-based change-request classification, and real-time
KPI/bottleneck analytics.

This package provides:
    * an ECM dataset generator covering a 6-month operational period,
    * a Digital Thread integration / traceability layer,
    * an NLP change-request classification pipeline with NER,
    * KPI / DAX-style analytics and a Bottleneck Severity Score,
    * an open-source analytics dashboard,
    * scripts that generate Tables 1-4 and Figures 2-5.

See README.md for usage, and docs/methodology.md for the underlying
equations and design decisions.
"""

__version__ = "1.0.0"

from . import config  # noqa: F401
