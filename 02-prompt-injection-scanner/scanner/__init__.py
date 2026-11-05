"""Hybrid (rules + machine learning) prompt injection scanner."""

from .core import ScanResult, scan, scan_document

__all__ = ["scan", "scan_document", "ScanResult"]
