"""
Ethics Model API Module

This module provides a FastAPI-based web API for the Ethics Model,
enabling model inference, analysis, and visualization through RESTful endpoints.
"""

from .app import app

__all__ = ['app']
