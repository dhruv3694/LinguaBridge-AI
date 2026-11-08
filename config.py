"""Compatibility shim: some code may import from config.py; prefer config package."""
from config.settings import *  # noqa: F401,F403
from config.model_config import *  # noqa: F401,F403
