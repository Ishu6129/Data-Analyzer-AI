# This makes 'src' a Python package
from .analyzer import DataAnalyzer
from .main import main

__all__ = ['DataAnalyzer', 'main']