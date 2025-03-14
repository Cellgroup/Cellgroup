"""Test suite for the cellgroup package.

This package contains all tests for the cellgroup synthetic cell simulation framework.
Tests are organized into:
- Unit tests for individual components
- Integration tests for component interactions
- System tests for complete workflows
"""

import os
import sys

# Add parent directory to Python path to allow importing cellgroup package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)