"""
Conftest for pytest configuration
"""

import pytest
import os

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "false"
