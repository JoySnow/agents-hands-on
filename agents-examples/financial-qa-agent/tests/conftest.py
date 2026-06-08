"""pytest conftest for financial-qa-agent tests.

Adds the package root to sys.path so that 'from src import ...' works.
"""

import os
import sys

# Add the financial-qa-agent directory to sys.path
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
