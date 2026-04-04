"""
Pytest configuration for aerial_gym tests.

These tests are pre-migration regression tests — they run against the CURRENT codebase
to establish a baseline. After Isaac Lab migration, re-run to confirm no regressions.

NOTE: Tests that require a live simulator (test_quaternion_convention, test_reward_sanity)
need the Isaac Sim Python interpreter:
    /home/cow_server01/pg-dev/isaacsim/python.sh -m pytest tests/

Tests that only need torch (test_joint_ordering math, test_math_utils) run under any Python.
"""
import sys
import os
from unittest.mock import MagicMock

# aerial_gym/__init__.py imports isaacgym unconditionally, and utils/math.py imports
# pytorch3d — both block tests when running outside Isaac Gym. Stub them out BEFORE
# any aerial_gym imports. The unit tests here only use submodules that don't exercise
# gymapi/gymtorch/pytorch3d handles at runtime.
_STUBS = [
    "isaacgym",
    "isaacgym.gymapi",
    "isaacgym.gymtorch",
    "isaacgym.gymutil",
    "isaacgym.torch_utils",
    "pytorch3d",
    "pytorch3d.transforms",
    "urdfpy",
]
for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Add project root to path so `aerial_gym` is importable without pip install
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
