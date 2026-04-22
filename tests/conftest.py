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
import types
import importlib.abc
import importlib.machinery
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# _AutoMod — a types.ModuleType that satisfies Python's module protocol so
# that `import omni.kit.commands` works even though there's no real package.
# ---------------------------------------------------------------------------

class _AutoMod(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        pkg = name.rsplit(".", 1)[0] if "." in name else name
        # Set string attrs explicitly so os.fspath / inspect don't crash
        for attr, val in [
            ("__file__",    f"<stub:{name}>"),
            ("__path__",    []),
            ("__spec__",    None),
            ("__loader__",  None),
            ("__package__", pkg),
        ]:
            object.__setattr__(self, attr, val)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        full = f"{self.__name__}.{name}"
        if full not in sys.modules:
            sys.modules[full] = _AutoMod(full)
        obj = sys.modules[full]
        object.__setattr__(self, name, obj)
        return obj

    def __call__(self, *args, **kwargs):  return MagicMock()
    def __bool__(self):                    return True
    def __iter__(self):                    return iter([])


# ---------------------------------------------------------------------------
# Import hook — intercepts `import omni.*` (and other heavy namespaces) and
# returns _AutoMod stubs so we never try to load Isaac Sim extensions.
# ---------------------------------------------------------------------------

_AUTO_PREFIXES = (
    "omni",
    "carb",
    "pxr",           # USD Python bindings
    "isaaclab.sim",  # pulls in mesh converters → omni.kit.commands etc.
    "isaaclab.assets",
    "isaaclab.scene",
    "isaaclab.sensors",
    "isaaclab.envs.mdp",
)


class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def _should_stub(self, fullname: str) -> bool:
        return any(
            fullname == p or fullname.startswith(p + ".")
            for p in _AUTO_PREFIXES
        )

    def find_spec(self, fullname, path, target=None):
        if self._should_stub(fullname):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        if spec.name not in sys.modules:
            sys.modules[spec.name] = _AutoMod(spec.name)
        return sys.modules[spec.name]

    def exec_module(self, module):
        pass   # nothing to execute — the stub is already populated


# Install at the FRONT so it wins before the real finders
sys.meta_path.insert(0, _AutoStubFinder())


# ---------------------------------------------------------------------------
# Flat MagicMock stubs for non-namespace packages
# ---------------------------------------------------------------------------

_FLAT_STUBS = [
    "isaacgym",
    "isaacgym.gymapi",
    "isaacgym.gymtorch",
    "isaacgym.gymutil",
    "isaacgym.torch_utils",
    "pytorch3d",
    "pytorch3d.transforms",
    "urdfpy",
    "warp",
    "warp.context",
]
for _mod in _FLAT_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Add project root to path so `aerial_gym` is importable without pip install
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
