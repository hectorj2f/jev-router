"""Which ladder the router is routing over.

policy.py owns the decision -- how hard the work is, which rung that earns --
and none of that is specific to a vendor. Everything that is (the rung names,
the slug that goes on the wire, the effort values, the shape of the transcript)
lives behind this package, so a second host is a module here rather than a
branch in the policy.
"""

from __future__ import annotations

import importlib
import os
import types

DEFAULT = "claude"
KNOWN = ("claude", "codex")


def get(name: str) -> types.ModuleType:
    """The provider module called `name`.

    An allow-list rather than a bare import_module: `name` arrives from the
    environment, and a typo should be an error naming the real options rather
    than an import of whatever else happens to be on sys.path.
    """
    if name not in KNOWN:
        raise ValueError(f"unknown provider {name!r}; known: {', '.join(KNOWN)}")
    return importlib.import_module(f"{__name__}.{name}")


def current() -> types.ModuleType:
    """The provider for this process, from $JEV_PROVIDER.

    An environment variable because the front ends are hooks: each runs in its
    own process with no argv to pass this on, so it is the one channel that
    reaches all of them at once.
    """
    return get(os.environ.get("JEV_PROVIDER", "").strip() or DEFAULT)
