#!/usr/bin/env python3
"""Register (or remove) the Agent-dispatch hook in ~/.claude/settings.json.

    ./install.py              # register, backing the file up first
    ./install.py --uninstall  # take it back out
    ./install.py --check      # say whether it is registered, change nothing

Hooks hot-reload, so this takes effect in sessions that are already running.

Writes an absolute path rather than `~/jev-router/...` so it works from a clone
anywhere, and is idempotent: re-running replaces our own entry instead of
stacking a second copy that would route every dispatch twice.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import stat
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
HOOK = HERE / "hook_agent.py"
SETTINGS = pathlib.Path.home() / ".claude" / "settings.json"
MATCHER = "^Agent$"


def is_ours(entry: dict) -> bool:
    """A PreToolUse entry pointing at this clone's hook, however it was written."""
    for h in entry.get("hooks", []):
        if "hook_agent.py" in str(h.get("command", "")):
            return True
    return False


def load() -> dict:
    if not SETTINGS.is_file():
        return {}
    try:
        return json.loads(SETTINGS.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{SETTINGS} is not valid JSON ({e}).\n"
                 f"  Fix it by hand first -- refusing to overwrite a file I can't read.")


def save(data: dict) -> None:
    if SETTINGS.is_file():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = SETTINGS.with_name(f"settings.json.bak-{stamp}")
        # Second granularity: install followed by uninstall lands on the same
        # name, and the backup overwriting the backup defeats the point of it.
        n = 2
        while backup.exists():
            backup = SETTINGS.with_name(f"settings.json.bak-{stamp}-{n}")
            n += 1
        shutil.copy(SETTINGS, backup)
        print(f"  backed up    {backup}")
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--uninstall", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()

    data = load()
    pre = data.get("hooks", {}).get("PreToolUse", [])
    installed = [e for e in pre if is_ours(e)]

    if a.check:
        print(f"{'registered' if installed else 'not registered'} in {SETTINGS}")
        return 0 if installed else 1

    if a.uninstall:
        if not installed:
            print("not registered; nothing to do")
            return 0
        data["hooks"]["PreToolUse"] = [e for e in pre if not is_ours(e)]
        # Don't leave empty scaffolding behind that wasn't there before us.
        if not data["hooks"]["PreToolUse"]:
            del data["hooks"]["PreToolUse"]
        if not data["hooks"]:
            del data["hooks"]
        save(data)
        print("  removed      the Agent hook. Takes effect immediately.")
        return 0

    if not HOOK.is_file():
        sys.exit(f"no hook at {HOOK}")
    HOOK.chmod(HOOK.stat().st_mode | stat.S_IXUSR)

    data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    data["hooks"]["PreToolUse"] = [e for e in data["hooks"]["PreToolUse"]
                                   if not is_ours(e)]
    data["hooks"]["PreToolUse"].append({
        "matcher": MATCHER,
        "hooks": [{"type": "command", "command": str(HOOK), "timeout": 8}],
    })
    save(data)

    print(f"  registered   {HOOK}")
    print(f"               on PreToolUse {MATCHER}, in {SETTINGS}")
    print("\nTakes effect now -- hooks reload per invocation, no restart.")
    print("Every subagent dispatch from here gets a model picked for it.")
    print("\nTo see it work, dispatch something mechanical and something deep, then:")
    print("    ./verify.py --project <your repo>")
    print("\nTo undo:")
    print("    ./install.py --uninstall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
