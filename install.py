#!/usr/bin/env python3
"""Register (or remove) the dispatch hook with Claude Code or with Codex.

    ./install.py              # register with Claude Code, backing the file up first
    ./install.py --codex      # register with Codex instead
    ./install.py --uninstall  # take it back out
    ./install.py --check      # say whether it is registered, change nothing

Hooks hot-reload on both hosts, so this takes effect in sessions that are
already running.

Writes an absolute path rather than `~/jev-router/...` so it works from a clone
anywhere, and is idempotent: re-running replaces our own entry instead of
stacking a second copy that would route every dispatch twice.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import stat
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
HOOK = HERE / "hook_agent.py"
CLAUDE_HOME = pathlib.Path.home() / ".claude"
SETTINGS = CLAUDE_HOME / "settings.json"
MATCHER = "^Agent$"
# Where `claude plugin install` records what it installed. Read-only here, and
# only to answer "is the mod also on", which register.js warns about in prose
# and nothing checked.
INSTALLED_PLUGINS = CLAUDE_HOME / "plugins" / "installed_plugins.json"
PLUGIN_MANIFEST = HERE / ".claude-plugin" / "plugin.json"

CODEX_HOOK = HERE / "hook_codex.py"
CODEX_HOME = pathlib.Path(os.environ.get("CODEX_HOME")
                          or pathlib.Path.home() / ".codex")
# Codex reads hook declarations from the folder holding the layer's config, so
# this is a sibling of config.toml rather than the hooks/hooks.json a plugin
# ships. Writing here keeps us out of a TOML file we would have to rewrite
# wholesale to edit: JSON round-trips, config.toml does not.
CODEX_HOOKS_JSON = CODEX_HOME / "hooks.json"
CODEX_CONFIG = CODEX_HOME / "config.toml"
# A matcher of only word characters compiles to an exact name match rather than
# a regex, and V2 hands the hook the namespace and the tool name run together
# (`collaborationspawn_agent`), so the bare name -- which reads like the `^Agent$`
# Claude Code needs -- is the one spelling that can never fire. The leading `.*`
# makes this match whether or not the engine anchors the pattern itself.
CODEX_MATCHER = r".*spawn_agent$"
DESCRIPTION = "jev-router: a model picked for each spawn_agent dispatch"

CODEX_TOML_SNIPPET = f"""\
[[hooks.PreToolUse]]
matcher = "{CODEX_MATCHER}"

[[hooks.PreToolUse.hooks]]
type = "command"
command = "{CODEX_HOOK}"
timeout = 8\
"""


def is_ours(entry: dict, hook_name: str) -> bool:
    """A PreToolUse entry pointing at this clone's hook, however it was written."""
    for h in entry.get("hooks", []):
        if hook_name in str(h.get("command", "")):
            return True
    return False


def load(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON ({e}).\n"
                 f"  Fix it by hand first -- refusing to overwrite a file I can't read.")


def save(path: pathlib.Path, data: dict) -> None:
    if path.is_file():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        # Second granularity: install followed by uninstall lands on the same
        # name, and the backup overwriting the backup defeats the point of it.
        n = 2
        while backup.exists():
            backup = path.with_name(f"{path.name}.bak-{stamp}-{n}")
            n += 1
        shutil.copy(path, backup)
        print(f"  backed up    {backup}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def registered(data: dict, hook_name: str) -> list[dict]:
    pre = data.get("hooks", {}).get("PreToolUse", [])
    return [e for e in pre if is_ours(e, hook_name)]


def deregister(data: dict, hook_name: str) -> None:
    pre = data["hooks"]["PreToolUse"]
    data["hooks"]["PreToolUse"] = [e for e in pre if not is_ours(e, hook_name)]
    # Don't leave empty scaffolding behind that wasn't there before us.
    if not data["hooks"]["PreToolUse"]:
        del data["hooks"]["PreToolUse"]
    if not data["hooks"]:
        del data["hooks"]


def codex_config_declares_hooks() -> bool:
    """Whether ~/.codex/config.toml already carries a `[hooks]` table.

    Codex loads hooks.json and config.toml for the same layer and warns when
    both have entries, but it still runs both -- a hook declared twice prices
    and rewrites every dispatch twice. Worth saying before we add the second.
    """
    try:
        text = CODEX_CONFIG.read_text()
    except OSError:
        return False
    try:
        import tomllib
    except ImportError:
        return any(line.lstrip().startswith(("[hooks", "[[hooks"))
                   for line in text.splitlines())
    try:
        return bool(tomllib.loads(text).get("hooks"))
    except tomllib.TOMLDecodeError:
        return False


def mod_installed() -> bool:
    """Whether the mod in hooks/register.js is installed as a plugin too.

    The two are independent front ends onto one policy and register.js says in
    prose to run one or the other -- both on means every dispatch is scored
    twice, logged twice, and re-tiered by whichever ran last. That is the same
    double-declaration install.py already warns about on Codex, and the only
    reason it went unwarned here is that the two live in different files.

    Best-effort by construction. `claude --plugin-dir ~/jev-router` loads the
    mod for one session and writes nothing down, so a False is "nothing on
    record", never "definitely not running". Any surprise in the file is read
    the same way: this is advice printed beside an install, and it has no
    business failing one.
    """
    try:
        name = json.loads(PLUGIN_MANIFEST.read_text())["name"]
        entries = json.loads(INSTALLED_PLUGINS.read_text())["plugins"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    # Checked here rather than left to the `.items()` below, which is evaluated
    # inside the generator and so outside the try above -- `{"plugins": []}`
    # raised AttributeError straight through this function and out of
    # claude_main, aborting the install before the hook was written. A warning
    # printed beside an install must not be able to stop one.
    if not isinstance(entries, dict):
        return False
    # Keyed `<plugin>@<marketplace>`, and the marketplace is whatever the user
    # added this clone under, so match on the plugin half. The installPath arm
    # is for a clone loaded from source; every entry a marketplace install
    # writes points into ~/.claude/plugins/cache, so on this machine it is the
    # name that matches and the path that never does.
    return any(key.split("@")[0] == name
               or any(str(i.get("installPath")) == str(HERE)
                      for i in val if isinstance(i, dict))
               for key, val in entries.items() if isinstance(val, list))


def claude_main(a: argparse.Namespace) -> int:
    data = load(SETTINGS)
    installed = registered(data, HOOK.name)

    if a.check:
        print(f"{'registered' if installed else 'not registered'} in {SETTINGS}")
        if installed and mod_installed():
            print(f"  note: the mod is installed too, and both would route "
                  f"every dispatch")
        return 0 if installed else 1

    if a.uninstall:
        if not installed:
            print("not registered; nothing to do")
            return 0
        deregister(data, HOOK.name)
        save(SETTINGS, data)
        print("  removed      the Agent hook. Takes effect immediately.")
        return 0

    if not HOOK.is_file():
        sys.exit(f"no hook at {HOOK}")
    HOOK.chmod(HOOK.stat().st_mode | stat.S_IXUSR)

    if mod_installed():
        # Not fatal, and not a refusal: on a CLI below 2.1.287 the mod is
        # installed and simply never loads, which is a perfectly ordinary
        # reason to want the hook as well.
        print("  careful      the jev-router mod is installed too.")
        print("               On 2.1.287+ both load, and every dispatch gets "
              "scored and")
        print("               logged twice. Run one or the other:")
        print("                   claude plugin uninstall jev-router   "
              "# keep the hook")
        print("                   ./install.py --uninstall             "
              "# keep the mod\n")

    data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    data["hooks"]["PreToolUse"] = [e for e in data["hooks"]["PreToolUse"]
                                   if not is_ours(e, HOOK.name)]
    data["hooks"]["PreToolUse"].append({
        "matcher": MATCHER,
        "hooks": [{"type": "command", "command": str(HOOK), "timeout": 8}],
    })
    save(SETTINGS, data)

    print(f"  registered   {HOOK}")
    print(f"               on PreToolUse {MATCHER}, in {SETTINGS}")
    print("\nTakes effect now -- hooks reload per invocation, no restart.")
    print("Every subagent dispatch from here gets a model picked for it.")
    print("\nTo see it work, dispatch something mechanical and something deep, then:")
    print("    ./verify.py --project <your repo>")
    print("\nTo undo:")
    print("    ./install.py --uninstall")
    return 0


def codex_main(a: argparse.Namespace) -> int:
    data = load(CODEX_HOOKS_JSON)
    installed = registered(data, CODEX_HOOK.name)

    if a.check:
        print(f"{'registered' if installed else 'not registered'} in {CODEX_HOOKS_JSON}")
        if not installed and codex_config_declares_hooks():
            print(f"  note: {CODEX_CONFIG} declares hooks of its own; if one of "
                  f"them is {CODEX_HOOK.name}, this check cannot see it")
        return 0 if installed else 1

    if a.uninstall:
        if not installed:
            print("not registered; nothing to do")
            return 0
        deregister(data, CODEX_HOOK.name)
        if data.get("description") == DESCRIPTION:
            del data["description"]
        save(CODEX_HOOKS_JSON, data)
        if not data:
            # Nothing of theirs left in it, so the whole file is scaffolding we
            # put there. The save above is what makes removing it undoable.
            CODEX_HOOKS_JSON.unlink()
        print("  removed      the spawn_agent hook. Takes effect immediately.")
        return 0

    if not CODEX_HOOK.is_file():
        sys.exit(f"no hook at {CODEX_HOOK}")
    CODEX_HOOK.chmod(CODEX_HOOK.stat().st_mode | stat.S_IXUSR)

    if codex_config_declares_hooks():
        # Not fatal: config.toml may well declare hooks that are nothing to do
        # with us. But we cannot merge into it safely -- rewriting TOML from a
        # parsed dict loses comments and ordering -- so if theirs is already a
        # jev-router entry, adding ours here doubles it.
        print(f"  careful      {CODEX_CONFIG} declares hooks too.")
        print(f"               Codex runs both files. If {CODEX_HOOK.name} is "
              f"already in there,")
        print( "               remove it before this one starts routing "
               "everything twice.\n")

    data.setdefault("description", DESCRIPTION)
    data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    data["hooks"]["PreToolUse"] = [e for e in data["hooks"]["PreToolUse"]
                                   if not is_ours(e, CODEX_HOOK.name)]
    data["hooks"]["PreToolUse"].append({
        "matcher": CODEX_MATCHER,
        "hooks": [{"type": "command", "command": str(CODEX_HOOK),
                   "timeout": 8,
                   "statusMessage": "jev: pricing the dispatch"}],
    })
    save(CODEX_HOOKS_JSON, data)

    print(f"  registered   {CODEX_HOOK}")
    print(f"               on PreToolUse {CODEX_MATCHER}, in {CODEX_HOOKS_JSON}")
    print("\nCodex will ask you to trust this hook the first time it loads it,")
    print("once per change to the command. Answer it from the session; there is")
    print("a --dangerously-bypass-hook-trust that skips the prompt, and skipping")
    print("it means the next edit to this file runs unreviewed too.")
    print("\nRouting only fires on a fresh spawn (fork_turns = \"none\"). A")
    print("full-history fork inherits the parent's model by design, and the")
    print("hook leaves those alone.")
    print("\nIf you would rather keep hooks in config.toml, uninstall this and")
    print(f"add to {CODEX_CONFIG} by hand -- nothing here will write that file:\n")
    print("\n".join(f"    {line}" if line else ""
                    for line in CODEX_TOML_SNIPPET.splitlines()))
    print("\nTo undo:")
    print("    ./install.py --codex --uninstall")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--codex", action="store_true",
                   help="target OpenAI Codex instead of Claude Code")
    p.add_argument("--uninstall", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    return codex_main(a) if a.codex else claude_main(a)


if __name__ == "__main__":
    sys.exit(main())
