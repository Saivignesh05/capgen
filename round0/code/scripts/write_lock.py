"""Write an environment's lock file: the exact version of every installed package. Run with that environment's Python.

Started by `./round0 setup` the first time it builds an environment. On its own:
    /path/to/environments/generator/bin/python -m scripts.write_lock environments/generator/requirements.lock.txt
"""
import importlib.metadata
import os
import re
import subprocess
import sys


def lock_lines() -> list[str]:
    """`pip freeze`, made portable: packages conda installed show up as "name @ file:///..." with
    a path that exists only on this machine; those become "name==version". CAREFUL L1"""
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    lines = []
    for line in frozen:
        match = re.fullmatch(r"(\S+) @ file://\S+", line.strip())
        if match:
            name = match.group(1)
            lines.append(f"{name}=={importlib.metadata.version(name)}")
        elif line.strip():
            lines.append(line.strip())
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.write_lock <path of requirements.lock.txt>")
    path = sys.argv[1]
    lines = lock_lines()
    if not lines:
        raise SystemExit("pip freeze listed no packages: not writing an empty lock file")
    # Write to a temporary name, then rename, so a crash never leaves an empty or half-written lock.
    with open(path + ".partial", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(path + ".partial", path)
    print(f"Wrote {path} ({len(lines)} packages)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# L1   Packages installed from git keep their exact commit ("name @ git+https://...@<commit>"), so
#      they install the same everywhere. Only local file:// paths are replaced.
# ---------------------------------------------------------------------------------------------
