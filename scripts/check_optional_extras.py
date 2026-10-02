"""Run the test suite as if the optional extras were not installed.

`pyproject.toml` declares tensorflow, selenium and requests as optional, and
importing them inside functions is what makes that true. The trouble is that it
is easy to break without noticing: everything passes locally, because the
packages are installed locally, and the only person who finds out is whoever
installs without the extras.

CI has a job for this. This is the same check locally, without uninstalling
anything:

    python scripts/check_optional_extras.py

It blocks the imports with a meta-path finder, so it cannot leave your working
install in a different state than it found it.

This has caught the same mistake four times now, in two repositories: an
optional dependency imported before the code had checked whether it was needed.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

#: Everything `pyproject.toml` calls optional.
BLOCKED = ("tensorflow", "keras", "selenium", "requests", "cv2", "PIL")

SITECUSTOMIZE = '''
import sys

BLOCKED = {blocked!r}


class _Block:
    """Refuse the optional imports, as if they were never installed."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("No module named %r" % name.split(".")[0])
        return None


sys.meta_path.insert(0, _Block())
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        (pathlib.Path(temporary) / "sitecustomize.py").write_text(
            SITECUSTOMIZE.format(blocked=BLOCKED), encoding="utf-8"
        )

        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{temporary}{os.pathsep}{existing}" if existing else temporary
        )

        print(f"running the suite with {', '.join(BLOCKED)} blocked\n")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rs", *sys.argv[1:]],
            env=environment,
            check=False,
        )

    if result.returncode == 0:
        print("\nthe optional extras really are optional")
    else:
        print("\nsomething in src/ needs an optional dependency it should not")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
