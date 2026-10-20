#!/usr/bin/env python3
"""Run every code example in docs/source/examples/ and report the results.

Design goals
------------
1.  The examples that readers see ARE the examples that get executed.  Pages
    embed them with ``literalinclude``, so there is no copy/paste drift.
2.  Examples that deliberately demonstrate an error are *still tested*: the
    file starts with a marker comment and the runner asserts that it fails for
    the right reason.
3.  Exit code is 0 only when every example behaved as declared, so CI and the
    Read the Docs ``pre_build`` job can rely on it.

Marker comments understood at the top of an example file
--------------------------------------------------------
    # REQUIRES: gpu              skip unless CUDA is available
    # REQUIRES: numpy            skip if numpy is missing
    # TIMEOUT: 120               per-file timeout override (seconds)
    # EXPECTED FAILURE: RuntimeError
    # EXPECTED OUTPUT: <substring that must appear in stdout or stderr>

Usage
-----
    python tests/run_examples.py                 # run everything
    python tests/run_examples.py tensors autograd # only files matching a name
    python tests/run_examples.py --json out.json # also write a machine report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "docs" / "source" / "examples"
DEFAULT_TIMEOUT = 180.0

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"


# --------------------------------------------------------------------------- #
# Marker parsing
# --------------------------------------------------------------------------- #

_MARKER_RE = re.compile(
    r"^\s*#\s*(REQUIRES|TIMEOUT|EXPECTED FAILURE|EXPECTED OUTPUT)\s*:?\s*(.*)$"
)


@dataclass
class Spec:
    """What an example file declared about itself."""

    path: Path
    requires: list[str] = field(default_factory=list)
    timeout: float = DEFAULT_TIMEOUT
    expected_failure: str | None = None
    expected_output: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.stem


def parse_spec(path: Path) -> Spec:
    """Read the leading marker comments of an example file."""
    spec = Spec(path=path)
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines()[:25]:
        m = _MARKER_RE.match(line)
        if not m:
            # Stop at the first non-marker, non-blank, non-shebang line.
            if line.strip() and not line.startswith("#!") and not line.startswith('"""'):
                break
            continue
        key, value = m.group(1).upper(), m.group(2).strip()
        if key == "REQUIRES":
            spec.requires.extend(p.strip().lower() for p in value.split(",") if p.strip())
        elif key == "TIMEOUT":
            try:
                spec.timeout = float(value)
            except ValueError:
                pass
        elif key == "EXPECTED FAILURE":
            spec.expected_failure = value or "Exception"
        elif key == "EXPECTED OUTPUT":
            if value:
                spec.expected_output.append(value)
    return spec


# --------------------------------------------------------------------------- #
# Requirement checks
# --------------------------------------------------------------------------- #

def check_requirements(spec: Spec) -> str | None:
    """Return a skip reason, or None when the example can run here."""
    for req in spec.requires:
        if req in {"gpu", "cuda"}:
            try:
                import torch  # noqa: WPS433 - import is the check

                if not torch.cuda.is_available():
                    return "no CUDA device available"
            except Exception as exc:  # pragma: no cover - torch missing entirely
                return f"torch import failed: {exc}"
        elif req == "cpu":
            continue
        else:
            try:
                __import__(req)
            except ImportError:
                return f"module '{req}' is not installed"
    return None


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    name: str
    path: str
    status: str
    returncode: int | None
    duration_s: float
    detail: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


def _tail(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…\n" + text[-limit:]


def run_example(spec: Spec, workdir: Path, verbose: bool = True) -> Result:
    """Execute one example in a clean temp cwd and grade the outcome."""
    env = os.environ.copy()
    # Keep runs reproducible and stop BLAS from spawning a thread per core.
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Examples must never write into the repository.
    env["KH_EXAMPLE_TMPDIR"] = str(workdir)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(spec.path)],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=spec.timeout,
        )
        returncode: int | None = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return Result(
            name=spec.name,
            path=str(spec.path.relative_to(REPO_ROOT)),
            status=STATUS_FAIL,
            returncode=None,
            duration_s=round(time.monotonic() - started, 2),
            detail=f"timed out after {spec.timeout:.0f}s",
            stdout_tail=_tail(exc.stdout or "" if isinstance(exc.stdout, str) else ""),
            stderr_tail=_tail(exc.stderr or "" if isinstance(exc.stderr, str) else ""),
        )
    duration = round(time.monotonic() - started, 2)

    detail_parts: list[str] = []
    ok = True

    if spec.expected_failure is None:
        if returncode != 0:
            ok = False
            detail_parts.append(
                f"expected success but exited {returncode}: "
                f"{stderr.strip().splitlines()[-1] if stderr.strip() else 'no stderr'}"
            )
    else:
        if returncode == 0:
            ok = False
            detail_parts.append(
                f"expected failure ({spec.expected_failure}) but the script exited 0"
            )
        elif spec.expected_failure.lower() not in (stderr + stdout).lower():
            ok = False
            detail_parts.append(
                f"failed with the wrong error: wanted '{spec.expected_failure}'"
            )

    combined = stdout + "\n" + stderr
    for needle in spec.expected_output:
        if needle not in combined:
            ok = False
            detail_parts.append(f"missing expected output: '{needle}'")

    return Result(
        name=spec.name,
        path=str(spec.path.relative_to(REPO_ROOT)),
        status=STATUS_PASS if ok else STATUS_FAIL,
        returncode=returncode,
        duration_s=duration,
        detail="; ".join(detail_parts),
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filters", nargs="*", help="substring filter on the file name")
    parser.add_argument("--json", metavar="PATH", help="write a JSON report")
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    parser.add_argument(
        "--keep-tmp", action="store_true", help="do not delete the scratch directory"
    )
    args = parser.parse_args(argv)

    if not EXAMPLES_DIR.is_dir():
        print(f"no examples directory at {EXAMPLES_DIR}", file=sys.stderr)
        return 2

    files = sorted(EXAMPLES_DIR.glob("*.py"))
    if args.filters:
        files = [f for f in files if any(s in f.name for s in args.filters)]
    if not files:
        print("no example matched the filters", file=sys.stderr)
        return 2

    have_torch = shutil.which(sys.executable) is not None
    try:
        import torch  # noqa: F401

        torch_version = torch.__version__
    except Exception:
        torch_version = "NOT INSTALLED"
        have_torch = False

    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch_version}")
    print(f"Examples dir  : {EXAMPLES_DIR.relative_to(REPO_ROOT)}")
    print(f"Examples found: {len(files)}")
    print("-" * 72)

    tmpdir = Path(tempfile.mkdtemp(prefix="kh-pytorch-examples-"))
    results: list[Result] = []
    try:
        for path in files:
            spec = parse_spec(path)
            reason = None if have_torch else "torch is not installed"
            reason = reason or check_requirements(spec)
            if reason:
                results.append(
                    Result(
                        name=spec.name,
                        path=str(path.relative_to(REPO_ROOT)),
                        status=STATUS_SKIP,
                        returncode=None,
                        duration_s=0.0,
                        detail=reason,
                    )
                )
                print(f"SKIP  {spec.name:<34} ({reason})")
                continue

            res = run_example(spec, tmpdir)
            results.append(res)
            tag = {"pass": "PASS ", "fail": "FAIL ", "skip": "SKIP "}[res.status]
            shown = res.status != STATUS_PASS or not args.quiet
            if shown:
                line = f"{tag} {res.name:<34} {res.duration_s:>7.2f}s  rc={res.returncode}"
                if spec.expected_failure:
                    line += f"  (expects {spec.expected_failure})"
                print(line)
                if res.status == STATUS_FAIL:
                    if res.detail:
                        print(f"      ! {res.detail}")
                    if res.stderr_tail:
                        for ln in res.stderr_tail.splitlines()[-12:]:
                            print(f"      | {ln}")
    finally:
        if args.keep_tmp:
            print(f"scratch dir kept at: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)

    n_pass = sum(r.status == STATUS_PASS for r in results)
    n_fail = sum(r.status == STATUS_FAIL for r in results)
    n_skip = sum(r.status == STATUS_SKIP for r in results)
    print("-" * 72)
    print(f"passed {n_pass} · failed {n_fail} · skipped {n_skip} · total {len(results)}")

    if args.json:
        payload = {
            "python": sys.version.split()[0],
            "torch": torch_version,
            "counts": {"pass": n_pass, "fail": n_fail, "skip": n_skip, "total": len(results)},
            "results": [asdict(r) for r in results],
        }
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON report: {args.json}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
