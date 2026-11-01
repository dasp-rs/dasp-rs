#!/usr/bin/env python3
"""Check the release evidence manifest or execute its local validation gates.

  python scripts/validate_release.py --check
  python scripts/validate_release.py --run --reference-python PATH_TO_PINNED_PYTHON

Execution writes per-gate logs and a JSON report. Remote CI results, release
version review and publication are outside this local command's scope.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/release.json"
PREFIXES = {"src/core/io.rs": "io", "src/core/ops.rs": "ops",
            "src/signal_processing/": "proc", "src/features/": "feat",
            "src/magnitude/": "mag", "src/pitch_core/": "pitch",
            "src/signal_generation/": "generate", "src/utils/": "util"}


def inventory():
    result = {}
    for path in sorted((ROOT / "src").rglob("*.rs")):
        relative = path.relative_to(ROOT).as_posix()
        namespace = next((value for key, value in PREFIXES.items() if relative.startswith(key)), None)
        if namespace:
            for name in re.findall(r"^pub (?:const )?fn\s+(\w+)", path.read_text(), re.M):
                key = f"{namespace}::{name}"
                if key in result:
                    raise ValueError(f"Ambiguous exported operation: {key}")
                result[key] = relative
    return result


def check_manifest():
    data = json.loads(MANIFEST.read_text())
    if data["schema_version"] != 1 or data["api_inventory"] != inventory():
        raise ValueError("Public free-function inventory changed; review validation/release.json")
    ids = [gate["id"] for gate in data["gates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate release gate")
    for area in data["evidence_areas"]:
        if not area["scope"] or not area["files"]:
            raise ValueError(f"Missing evidence scope: {area['area']}")
        for file in area["files"]:
            if not (ROOT / file).is_file():
                raise ValueError(f"Missing evidence: {file}")
    workflow = (ROOT / data["required_ci"]["workflow"]).read_text()
    for job in data["required_ci"]["jobs"]:
        if not re.search(rf"^  {re.escape(job)}:\s*$", workflow, re.M):
            raise ValueError(f"Required CI job absent: {job}")
    if not data["remaining_independent_reference_gaps"] or not data["limitations"]:
        raise ValueError("Coverage gaps and limitations must remain explicit")
    print(f"Manifest valid: {len(data['api_inventory'])} free functions, "
          f"{len(data['evidence_areas'])} evidence areas, {len(ids)} local gates")
    return data


def source_hashes():
    files = [*ROOT.glob("src/**/*.rs"), *ROOT.glob("tests/**/*.rs"),
             *ROOT.glob("benches/**/*.rs"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("tests/fixtures/**/*"), MANIFEST, ROOT / "Cargo.toml",
             ROOT / ".github/workflows/ci.yml", ROOT / "README.md", ROOT / "CHANGELOG.md"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files)) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check manifest and evidence paths only (default)")
    mode.add_argument("--run", action="store_true", help="Run all local validation gates serially")
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=ROOT / "target/release-validation/report.json")
    args = parser.parse_args()
    data = check_manifest()
    if not args.run:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
    # Keep doctest scratch out of small shared /tmp filesystems.
    temporary = ROOT / "target/doctest-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        environment["TMPDIR"] = str(temporary)
    before = source_hashes()
    report = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": before, "gates": [], "remote_ci_verified": False,
              "release_ready": None,
              "notes": ["Local gates only; remote CI and maintainer release/version review remain required.",
                        "Passing tests does not remove the manifest's explicit reference gaps and limits."]}
    substitutions = {"{python}": sys.executable, "{reference_python}": args.reference_python}
    for gate in data["gates"]:
        command = [substitutions.get(part, part) for part in gate["command"]]
        env = dict(environment, **gate.get("env", {}))
        print(f"Running release gate: {gate['id']}", flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        log = args.output.parent / (gate["id"] + ".log")
        log.write_text(result.stdout)
        passed = result.returncode == 0
        ignored = None
        if "maximum_ignored" in gate:
            matches = re.findall(r"test result: ok\..*? (\d+) ignored", result.stdout)
            ignored = sum(map(int, matches))
            passed = passed and bool(matches) and ignored <= gate["maximum_ignored"]
        report["gates"].append({"id": gate["id"], "command": command,
                                "seconds": time.monotonic() - started, "passed": passed,
                                "returncode": result.returncode, "ignored_tests": ignored,
                                "log": str(log)})
        if not passed:
            print(f"FAILED {gate['id']}; see {log}", file=sys.stderr)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["source_changed_during_run"] = before != source_hashes()
    report["local_gates_passed"] = all(g["passed"] for g in report["gates"]) and not report["source_changed_during_run"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Local release gates passed={report['local_gates_passed']}; report: {args.output}")
    return 0 if report["local_gates_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
