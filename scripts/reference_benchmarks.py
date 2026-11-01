#!/usr/bin/env python3
"""Run numerical reference tests and write an inspectable JSON accuracy report.

The default path needs only Python's standard library and Cargo. Fixture
regeneration additionally needs tests/fixtures/requirements-reference.txt.
"""
import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
GROUPS = ("core", "spectral", "transforms", "cqt", "cqt_sweep", "cqt_inverse",
          "iir", "reassignment", "rhythm", "recovery", "corpus", "pitch")
LINE = re.compile(r"(?:REFERENCE ([\w.-]+)|(diagnostic_[\w.-]+)): ([^\r\n]+)")
NUMBER = re.compile(r"([\w]+)=([-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf(?:inity)?))(?![\w.])", re.IGNORECASE)
TARGET = re.compile(r"Running tests[/\\]([\w]+)\.rs")
TEST = re.compile(r"^test ([\w:]+) \.\.\.")


MANIFEST_PATH = FIXTURES / "reference_inputs.json"


def fixture_values(path):
    with path.open() as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        values = {}
        for row in rows:
            if "case" in row:
                for column, field in enumerate(("yin_hz", "pyin_hz", "voiced")):
                    key = (row["case"], int(row["frame"]), column)
                    value = (float(row[field]), 0.0)
                    if key in values or not all(math.isfinite(v) for v in value):
                        raise ValueError(f"Duplicate/nonfinite pitch fixture value: {path}: {key}")
                    values[key] = value
                continue
            key = (row["name"], int(row["row"]), int(row["col"]))
            value = (float(row["real"]), float(row["imag"]))
            if key in values or not all(math.isfinite(v) for v in value):
                raise ValueError(f"Duplicate/nonfinite fixture value: {path}: {key}")
            values[key] = value
    if not values:
        raise ValueError(f"Empty fixture: {path}")
    return values


def fixture_path(group):
    name = {"corpus": "audio/corpus_reference.csv", "pitch": "pitch_librosa_0_11.csv"}.get(group, f"{group}_reference.csv")
    return FIXTURES / name


def input_inventory(group, values):
    entry = json.loads(MANIFEST_PATH.read_text())["groups"][group]
    names = {key[0] for key in values}
    classified = set(entry["f32_inputs"]) | set(entry["exact_inputs"]) | set(entry["reference_outputs"])
    if names != classified:
        raise ValueError(f"Unaudited or stale fixture names in {group}: {sorted(names ^ classified)}")
    for name, columns in entry.get("exact_columns", {}).items():
        if name not in names or any(not any(k[0] == name and k[2] == c for k in values) for c in columns):
            raise ValueError(f"Stale exact input coordinates: {group}/{name}")
    return entry


def f32_bits(value):
    # Rust tests cast CSV f64 to f32, including supplied magnitude/spectrum
    # matrices. Compare the public input bits, not an absolute error floor.
    try:
        result = struct.pack("!f", value)
    except OverflowError as error:
        raise ValueError(f"Unrepresentable f32 fixture input: {value}") from error
    return result


def compare_fixtures(original, generated, group=None):
    if group is None:
        group = next(g for g in GROUPS if fixture_path(g).name == original.name)
    expected, actual = fixture_values(original), fixture_values(generated)
    inventory = input_inventory(group, expected)
    if expected.keys() != actual.keys():
        raise ValueError(f"Fixture names, shapes, or coordinates changed: {original.name}")
    f32_inputs, exact_inputs = set(inventory["f32_inputs"]), set(inventory["exact_inputs"])
    for key, values in expected.items():
        is_f32 = key[0] in f32_inputs
        is_exact = key[0] in exact_inputs or key[2] in inventory.get("exact_columns", {}).get(key[0], [])
        # This comparison-only SVD solution has condition ~1.97e7. Its
        # kappa*eps64 sensitivity justifies a separate regeneration allowance.
        ill_conditioned = group == "cqt" and key[0] == "narrow_inverse_solution"
        rtol, atol = (1e-7, 1e-7) if ill_conditioned else (1e-10, 5e-12)
        for x, y in zip(values, actual[key]):
            equal = f32_bits(x) == f32_bits(y) if is_f32 else x == y if is_exact else math.isclose(x, y, rel_tol=rtol, abs_tol=atol)
            if not equal:
                raise ValueError(f"Fixture drift: {original.name}: {key}: {x} != {y}")
    print(f"Verified {original.name}: {len(expected)} values", flush=True)


def generate(python, check, groups):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="dasp-references-") as directory:
        for group in groups:
            if group == "corpus":
                # Audio assets have their own explicit regeneration workflow.
                # Validate their hashes and recompute summaries from the pinned
                # checked-in clips, without fetching or replacing recordings.
                result = subprocess.run([python, str(ROOT / "scripts/prepare_audio_corpus.py"), "--check-references"],
                                        cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, check=False)
                print(result.stdout, end="", flush=True)
                result.check_returncode()
                continue
            source = FIXTURES / f"generate_{group}_reference.py"
            script = source
            if check:
                # Generators derive their output path from __file__; copying only
                # the script keeps check mode from modifying tracked fixtures.
                script = Path(directory) / source.name
                shutil.copyfile(source, script)
            result = subprocess.run([python, str(script)], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, check=False)
            print(result.stdout, end="", flush=True)
            result.check_returncode()
            if check:
                original = fixture_path(group)
                compare_fixtures(original, Path(directory) / original.name, group)


def capture(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def observations_from_log(log):
    observations = []
    target = test = None
    occurrences = {}
    for line in log.splitlines():
        if re.search(r"Running unittests src[/\\]lib\.rs", line):
            target, test = "lib", None
        if match := TARGET.search(line):
            target, test = match.group(1), None
        if match := TEST.search(line):
            test = match.group(1)
        if match := LINE.search(line):
            reference_name, diagnostic_name, details = match.groups()
            name = reference_name or diagnostic_name
            key = f"{target}/{test}/{name}"
            occurrences[key] = occurrences.get(key, 0) + 1
            metrics = {key: float(value) for key, value in NUMBER.findall(details)}
            # Keep failed reports JSON-serializable while explicitly rejecting
            # NaN/Inf metrics below, rather than silently dropping their keys.
            metrics = {key: value if math.isfinite(value) else None for key, value in metrics.items()}
            observations.append({"id": f"{key}/{occurrences[key]}", "target": target,
                                 "test": test, "name": name, "metrics": metrics,
                                 "details": details,
                                 "comparison_only": name.startswith(("recursive_librosa_", "diagnostic_"))
                                 or name == "narrow_inverse"})
    return observations


def target_for(group):
    return {"corpus": "real_audio_workflows", "pitch": "pitch_correctness"}.get(group, "reference_" + group)


def validate_runs(log, groups, observations):
    """Require each requested target, a positive successful run, and its metrics."""
    expected = {target_for(group) for group in groups}
    if "iir" in groups:
        expected.add("lib")
    target = None
    summaries = {}
    errors = []
    for line in log.splitlines():
        if re.search(r"Running unittests src[/\\]lib\.rs", line):
            target = "lib"
        if match := TARGET.search(line):
            target = match.group(1)
        if match := re.search(r"test result: (\w+)\. (\d+) passed; (\d+) failed;", line):
            status, passed, failed = match.groups()
            summaries[target] = {"status": status, "passed": int(passed), "failed": int(failed)}
    for target in sorted(expected):
        summary = summaries.get(target)
        if not summary or summary["status"] != "ok" or summary["passed"] == 0 or summary["failed"]:
            errors.append(f"Missing, empty, or unsuccessful test run: {target}")
        if not any(o["target"] == target and o["metrics"] for o in observations):
            errors.append(f"Missing numerical metrics: {target}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    for group in groups:
        for target, count in manifest["groups"][group]["required_test_counts"].items():
            if summaries.get(target, {}).get("passed") != count:
                errors.append(f"Missing or changed test count: {target}: expected {count}")
        for test, count in manifest["groups"][group].get("required_metric_tests", {}).items():
            target, name = test.split("/", 1)
            found = sum(o["target"] == target and o["test"] == name and bool(o["metrics"]) for o in observations)
            if found != count:
                errors.append(f"Missing or changed numerical observations: {test}: expected {count}, found {found}")
    for observation in observations:
        if observation["target"] not in expected or not observation["test"] or not observation["metrics"]:
            errors.append(f"Unattributed/empty observation: {observation['id']}")
        if not all(v is not None and math.isfinite(v) for v in observation["metrics"].values()):
            errors.append(f"Nonfinite observation: {observation['id']}")
    return summaries, errors


def benchmark_paths():
    return [Path(__file__).resolve(), ROOT / "Cargo.toml", ROOT / "Cargo.lock",
            MANIFEST_PATH, FIXTURES / "requirements-reference.txt",
            ROOT / "scripts/prepare_audio_corpus.py", ROOT / "tests/real_audio_workflows.rs",
            ROOT / "tests/pitch_correctness.rs", ROOT / "tests/common/reference.rs",
            *ROOT.glob("tests/reference_*.rs"), *FIXTURES.glob("generate_*_reference.py")]


def hashes(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file()}


def snapshot(groups):
    paths = [*(ROOT / "src").rglob("*.rs"), *benchmark_paths(),
             *(fixture_path(g) for g in groups)]
    if "corpus" in groups:
        paths.extend((FIXTURES / "audio").iterdir())
    return hashes(paths)


def snapshot_changes(before, after):
    # Cargo may create its lock file on a clean checkout. Its final hash is
    # recorded; an existing lock file changing during validation is a failure.
    return [p for p in sorted(before.keys() | after.keys())
            if before.get(p) != after.get(p) and not (p == "Cargo.lock" and p not in before)]


def self_test():
    """Stdlib-only regression checks for the reporting gate, not DSP accuracy."""
    checks = 0

    def require(condition, message):
        nonlocal checks
        if not condition:
            raise ValueError(f"Runner self-test failed: {message}")
        checks += 1

    def rejected(call):
        try:
            with redirect_stdout(io.StringIO()):
                call()
        except ValueError:
            return True
        return False

    manifest = json.loads(MANIFEST_PATH.read_text())
    for group in GROUPS:
        input_inventory(group, fixture_values(fixture_path(group)))
    require(True, "all fixture matrices must be explicitly classified")
    with tempfile.TemporaryDirectory(prefix="dasp-reference-selftest-") as directory:
        # Each mutation was previously below the generic absolute regeneration
        # tolerance, or belonged to a supplied-input/config name it overlooked.
        for group, name in [("rhythm", "env_quiet"), ("spectral", "quiet_even_magnitude"),
                            ("transforms", "spectrum_even"), ("cqt_sweep", "tone_16k_config")]:
            original = fixture_path(group)
            lines = original.read_text().splitlines()
            for index, line in enumerate(lines):
                if line.startswith(name + ","):
                    columns = line.split(",")
                    value = float(columns[3])
                    if name.endswith("config"):
                        columns[3] = repr(value + 1)
                    else:
                        bits = struct.unpack("!I", f32_bits(value))[0]
                        columns[3] = repr(struct.unpack("!f", struct.pack("!I", bits + 1))[0])
                    lines[index] = ",".join(columns)
                    break
            generated = Path(directory) / original.name
            generated.write_text("\n".join(lines) + "\n")
            require(rejected(lambda: compare_fixtures(original, generated, group)), f"reject {name} drift")
        original = fixture_path("pitch")
        lines = original.read_text().splitlines()
        for index, line in enumerate(lines):
            if line.startswith("tone_100,"):
                columns = line.split(",")
                columns[4] = "0" if columns[4] == "1" else "1"
                lines[index] = ",".join(columns)
                break
        generated = Path(directory) / original.name
        generated.write_text("\n".join(lines) + "\n")
        require(rejected(lambda: compare_fixtures(original, generated, "pitch")), "reject legacy pitch voicing drift")

    logs = []
    for group, entry in manifest["groups"].items():
        for target, count in entry["required_test_counts"].items():
            logs.append("Running unittests src/lib.rs" if target == "lib" else f"Running tests/{target}.rs")
            for test, observations in entry["required_metric_tests"].items():
                owner, name = test.split("/", 1)
                if owner == target:
                    logs.append(f"test {name} ...")
                    logs.extend("REFERENCE selftest: count=1 max_abs=0" for _ in range(observations))
            logs.append(f"test result: ok. {count} passed; 0 failed;")
    log = "\n".join(logs)
    observed = observations_from_log(log)
    require(not validate_runs(log, GROUPS, observed)[1], "complete run accepted")
    require(bool(validate_runs(log, GROUPS, observed[:-1])[1]), "missing metric rejected")
    nonfinite = observations_from_log(log.replace("max_abs=0", "max_abs=NaN", 1))
    require(bool(validate_runs(log, GROUPS, nonfinite)[1]), "nonfinite metric rejected")
    require(bool(validate_runs("", GROUPS, [])[1]), "missing suites rejected")
    empty = re.sub(r"test result: ok\. \d+ passed", "test result: ok. 0 passed", log)
    require(bool(validate_runs(empty, GROUPS, observed)[1]), "zero-test success rejected")
    require(bool(validate_runs(log.replace("test result: ok.", "test result: FAILED."), GROUPS, observed)[1]), "failed run rejected")
    diagnostics = observations_from_log("Running tests/reference_rhythm.rs\ntest example ... diagnostic_same: error=1\ndiagnostic_same: error=2")
    require(len(diagnostics) == 2 and all(o["comparison_only"] for o in diagnostics)
            and len({o["id"] for o in diagnostics}) == 2, "bare diagnostics and unique IDs")
    require(snapshot_changes({}, {"Cargo.lock": "new"}) == [], "fresh lock file allowed")
    require(snapshot_changes({"Cargo.lock": "old"}, {"Cargo.lock": "new"}) == ["Cargo.lock"], "existing lock drift rejected")
    require(snapshot_changes({"src/lib.rs": "old"}, {"src/lib.rs": "new"}) == ["src/lib.rs"], "source drift rejected")
    require(f32_bits(1.0) == f32_bits(math.nextafter(1.0, 2.0)), "equivalent f64 encodings preserve f32 input")
    print(f"PASS: {checks} runner self-checks (stdlib only)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run stdlib-only negative checks for fixture drift, missing metrics/runs, and source consistency; no Cargo")
    regeneration = parser.add_mutually_exclusive_group()
    regeneration.add_argument("--check-fixtures", action="store_true", help="Regenerate in a temporary directory and reject numerical fixture drift")
    regeneration.add_argument("--regenerate", action="store_true", help="Replace fixtures using the pinned independent reference libraries")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter with the optional pinned reference dependencies")
    parser.add_argument("--release", action="store_true", help="Benchmark the Rust release build")
    parser.add_argument("--group", action="append", choices=GROUPS,
                        help="Run a selected suite; repeat to select multiple suites (default: all)")
    parser.add_argument("--output", type=Path, default=ROOT / "target/reference-report.json")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    groups = tuple(dict.fromkeys(args.group or GROUPS))
    if args.check_fixtures or args.regenerate:
        log = io.StringIO()
        try:
            with redirect_stdout(log):
                generate(args.python, args.check_fixtures, groups)
        except (ValueError, subprocess.CalledProcessError, OSError) as error:
            # Publish diagnostics even when fixture verification prevents the
            # Rust stage from running, so CI's artifact always explains failure.
            report = {"schema_version": 1, "passed": False,
                      "stage": "fixture_generation", "error": str(error),
                      "observations": [], "fixture_regeneration_checked": False}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            args.output.with_suffix(".log").write_text(log.getvalue() + str(error) + "\n")
            print(log.getvalue(), end="")
            print(f"FAIL: {error}; report: {args.output}", file=sys.stderr)
            return 1
        print(log.getvalue(), end="", flush=True)

    try:
        for group in groups:
            input_inventory(group, fixture_values(fixture_path(group)))
        before = snapshot(groups)
    except (ValueError, OSError, KeyError) as error:
        report = {"schema_version": 2, "passed": False, "stage": "input_inventory",
                  "error": str(error), "observations": []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        args.output.with_suffix(".log").write_text(str(error) + "\n")
        print(f"FAIL: {error}; report: {args.output}", file=sys.stderr)
        return 1
    command = ["cargo", "test"]
    if args.release:
        command.append("--release")
    for group in groups:
        command.extend(["--test", target_for(group)])
    command.extend(["--", "--nocapture", "--test-threads=1"])
    env = dict(os.environ)
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    commands = [command]
    if "iir" in groups:
        # Private filter design is checked inside the library; the integration
        # target separately checks the public filterbank and its frame energy.
        design_command = ["cargo", "test", *(["--release"] if args.release else []),
                          "--lib", "iir_design", "--", "--nocapture", "--test-threads=1"]
        design = subprocess.run(design_command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
        commands.append(design_command)
        result.stdout += "\n" + design.stdout
        result.returncode = result.returncode or design.returncode
    observations = observations_from_log(result.stdout)
    after = snapshot(groups)
    changes = snapshot_changes(before, after)
    summaries, validation_errors = validate_runs(result.stdout, groups, observations)
    if changes:
        validation_errors.append("Sources or fixtures changed during validation: " + ", ".join(changes))
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "commands": commands,
        "groups": groups,
        "cargo_target_dir": env["CARGO_TARGET_DIR"],
        "rustc": capture(["rustc", "--version"]),
        "git_revision": capture(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(capture(["git", "status", "--porcelain"])),
        "source_sha256": hashes((ROOT / "src").rglob("*.rs")),
        "benchmark_sha256": hashes(benchmark_paths()),
        "snapshot_before_sha256": before,
        "snapshot_after_sha256": after,
        "snapshot_consistent": not changes,
        "validation_errors": validation_errors,
        "selected_test_targets": sorted(summaries),
        "suite_results": summaries,
        "input_inventory": {g: json.loads(MANIFEST_PATH.read_text())["groups"][g] for g in groups},
        "fixture_sha256": {group: hashlib.sha256(fixture_path(group).read_bytes()).hexdigest() for group in groups},
        "corpus_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((FIXTURES / "audio").iterdir()) if p.is_file()}
                         if "corpus" in groups else {},
        "fixture_regeneration_checked": args.check_fixtures,
        "reference_versions": (FIXTURES / "requirements-reference.txt").read_text().splitlines(),
        "passed": result.returncode == 0 and not validation_errors,
        "observations": observations,
        "test_summaries": re.findall(r"test result: [^\n]+", result.stdout),
        "notes": ["Numerical accuracy comparisons, not throughput measurements.",
                  "Comparison-only observations describe different algorithms or ill-conditioned inverses; they are not parity claims.",
                  "Source snapshots bracket Cargo execution; intentional fixture regeneration happens first.",
                  "Pitch compares independently synthesized analytic signals, not identical stored samples; see input manifest policy.",
                  "Absolute errors carry each output's units; compare tolerance ratios within the associated test, not errors across unrelated quantities."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".log").write_text(result.stdout)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(observations)} numerical observations; report: {args.output}")
    for group in groups:
        print(f"  fixture {group}: {report['fixture_sha256'][group]}")
    if not report["passed"]:
        print("\n".join(validation_errors), file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        return result.returncode or 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
