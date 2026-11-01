#!/usr/bin/env python3
"""Reproducible warmed DSP latency and separate allocation measurements.

Examples:
  python scripts/performance_benchmarks.py build
  python scripts/performance_benchmarks.py run --output target/performance-baseline.json
  python scripts/performance_benchmarks.py run --cases stft_short_512 --repetitions 3
  python scripts/performance_benchmarks.py compare BASELINE.json CANDIDATE.json

Run baseline/candidate serially on the same idle machine and toolchain. Comparisons
report changes; they never enforce noisy wall-clock regression limits. Input and
inverse-spectrum preparation are excluded from latency/allocation counters. Each
call includes library setup, processing, result allocation and result destruction.
The optional RSS observation includes the whole isolated process and its inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ('dsp_timing', 'dsp_allocations')


def hashes():
    paths = [*sorted((ROOT / 'src').rglob('*.rs')), *sorted((ROOT / 'benches').rglob('*.rs')),
             ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def capture(args, env):
    return subprocess.check_output(args, cwd=ROOT, env=env, text=True).strip()


def environment(env):
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        match = re.search(r'^model name\s*:\s*(.+)$', cpuinfo.read_text(), re.M)
        if match:
            cpu = match[1]
    governor = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
    try:
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    except OSError:
        affinity = None
    cargo_home = Path(env.get('CARGO_HOME', str(Path.home()/'.cargo')))
    configs = [ROOT/'.cargo/config', ROOT/'.cargo/config.toml', cargo_home/'config', cargo_home/'config.toml']
    config_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in configs if p.exists()}
    return {'cpu_affinity': affinity, 'cargo_config_sha256': config_hashes, 'platform': platform.platform(), 'machine': platform.machine(), 'host': platform.node(),
            'cpu': cpu, 'logical_cpus': os.cpu_count(), 'python': platform.python_version(),
            'rustc': capture(['rustc', '-vV'], env), 'cargo': capture(['cargo', '--version'], env),
            'governor': governor.read_text().strip() if governor.exists() else None,
            'profile': 'release', 'timing_allocator': 'unmodified Rust default',
            'allocation_allocator': 'stats_alloc 0.1.10 wrapping System',
            'variables': {k: env.get(k) for k in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
                           'CARGO_BUILD_TARGET', 'CARGO_PROFILE_RELEASE_OPT_LEVEL',
                           'CARGO_PROFILE_RELEASE_LTO', 'RAYON_NUM_THREADS')}}


def build(env):
    # This library does not track Cargo.lock. Preserve an existing lock, but make
    # fresh-clone benchmark builds work before using --locked for reproducibility.
    if not (ROOT/'Cargo.lock').exists():
        subprocess.run(['cargo', 'generate-lockfile'], cwd=ROOT, env=env, check=True)
    command = ['cargo', 'build', '--release', '--locked', '--message-format=json']
    for name in BENCHES:
        command.extend(['--bench', name])
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=sys.stderr, check=True)
    executables = {}
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get('reason') == 'compiler-artifact' and message.get('executable'):
            name = message['target']['name']
            if name in BENCHES:
                executables[name] = message['executable']
    if set(executables) != set(BENCHES):
        raise ValueError(f'Missing bench executables in Cargo build output: {executables}')
    return executables


def invoke(binary, case, repetitions, warmup, env, timeout):
    command = [binary, '--run', '--case', case, '--repetitions', str(repetitions),
               '--warmup', str(warmup)]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]


def rss_observation(binary, case, env, timeout):
    # GNU time is optional. macOS time has different flags/units; do not mix them.
    if platform.system() != 'Linux' or not Path('/usr/bin/time').exists():
        return {'available': False, 'reason': 'Requires Linux GNU /usr/bin/time -v'}
    with tempfile.TemporaryDirectory(prefix='dasp-rss-') as folder:
        output = Path(folder) / 'time.txt'
        subprocess.run(['/usr/bin/time', '-v', '-o', str(output), binary, '--run', '--case', case,
                        '--repetitions', '1', '--warmup', '1'], cwd=ROOT, env=dict(env, LC_ALL='C'), timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        text = output.read_text()
        match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', text)
        return {'available': bool(match), 'peak_rss_bytes': int(match[1])*1024 if match else None,
                'scope': 'separate uninstrumented process, includes preparation and one warmup'}


def run(args, env):
    binaries = build(env)
    configurations = [json.loads(line) for line in capture([binaries['dsp_timing'], '--list'], env).splitlines()]
    available = {case['name']: case for case in configurations}
    selected = args.cases.split(',') if args.cases else list(available)
    if not selected or set(selected) - available.keys():
        raise ValueError(f'Unknown cases: {set(selected) - available.keys()}; choices: {list(available)}')
    before = hashes()
    report = {'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'environment': environment(env), 'source_sha256': before,
              'git_revision': capture(['git', 'rev-parse', 'HEAD'], env),
              'working_tree_dirty': bool(capture(['git', 'status', '--porcelain'], env)),
              'binary_sha256': {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in binaries.items()},
              'settings': {'repetitions': args.repetitions, 'warmup': args.warmup,
                           'allocation_repetitions': args.allocation_repetitions,
                           'allocation_warmup': 1, 'case_timeout_seconds': args.timeout},
              'observations': [], 'errors': [],
              'notes': ['Timing binary has no instrumented allocator; allocation binary is measured separately.',
                        'Input/spectrum construction excluded; per-call setup and output destruction included.',
                        'Allocation bytes are cumulative requested heap traffic, not peak live memory.',
                        'p95 uses nearest-rank quantile; small sample counts have substantial uncertainty.',
                        'No timing threshold is a CI pass/fail criterion; use an idle same-environment baseline.']}
    for name in selected:
        print(f'Benchmark {name}: warmup={args.warmup}, repetitions={args.repetitions}', flush=True)
        try:
            timing = invoke(binaries['dsp_timing'], name, args.repetitions, args.warmup, env, args.timeout)
            allocation = invoke(binaries['dsp_allocations'], name, args.allocation_repetitions, 1, env, args.timeout)
            if len(timing) != 1 or len(allocation) != args.allocation_repetitions:
                raise ValueError(f'Unexpected observation counts for {name}')
            samples = timing[0]['seconds']
            if len(samples) != args.repetitions or not all(math.isfinite(x) and x>0 for x in samples):
                raise ValueError(f'Invalid timing samples for {name}: {samples}')
            median = statistics.median(samples)
            p95 = sorted(samples)[max(0, math.ceil(0.95*len(samples))-1)]
            duration = available[name]['audio_seconds']
            fields = ('allocations', 'deallocations', 'reallocations', 'bytes_allocated',
                      'bytes_deallocated', 'bytes_reallocated')
            entry = {'case': available[name], 'timing': {'samples_seconds': samples,
                     'median_seconds': median, 'p95_seconds': p95,
                     'audio_seconds_per_wall_second': duration/median,
                     'input_samples_per_second': available[name]['samples']/median,
                     'realtime_factor': median/duration},
                     'allocation': {'samples': [{key: row[key] for key in fields} for row in allocation],
                                    'median_per_call': {key: statistics.median(row[key] for row in allocation) for key in fields}}}
            if args.rss:
                entry['rss'] = rss_observation(binaries['dsp_timing'], name, env, args.timeout)
            report['observations'].append(entry)
        except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            report['errors'].append({'case': name, 'error': str(error), 'stderr': str(getattr(error,'stderr',''))})
            print(f'Failed {name}: {error}', file=sys.stderr)
    report['source_changed_during_run'] = hashes() != before
    report['complete'] = not report['errors'] and not report['source_changed_during_run']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Performance report: {args.output} (complete={report["complete"]})')
    return 0 if report['complete'] else 1


def compare(args):
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if baseline['environment'] != candidate['environment']:
        differing = [key for key in baseline['environment'] if baseline['environment'].get(key) != candidate['environment'].get(key)]
        raise ValueError(f'Comparison requires the same environment; differing fields: {differing}')
    if not baseline['complete'] or not candidate['complete']:
        raise ValueError('Comparison requires complete reports with unchanged sources during measurement')
    workload = 'benches/common/mod.rs'
    if baseline['source_sha256'].get(workload) != candidate['source_sha256'].get(workload):
        raise ValueError('Workload implementation changed despite possibly identical case metadata')
    old = {entry['case']['name']: entry for entry in baseline['observations']}
    new = {entry['case']['name']: entry for entry in candidate['observations']}
    if old.keys() != new.keys():
        raise ValueError('Reports must contain the same workload cases')
    changes = []
    for name, current in new.items():
        previous = old[name]
        if previous['case'] != current['case']:
            raise ValueError(f'Workload configuration changed: {name}')
        changes.append({'name': name,
            'median_latency_ratio': current['timing']['median_seconds']/previous['timing']['median_seconds'],
            'p95_latency_ratio': current['timing']['p95_seconds']/previous['timing']['p95_seconds'],
            'allocation_count_delta': current['allocation']['median_per_call']['allocations']-previous['allocation']['median_per_call']['allocations'],
            'allocated_bytes_delta': current['allocation']['median_per_call']['bytes_allocated']-previous['allocation']['median_per_call']['bytes_allocated']})
    result = {'baseline': str(args.baseline), 'candidate': str(args.candidate), 'same_environment': True,
              'notes': ['Ratios above 1 are slower; changes are descriptive and have no pass/fail threshold.'],
              'changes': changes}
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build', help='Compile both release binaries without running workloads')
    bench = sub.add_parser('run', help='Measure warmed latency and separate allocation counters')
    bench.add_argument('--cases', help='Comma-separated workload names; default: all 13 cases')
    bench.add_argument('--repetitions', type=int, default=15)
    bench.add_argument('--warmup', type=int, default=3)
    bench.add_argument('--allocation-repetitions', type=int, default=3)
    bench.add_argument('--timeout', type=float, default=600)
    bench.add_argument('--rss', action='store_true', help='Also collect optional isolated-process GNU time RSS')
    bench.add_argument('--output', type=Path, default=ROOT/'target/performance-report.json')
    diff = sub.add_parser('compare', help='Compare complete reports from the same environment')
    diff.add_argument('baseline', type=Path)
    diff.add_argument('candidate', type=Path)
    diff.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'compare':
        return compare(args)
    env = dict(os.environ)
    env.setdefault('CARGO_TARGET_DIR', str(ROOT/'target'))
    env.setdefault('RAYON_NUM_THREADS', '1')
    if args.command == 'build':
        print(json.dumps(build(env), indent=2))
        return 0
    if min(args.repetitions, args.warmup, args.allocation_repetitions) < 1 or args.timeout <= 0:
        raise ValueError('Repetitions, warmup, allocation repetitions, and timeout must be positive')
    return run(args, env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
