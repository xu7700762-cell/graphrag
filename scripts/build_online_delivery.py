"""Build a private, allowlisted online Pi bundle and verify every archive hash.

ARM64 wheels are downloaded to an isolated cache, never installed on this host.
This checks package availability and payload integrity, not Pi hardware speed.
"""
import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = PROJECT / 'outputs/online_implementation/20261011'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def wheel_preflight(wheelhouse, download=False):
    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.utils import canonicalize_name
    except ImportError:
        from pip._vendor.packaging.markers import default_environment
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.specifiers import SpecifierSet
        from pip._vendor.packaging.utils import canonicalize_name
    wheelhouse.mkdir(parents=True, exist_ok=True)
    if download:
        command = [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                   '--implementation', 'cp', '--python-version', '311',
                   '--abi', 'cp311', '--abi', 'abi3', '--abi', 'none',
                   '--dest', str(wheelhouse), '-r', str(PROJECT / 'requirements-online-pi.txt')]
        for platform in ('manylinux_2_28_aarch64', 'manylinux_2_27_aarch64',
                         'manylinux_2_17_aarch64', 'manylinux2014_aarch64'):
            command += ['--platform', platform]
        subprocess.run(command, check=True)
    packages = {}
    for path in sorted(wheelhouse.glob('*.whl')):
        filename = path.name
        if not (filename.endswith('-any.whl') or ('aarch64' in filename and 'manylinux' in filename)):
            raise ValueError('Non-ARM64/pure-Python wheel in isolated cache: ' + filename)
        if any(forbidden in filename.lower() for forbidden in ('nvidia_', 'cuda', 'triton', 'mamba_ssm')):
            raise ValueError('GPU or CUDA-extension wheel is forbidden: ' + filename)
        with zipfile.ZipFile(path) as archive:
            name = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
            meta = BytesParser().parsebytes(archive.read(name))
            native_libraries = [name for name in archive.namelist()
                                if '.so' in Path(name).name and not name.endswith('/')]
            for name in native_libraries:
                with archive.open(name) as handle:
                    header = handle.read(64)
                if header[:4] != b'\x7fELF':
                    raise ValueError('Native wheel library is not ELF: ' + name)
                endian = 'little' if header[5] == 1 else 'big'
                if int.from_bytes(header[18:20], endian) != 183:
                    raise ValueError('Native wheel library is not AArch64: ' + name)
            if meta['Name'].lower() == 'torch':
                if any('cuda' in name.lower() for name in native_libraries):
                    raise ValueError('Torch target wheel includes CUDA libraries')
        key = canonicalize_name(meta['Name'])
        if key in packages:
            raise ValueError('Duplicate wheel versions in target cache: ' + key)
        packages[key] = {'name': meta['Name'], 'version': meta['Version'], 'file': filename,
                         'sha256': sha(path), 'bytes': path.stat().st_size,
                         'requires_python': meta.get('Requires-Python'),
                         'native_libraries': len(native_libraries), 'elf_architecture': 'AArch64' if native_libraries else None,
                         'requirements': meta.get_all('Requires-Dist', [])}
    environment = default_environment()
    environment.update(os_name='posix', platform_machine='aarch64', platform_system='Linux',
                       sys_platform='linux', python_version='3.11', python_full_version='3.11.14',
                       implementation_name='cpython', platform_python_implementation='CPython', extra='')
    pending = [Requirement(line) for line in (PROJECT / 'requirements-online-pi.txt').read_text(encoding='utf-8').splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
    selected = {}
    while pending:
        requirement = pending.pop()
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        key = canonicalize_name(requirement.name)
        item = packages.get(key)
        if item is None or item['version'] not in requirement.specifier:
            raise ValueError('ARM64 dependency missing or incompatible: ' + str(requirement))
        if item['requires_python'] and '3.11.14' not in SpecifierSet(item['requires_python']):
            raise ValueError('Wheel requires an incompatible Python version: ' + item['file'])
        if key not in selected:
            selected[key] = item
            pending.extend(Requirement(value) for value in item['requirements'])
    lock = DEFAULT_EVIDENCE / 'requirements-online-pi.lock.txt'
    lock.write_text('# ARM64 CPython 3.11 dependency closure; generated from downloaded wheel metadata.\n' +
                   '\n'.join(f"{key}=={item['version']} --hash=sha256:{item['sha256']}"
                             for key, item in sorted(selected.items())) + '\n', encoding='utf-8')
    report = {'status': 'passed', 'kind': 'actual wheel download and target dependency metadata check',
              'hardware_tested': False, 'target': 'Linux aarch64 CPython 3.11 glibc>=2.28',
              'requirements_sha256': sha(PROJECT / 'requirements-online-pi.txt'),
              'wheelhouse': str(wheelhouse), 'packages': list(selected.values()),
              'selected_wheels': len(selected), 'cache_wheels': len(packages),
              'total_selected_bytes': sum(item['bytes'] for item in selected.values()),
              'unused_host_marker_wheels': sorted(set(packages) - set(selected)),
              'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(DEFAULT_EVIDENCE / 'arm64_wheel_preflight.json', report)
    return report, lock


def bundle_files(asset_manifest, wheels, lock, fix_evidence=None):
    files = {}
    def add(path, destination=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        name = destination or path.relative_to(PROJECT).as_posix()
        if name.startswith('/') or '..' in Path(name).parts:
            raise ValueError('Unsafe archive path')
        if name in files and files[name] != path:
            raise ValueError('Duplicate archive path: ' + name)
        files[name] = path
    for name in ('pyproject.toml', 'requirements-online-pi.txt', 'README.md',
                 'deploy/pi_online.yaml', 'deploy/eeg-agent-online.service',
                 'deploy/eeg-agent-online-journald.conf', 'deploy/eeg-agent-online.logrotate',
                 'docs/ONLINE_DEPLOYMENT.md', 'docs/ONLINE_MODEL.md',
                 'docs/ONLINE_ANALYSIS.md', 'docs/RAG_DEFAULT.md', 'docs/GRAPH_RAG.md',
                 'scripts/build_online_delivery.py', 'scripts/benchmark_online_pi.py',
                 'scripts/validate_online_stream.py', 'scripts/prepare_online_assets.py',
                 'scripts/validate_online_model.py', 'scripts/freeze_online_baseline.py'):
        add(PROJECT / name)
    optional_script = PROJECT / 'scripts/evaluate_online_numeric.py'
    if optional_script.is_file():
        add(optional_script)
    for pattern in ('eeg_agent/*.py', 'eeg_agent/online/*.py', 'eeg_agent/online/licenses/*',
                    'eeg_agent/web/*.html', 'eeg_agent/web/*.css', 'eeg_agent/web/*.js',
                    'eeg_agent/brain_knowledge/*/*.md', 'knowledge/**/*.md'):
        for path in sorted(PROJECT.glob(pattern)):
            # The unassessed v2 candidate is reviewed separately from this release.
            if path == PROJECT / 'eeg_agent/graph_rag_v2.py':
                continue
            add(path)
    if (PROJECT / 'scripts/evaluate_online_numeric.py').is_file():
        add(PROJECT / 'scripts/evaluate_online_numeric.py')
    add(PROJECT / 'deploy/pi_online.yaml', 'configs/local.yaml')
    from eeg_agent.default_rag import REFERENCE_MANIFEST, REFERENCE_MANIFEST_SHA256
    if sha(REFERENCE_MANIFEST) != REFERENCE_MANIFEST_SHA256:
        raise ValueError('Default RAG reference manifest differs')
    rag_assets = json.loads(REFERENCE_MANIFEST.read_text(encoding='utf-8'))
    add(REFERENCE_MANIFEST)
    for filename, expected in rag_assets['files'].items():
        if Path(filename).name != filename or not filename.startswith('fold_') or not filename.endswith('.json'):
            raise ValueError('Unsafe RAG asset path')
        path = REFERENCE_MANIFEST.parent / filename
        if sha(path) != expected:
            raise ValueError('Default RAG reference fold differs')
        add(path)
    from eeg_agent.default_rag import SNAPSHOT, SNAPSHOT_SHA256
    from eeg_agent.graph_rag import GRAPH_MANIFEST, GRAPH_MANIFEST_SHA256
    for manifest_path, pinned_sha in ((SNAPSHOT, SNAPSHOT_SHA256),
                                      (GRAPH_MANIFEST, GRAPH_MANIFEST_SHA256)):
        if sha(manifest_path) != pinned_sha:
            raise ValueError('Pinned dual RAG manifest differs')
        add(manifest_path)
        manifest_value = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, expected in manifest_value['files'].items():
            if Path(name).name != name:
                raise ValueError('Unsafe dual RAG asset path')
            path = manifest_path.parent / name
            if sha(path) != expected:
                raise ValueError('Dual RAG asset hash mismatch')
            add(path)
    comparison = PROJECT / 'outputs/rag_assets/comparison_v1/comparison.json'
    if comparison.is_file():
        add(comparison)
    value = json.loads(asset_manifest.read_text(encoding='utf-8'))
    if value.get('schema') != 'online_cpu_seed2026_v1' or len(value.get('folds', {})) != 24:
        raise ValueError('Expected portable frozen twenty-four-fold online manifest')
    asset_prefix = 'outputs/online_assets/seed2026/'
    add(asset_manifest, asset_prefix + 'manifest.json')
    records = [value['encoder']] + [item for fold in value['folds'].values() for item in fold.values()
                                  if isinstance(item, dict) and 'file' in item and 'sha256' in item]
    for item in records:
        relative = Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe portable asset path')
        path = asset_manifest.parent / relative
        if sha(path) != item['sha256']:
            raise ValueError('Frozen asset hash mismatch: ' + item['file'])
        add(path, asset_prefix + relative.as_posix())
    add(lock, 'requirements-online-pi.lock.txt')
    wheelhouse = Path(wheels['wheelhouse'])
    for item in wheels['packages']:
        add(wheelhouse / item['file'], 'wheels/arm64_cp311/' + item['file'])
    for name in ('stream_validation.json', 'arm64_wheel_preflight.json',
                 'model_validation/report.json', 'model_validation/all_paths_report.json',
                 'numeric_evaluation.json', 'test_report.json', 'freeze_verify.json',
                 'local_1x_130s_final.json', 'pi_hardware_status_final.json',
                 '实施与验收.md'):
        path = DEFAULT_EVIDENCE / name
        if path.is_file():
            add(path, 'validation/' + name)
    # Include the synthetic shared judge fixtures imported by online cloud tests.
    # No private recording, credentials or historical result fixtures are copied.
    add(PROJECT / 'tests/__init__.py')
    add(PROJECT / 'tests/test_raw_independent_judge.py')
    for path in sorted((PROJECT / 'tests').glob('test_online*.py')):
        add(path)
    if fix_evidence is not None:
        for name in ('修复与验证.md', 'test_report.json', 'freeze_verify.json', 'sse_poll_benchmark.json',
                     'local_replay.json', 'prefix_regression.json', 'browser_ui.json'):
            path = Path(fix_evidence) / name
            if path.is_file():
                add(path, 'validation/fixes/' + name)
    return files, value


def build(args):
    wheels, lock = wheel_preflight(args.wheelhouse.resolve(), args.check_wheels)
    if args.wheels_only:
        print(json.dumps({key: wheels[key] for key in ('status', 'selected_wheels', 'total_selected_bytes')}, indent=2))
        return
    files, assets = bundle_files(args.assets.resolve(), wheels, lock, args.fix_evidence)
    entries = [{'file': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for name, path in sorted(files.items())]
    manifest = {'schema': 'private_online_pi_bundle_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'target': 'Raspberry Pi 5 8GB, Raspberry Pi OS 64-bit, CPython 3.11',
                'backend': 'mamba_minimal_cpu_fp32', 'seed': 2026, 'hardware_tested': False,
                'folds': sorted(map(int, assets['folds'])), 'missing_folds': [22, 26],
                'private_assets_included': True, 'raw_data_included': False,
                'api_credentials_included': False, 'api_responses_included': False,
                'entries': entries, 'payload_bytes': sum(item['bytes'] for item in entries),
                'arm64_wheel_preflight': {'status': wheels['status'], 'selected_wheels': wheels['selected_wheels']}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.building.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if name.endswith('.whl') else zipfile.ZIP_DEFLATED)
        archive.writestr('deploy/online_bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        actual = set(archive.namelist())
        if actual != set(files) | {'deploy/online_bundle_manifest.json'}:
            raise ValueError('Archive allowlist differs')
        for item in entries:
            result = hashlib.sha256()
            with archive.open(item['file']) as handle:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    result.update(block)
            if result.hexdigest() != item['sha256']:
                raise ValueError('Archive content hash mismatch: ' + item['file'])
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
    temporary.replace(args.out)
    manifest.update(archive_sha256=sha(args.out), archive_bytes=args.out.stat().st_size,
                    verified_entries=len(entries), archive_verified=True)
    write_json(args.manifest, manifest)
    print(json.dumps({key: manifest[key] for key in
                      ('archive_verified', 'verified_entries', 'archive_bytes', 'archive_sha256', 'hardware_tested')}, indent=2))
    print(str(args.out.resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=PROJECT / 'outputs/online_assets/seed2026/manifest.json')
    parser.add_argument('--wheelhouse', type=Path, default=DEFAULT_EVIDENCE / 'arm64_cp311_wheels')
    parser.add_argument('--check-wheels', action='store_true', help='Download binary ARM64 dependencies before validating')
    parser.add_argument('--fix-evidence', type=Path, help='Optional current fix verification directory')
    parser.add_argument('--wheels-only', action='store_true')
    parser.add_argument('--out', type=Path, default=PROJECT / 'deploy/EEGAgent_Online_Pi_bundle.zip')
    parser.add_argument('--manifest', type=Path, default=PROJECT / 'deploy/online_bundle_manifest.json')
    build(parser.parse_args())
