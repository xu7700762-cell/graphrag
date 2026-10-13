"""Build reviewed, same-fold graphs from the pinned production reference assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from eeg_agent import graph_rag


def build(output):
    manifest = graph_rag._read_verified(graph_rag.REFERENCE_MANIFEST,
                                        graph_rag.REFERENCE_MANIFEST_SHA256)
    source = (PROJECT / 'eeg_agent/brain_knowledge/vrms/03_nine_evidence_guide.md').read_text(encoding='utf-8')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}

    def save(name, value):
        data = (value if isinstance(value, str) else
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)).encode('utf-8')
        (output / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()

    save('method_source.md', source)
    save('method_graph.json', graph_rag.build_method_graph(source))
    for filename, expected in sorted(manifest['files'].items()):
        entry = graph_rag._read_verified(graph_rag.REFERENCE_MANIFEST.parent / filename, expected)
        save(filename, graph_rag.build_fold_graph(entry['bank'], entry['provenance']))
    result = {'schema': graph_rag.GRAPH_SCHEMA, 'algorithm_version': graph_rag.ALGORITHM_VERSION,
              'reference_manifest_sha256': graph_rag.REFERENCE_MANIFEST_SHA256,
              'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'files': files}
    save('manifest.json', result)
    return files['manifest.json']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=graph_rag.GRAPH_MANIFEST.parent)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty; keep existing immutable graph versions or back them up first.')
    print('graph_manifest_sha256=' + build(args.output))
