"""Build reviewed semantic assets without modifying frozen v1/v2/v3 files."""
import argparse
import hashlib
import json
from pathlib import Path

from eeg_agent import state_graph_rag as semantic


def build(root=semantic.KNOWLEDGE, out=semantic.MANIFEST.parent):
    root, out = Path(root), Path(out)
    graph = semantic.build_graph(json.loads((root / 'sources.json').read_text(encoding='utf-8')),
                                 json.loads((root / 'claims.json').read_text(encoding='utf-8')))
    out.mkdir(parents=True, exist_ok=True)
    graph_file = out / 'state_graph.json'
    graph_file.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = {'schema': semantic.SCHEMA, 'variant': semantic.VARIANT, 'version': semantic.VERSION,
                'base_graph_manifest_sha256': semantic.boundary.GRAPH_MANIFEST_SHA256,
                'reference_manifest_sha256': semantic.boundary.REFERENCE_MANIFEST_SHA256,
                'files': {'state_graph.json': hashlib.sha256(graph_file.read_bytes()).hexdigest()},
                'source_files': {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(root.rglob('*')) if path.is_file()},
                'promotion_status': 'not_evaluated', 'model_calls': 0,
                'promotion_gate': {'n': 146, 'minimum_correct': 112, 'minimum_bacc': 107/146,
                                   'all_valid': True, 'engineering_and_citation_audit': True}}
    # BACC is the frozen ordinary baseline, not its ACC.
    manifest['promotion_gate']['minimum_bacc'] = .7321428571428571
    path = out / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'nodes': len(graph['nodes']), 'edges': len(graph['edges']), 'model_calls': 0}))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=semantic.MANIFEST.parent)
    build(out=parser.parse_args().out)
