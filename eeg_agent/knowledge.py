"""Small local BM25 knowledge index with source-grounded stable citations."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .config import PROJECT, write_json

INDEX_VERSION = 'vrms-bm25-1.0'
CHUNK_SIZE = 600
OVERLAP = 100
_STOP = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are', 'was', 'can', 'not',
         '一个', '可以', '不能', '当前', '我们', '以及', '进行', '因为', '根据', '采用', '这个'}


def _terms(text):
    text = str(text).lower()
    result = re.findall(r'[a-z][a-z0-9_\-]*', text)
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        # Deterministic Chinese bigrams avoid a Pi-side segmenter dependency.
        result.extend(run[i:i+2] for i in range(max(0, len(run)-1)))
    return [term for term in result if term not in _STOP]


def _hash(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _sections(text):
    title, lines = '正文', []
    for line in text.splitlines():
        if re.match(r'^#{1,6}\s+', line):
            if lines and '\n'.join(lines).strip():
                yield title, '\n'.join(lines).strip()
            title, lines = re.sub(r'^#{1,6}\s+', '', line).strip(), []
        else:
            lines.append(line)
    if lines and '\n'.join(lines).strip():
        yield title, '\n'.join(lines).strip()


class KnowledgeIndex:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.root = Path(self.cfg.get('knowledge_root', PROJECT / 'knowledge'))
        output = Path(self.cfg.get('output_root', PROJECT / 'outputs_v2'))
        self.index_path = Path(self.cfg.get('knowledge_index_path', output / 'knowledge' / 'bm25_index.json'))
        self.chunks = []
        self.document_hashes = {}
        self._fingerprint = None
        self._loaded = False

    def _documents(self):
        if not self.root.exists():
            return []
        return [(p.relative_to(self.root).as_posix(), p.read_text(encoding='utf-8'))
                for p in sorted(self.root.rglob('*.md')) if p.is_file()]

    def build(self):
        docs = self._documents()
        chunks = []
        self.document_hashes = {source: _hash(text) for source, text in docs}
        for source, text in docs:
            version_match = re.search(r'^资料版本[：:]\s*(.+)$', text, re.MULTILINE)
            version = version_match.group(1).strip() if version_match else '1.0'
            kind_match = re.search(r'^证据类型[：:]\s*(.+)$', text, re.MULTILINE)
            kind = kind_match.group(1).strip() if kind_match else 'project_definition'
            for section, body in _sections(text):
                # Section boundaries never merge. Use paragraph ends when possible.
                start = 0
                while start < len(body):
                    end = min(start + CHUNK_SIZE, len(body))
                    if end < len(body):
                        boundary = body.rfind('\n\n', start + 300, end)
                        if boundary > start:
                            end = boundary
                    content = body[start:end].strip()
                    if content:
                        identity = '\n'.join((source, section, version, content))
                        chunks.append({'citation_id': 'kb:' + _hash(identity)[:20],
                                       'source': source, 'section': section, 'version': version,
                                       'source_sha256': self.document_hashes[source],
                                       'content_sha256': _hash(content), 'evidence_type': kind,
                                       'text': content, 'terms': _terms(content)})
                    if end >= len(body):
                        break
                    start = max(start + 1, end - OVERLAP)
        self.chunks = chunks
        self._fingerprint = _hash(json.dumps(self.document_hashes, sort_keys=True, ensure_ascii=False))
        self._loaded = True
        write_json(self.index_path, {'index_version': INDEX_VERSION, 'fingerprint': self._fingerprint,
                                    'chunk_characters': CHUNK_SIZE, 'overlap_characters': OVERLAP,
                                    'document_hashes': self.document_hashes, 'chunks': chunks})
        return self.summary()

    def summary(self):
        return {'index_version': INDEX_VERSION, 'sources': len(self.document_hashes),
                'chunks': len(self.chunks), 'fingerprint': self._fingerprint,
                'chunk_characters': CHUNK_SIZE, 'overlap_characters': OVERLAP,
                'maximum_results': 5, 'index_path': str(self.index_path),
                'status': 'ready' if self.chunks else 'insufficient'}

    def _ensure(self):
        if self._loaded:
            return
        docs = self._documents()
        hashes = {source: _hash(text) for source, text in docs}
        if self.index_path.exists():
            try:
                saved = json.loads(self.index_path.read_text(encoding='utf-8'))
                if saved.get('index_version') == INDEX_VERSION and saved.get('document_hashes') == hashes:
                    self.chunks = saved['chunks']
                    self.document_hashes = hashes
                    self._fingerprint = saved['fingerprint']
                    self._loaded = True
                    return
            except (ValueError, KeyError, OSError, TypeError):
                pass
        self.build()

    def search(self, query, limit=5):
        self._ensure()
        limit = max(0, min(5, int(limit)))
        terms = set(_terms(query))
        n = len(self.chunks)
        if not n or not terms or not limit:
            return {'status': 'insufficient', 'reason': 'no matching local evidence',
                    'query': str(query), 'results': [], 'index_version': INDEX_VERSION}
        frequencies = Counter()
        for chunk in self.chunks:
            frequencies.update(set(chunk['terms']))
        mean_length = sum(len(c['terms']) for c in self.chunks)/float(n)
        scored = []
        for chunk in self.chunks:
            counts = Counter(chunk['terms'])
            score = 0.
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = frequencies[term]
                idf = math.log(1. + (n-df+.5)/(df+.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * len(chunk['terms'])/max(mean_length, 1.)))
            if score > 0. and math.isfinite(score):
                scored.append((score, chunk['citation_id'], chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = [{key: value for key, value in chunk.items() if key != 'terms'}
                   for _, _, chunk in scored[:limit]]
        for result, (score, _, _) in zip(results, scored):
            result['score'] = float(score)
            result['id'] = result['citation_id']
        return {'status': 'ok' if results else 'insufficient',
                'reason': None if results else 'no matching local evidence',
                'query': str(query), 'results': results, 'index_version': INDEX_VERSION,
                'fingerprint': self._fingerprint}
