"""Hash-checked portable frozen assets; never reads query labels or datasets."""
import hashlib
import json
from pathlib import Path

import joblib


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class Assets:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.value = json.loads(self.path.read_text(encoding='utf-8'))
        if (self.value.get('schema') != 'online_cpu_seed2026_v1'
                or self.value.get('seed') != 2026
                or self.value.get('baseline_policy') != 'disabled'
                or self.value.get('dtype') != 'float32'):
            raise ValueError('Expected frozen seed2026 FP32 online assets without resting reference')
        self._verified = {}

    def item(self, fold, name):
        if name == 'encoder':
            record = self.value['encoder']
        else:
            record = self.value['folds'].get(str(fold), {}).get(name)
            if record is None:
                raise ValueError('Frozen asset unavailable for this held-out subject')
        relative = Path(record['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Asset paths must remain inside the portable bundle')
        path = (self.path.parent / relative).resolve()
        if not path.is_relative_to(self.path.parent):
            raise ValueError('Asset escapes the portable bundle')
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        if self._verified.get(path) != stamp:
            if sha(path) != record['sha256']:
                raise ValueError('Frozen model asset hash differs')
            self._verified[path] = stamp
        return path, record['sha256']

    def optional(self, fold, name):
        return self.item(fold, name)[0] if name in self.value['folds'][str(fold)] else None

    def bundle(self, fold, name):
        path, expected = self.item(fold, name)
        value = joblib.load(path)
        split = value['split']
        if split['outer_subject'] != fold or any(fold in split.get(key, []) for key in
                ('base_train', 'meta_calibration', 'policy_validation',
                 'head_fit', 'probability_calibration')):
            raise ValueError('Query overlaps numerical fitting subjects')
        return value, expected
