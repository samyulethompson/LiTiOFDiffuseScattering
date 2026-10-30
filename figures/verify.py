#!/usr/bin/env python3
"""Verify bundled file hashes and compare regenerated PNGs to article references."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    failed = False
    for bundle in sorted(ROOT.glob('figure_*')):
        entries = json.loads((bundle / 'provenance.json').read_text())
        bad = [entry['bundled'] for entry in entries
               if sha256(bundle / entry['bundled']) != entry['sha256']]
        print(f'{bundle.name}: {len(entries) - len(bad)}/{len(entries)} provenance hashes match')
        for path in bad:
            print(f'  HASH MISMATCH: {path}')
        failed |= bool(bad)
        reference = next((bundle / 'reference').glob('*.png'))
        output = bundle / 'output' / reference.name
        if not output.exists():
            print('  Missing output PNG; run this bundle’s plot.py first.')
            failed = True
            continue
        with Image.open(reference) as image:
            expected = np.asarray(image.convert('RGBA'))
        with Image.open(output) as image:
            actual = np.asarray(image.convert('RGBA'))
        exact = np.array_equal(expected, actual)
        print(f'  PNG pixels identical: {exact}; reference shape={expected.shape}; output shape={actual.shape}')
        failed |= not exact
    return int(failed)

if __name__ == '__main__':
    raise SystemExit(main())
