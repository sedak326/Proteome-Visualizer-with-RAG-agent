#!/usr/bin/env python
"""Generate only the animations that include polarbear and/or hippo (new species)."""

import subprocess
import sys
from itertools import combinations
from pathlib import Path

ALL_SPECIES = ['sealion', 'bottlenose', 'graywhale', 'orca', 'harborseal', 'polarbear', 'hippo']
NEW_SPECIES = {'polarbear', 'hippo'}
MAX_COMBO_SIZE = 3  # animations capped at 3 species

output_dir = Path('prerendered_animations')
output_dir.mkdir(exist_ok=True)

combos = []
for size in range(1, MAX_COMBO_SIZE + 1):
    for combo in combinations(ALL_SPECIES, size):
        if NEW_SPECIES & set(combo):  # only combos that include at least one new species
            combos.append(sorted(combo))

print(f"Generating {len(combos)} new animations...")

ok, fail = 0, 0
for i, species_list in enumerate(combos, 1):
    name = '_'.join(species_list) + '.gif'
    out = output_dir / name
    last_frame = output_dir / name.replace('.gif', '_last_frame.png')

    if out.exists():
        print(f"[{i}/{len(combos)}] SKIP (exists): {name}")
        ok += 1
        continue

    print(f"[{i}/{len(combos)}] Generating: {name}")
    cmd = [sys.executable, 'animal_morph_fixed.py'] + species_list + ['--output', str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and out.exists():
            mb = out.stat().st_size / 1024 / 1024
            print(f"  OK ({mb:.1f} MB)")
            ok += 1
        else:
            print(f"  FAIL: {result.stderr[:300]}")
            fail += 1
    except subprocess.TimeoutExpired:
        print(f"  FAIL: timeout")
        fail += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        fail += 1

print(f"\nDone. OK={ok} FAIL={fail}")
