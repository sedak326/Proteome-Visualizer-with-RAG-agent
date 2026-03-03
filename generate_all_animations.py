#!/usr/bin/env python
# We pre-generate all animal-morph animations to save time later on 
"""
Generate all pre-rendered animation combinations for proteome explorer
Generates 15 animations: 5 singles + 10 pairs
"""

import subprocess
import sys
from itertools import combinations
from pathlib import Path

# All available species
SPECIES = ['sealion', 'bottlenose', 'graywhale', 'orca', 'harborseal', 'cobra']

def generate_all_animations():
    """Generate all non-empty subset combinations (singles through all-six)"""

    output_dir = Path('prerendered_animations')
    output_dir.mkdir(exist_ok=True, parents=True)

    all_combinations = []

    # All subsets of size 1 through len(SPECIES)
    for size in range(1, len(SPECIES) + 1):
        for combo in combinations(SPECIES, size):
            all_combinations.append(list(combo))

    print(f"Generating {len(all_combinations)} animations...")
    print("=" * 60)

    successful = 0
    failed = 0

    for i, species_list in enumerate(all_combinations, 1):
        species_str = '_'.join(sorted(species_list))
        output_path = output_dir / f"{species_str}.gif"

        print(f"\n[{i}/{len(all_combinations)}] Generating: {' + '.join(species_list)}")
        print(f"    Output: {output_path}")

        # Run the animation script
        cmd = [sys.executable, 'animal_morph_fixed.py'] + species_list + ['--output', str(output_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                print(f"    [OK] Success! ({file_size:.2f} MB)")
                successful += 1
            else:
                print(f"    [FAIL] Failed!")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                failed += 1

        except subprocess.TimeoutExpired:
            print(f"    [FAIL] Timeout (>120s)")
            failed += 1
        except Exception as e:
            print(f"    [FAIL] Error: {str(e)[:100]}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Generation complete!")
    print(f"  [OK] Successful: {successful}/{len(all_combinations)}")
    print(f"  [FAIL] Failed: {failed}/{len(all_combinations)}")
    print(f"  Output directory: {output_dir.absolute()}")
    print("=" * 60)

    return successful, failed

if __name__ == '__main__':
    print("Proteome Explorer - Animation Pre-rendering")
    print("=" * 60)
    print(f"This will generate 63 animations (all subsets of {len(SPECIES)} species):")
    print("  - 6 singles, 15 pairs, 20 triples, 15 quads, 6 quints, 1 all-six")
    print("\nEstimated time: ~2 hours")
    print("=" * 60)
    print("\nStarting generation...")

    successful, failed = generate_all_animations()

    if failed == 0:
        print("\n[OK] All animations generated successfully!")
        sys.exit(0)
    else:
        print(f"\n[WARNING] {failed} animation(s) failed to generate")
        sys.exit(1)
