from pathlib import Path
import numpy as np

root = Path("data/gestures")

print()
print("=" * 60)
print("       NEUROGESTURE AI - DATASET CHECK")
print("=" * 60)

if not root.exists():
    print("\nERROR: data/gestures folder was not found.")
    raise SystemExit(1)

gesture_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

if not gesture_dirs:
    print("\nERROR: No gesture folders found.")
    raise SystemExit(1)

total = 0

print("\nGESTURES")
print("-" * 60)

for gesture_dir in gesture_dirs:
    files = sorted(gesture_dir.glob("*.npy"))
    count = len(files)
    total += count

    print(f"{gesture_dir.name:20} : {count} samples")

print("-" * 60)
print(f"{'TOTAL':20} : {total} samples")

print("\nSAMPLE DETAILS")
print("-" * 60)

for gesture_dir in gesture_dirs:
    files = sorted(gesture_dir.glob("*.npy"))

    if not files:
        print(f"{gesture_dir.name}: NO .npy FILES")
        continue

    sample = files[0]

    try:
        data = np.load(sample, allow_pickle=False)

        print(f"\nGesture : {gesture_dir.name}")
        print(f"File    : {sample.name}")
        print(f"Shape   : {data.shape}")
        print(f"Dtype   : {data.dtype}")
        print(f"Min     : {data.min():.6f}")
        print(f"Max     : {data.max():.6f}")

    except Exception as e:
        print(f"\nERROR reading {sample}")
        print(f"Reason: {e}")

print()
print("=" * 60)
print("DATASET CHECK COMPLETE")
print("=" * 60)