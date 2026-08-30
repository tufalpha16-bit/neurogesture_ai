from pathlib import Path
import json

import numpy as np


# ============================================================
# NeuroGesture AI - Phase 3B
# Dataset Preparation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GESTURE_DIR = PROJECT_ROOT / "data" / "gestures"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared"

EXPECTED_FRAMES = 30
EXPECTED_FEATURES = 126


def load_dataset():
    sequences = []
    labels = []

    gesture_dirs = sorted(
        path for path in GESTURE_DIR.iterdir()
        if path.is_dir()
    )

    if not gesture_dirs:
        raise RuntimeError("No gesture folders found.")

    class_names = [path.name for path in gesture_dirs]
    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    print("\nClasses:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")

    print("\nLoading samples...")

    for gesture_dir in gesture_dirs:
        label = class_to_index[gesture_dir.name]
        files = sorted(gesture_dir.glob("*.npy"))

        print(f"  {gesture_dir.name}: {len(files)} samples")

        for file_path in files:
            try:
                data = np.load(file_path, allow_pickle=False)

                if data.shape != (EXPECTED_FRAMES, EXPECTED_FEATURES):
                    print(
                        f"WARNING: Skipping {file_path.name} "
                        f"because shape is {data.shape}, "
                        f"expected {(EXPECTED_FRAMES, EXPECTED_FEATURES)}"
                    )
                    continue

                if not np.isfinite(data).all():
                    print(
                        f"WARNING: Skipping {file_path.name} "
                        "because it contains NaN/Inf values."
                    )
                    continue

                sequences.append(data.astype(np.float32))
                labels.append(label)

            except Exception as exc:
                print(f"WARNING: Could not read {file_path}: {exc}")

    if not sequences:
        raise RuntimeError("No valid samples were found.")

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return X, y, class_names


def main():
    print("=" * 60)
    print("       NEUROGESTURE AI - DATASET PREPARATION")
    print("=" * 60)

    if not GESTURE_DIR.exists():
        raise RuntimeError(
            f"Dataset folder not found:\n{GESTURE_DIR}"
        )

    X, y, class_names = load_dataset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    metadata = {
        "classes": class_names,
        "num_classes": len(class_names),
        "num_samples": int(len(X)),
        "sequence_length": int(X.shape[1]),
        "features_per_frame": int(X.shape[2]),
        "dtype": str(X.dtype),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {len(class_names)}")
    print(f"Samples: {len(X)}")

    print("\nClass distribution:")

    for index, name in enumerate(class_names):
        count = int(np.sum(y == index))
        print(f"  {name:20} {count}")

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'X.npy'}")
    print(f"  {OUTPUT_DIR / 'y.npy'}")
    print(f"  {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()