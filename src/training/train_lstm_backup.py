from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# NeuroGesture AI
# Phase 3C - LSTM Training
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "prepared"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
METADATA_PATH = DATA_DIR / "metadata.json"

MODEL_PATH = MODEL_DIR / "neurogesture_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.20
EPOCHS = 60
BATCH_SIZE = 16

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data():
    print("=" * 60)
    print("NEUROGESTURE AI - LSTM TRAINING")
    print("=" * 60)

    if not X_PATH.exists():
        raise FileNotFoundError(f"Missing: {X_PATH}")

    if not Y_PATH.exists():
        raise FileNotFoundError(f"Missing: {Y_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {METADATA_PATH}")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = metadata["classes"]

    print("\nDataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {len(class_names)}")

    return X, y, class_names


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.LSTM(
            32
        ),

        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    X, y, class_names = load_data()

    # --------------------------------------------------------
    # Validate data
    # --------------------------------------------------------

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to have 3 dimensions, got {X.shape}"
        )

    if X.shape[1:] != (30, 126):
        raise ValueError(
            f"Expected each sample to be (30, 126), "
            f"got {X.shape[1:]}"
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y contain different numbers of samples."
        )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print("\nSplit:")
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        input_shape=(30, 126),
        num_classes=len(class_names)
    )

    print("\nModel:")
    model.summary()

    # --------------------------------------------------------
    # Training callbacks
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    loss, accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    print(f"\nValidation loss:     {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # Save labels
    # --------------------------------------------------------

    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "classes": class_names
            },
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = MODEL_DIR / "training_history.json"

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
            f,
            indent=4
        )

    print("\nSaved:")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Labels:  {LABELS_PATH}")
    print(f"  History: {history_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()