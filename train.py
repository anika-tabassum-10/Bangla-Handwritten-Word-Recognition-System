import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
IMG_SIZE = 64
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

cv2.setNumThreads(1)


def find_image_root(data_dir: Path) -> Path:
    candidates = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]
    expected = {str(i) for i in range(1, 85)}
    for folder in candidates:
        numeric = {p.name for p in folder.iterdir() if p.is_dir() and p.name.isdigit()}
        if len(numeric.intersection(expected)) >= 60:
            return folder
    raise FileNotFoundError(f"Could not find class folders under {data_dir}")


def prepare_image(path: Path) -> np.ndarray:
    """Grayscale -> Otsu inverse (ink=255) -> crop -> pad square -> 64x64 uint8.
    app.py uses exactly the same steps."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unreadable image")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binary = binary[y:y + h, x:x + w]
    h, w = binary.shape
    side = max(h, w) + 12
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = binary
    return cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def process_one(item):
    path, class_id = item
    try:
        return prepare_image(path), class_id, None
    except Exception as exc:
        return None, class_id, str(exc)


def load_data(data_dir: Path, mapping: dict, workers: int):
    image_root = find_image_root(data_dir)
    class_ids = sorted(
        [p.name for p in image_root.iterdir() if p.is_dir() and p.name.isdigit()], key=int
    )
    missing = [c for c in class_ids if not mapping.get(c) or "REPLACE" in str(mapping.get(c))]
    if missing:
        raise ValueError("Complete labels.json first. Missing classes: " + ", ".join(missing))

    samples = []
    print("Scanning image folders...")
    for class_id in class_ids:
        for path in (image_root / class_id).rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                samples.append((path, class_id))

    total = len(samples)
    if total == 0:
        raise RuntimeError("No readable image files found.")
    print(f"Found {total:,} image files. Preprocessing with {workers} workers...")

    images, labels, skipped = [], [], 0
    start = time.time()
    chunk_size = 2048
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for s in range(0, total, chunk_size):
            chunk = samples[s:s + chunk_size]
            for image, class_id, error in executor.map(process_one, chunk):
                if image is None:
                    skipped += 1
                    continue
                images.append(image)
                labels.append(class_id)
            completed = min(s + len(chunk), total)
            speed = completed / max(time.time() - start, 0.001)
            print(f"\rProcessed {completed:,}/{total:,} ({completed / total * 100:.1f}%) | "
                  f"{speed:.0f} img/s | Skipped: {skipped}", end="", flush=True)
    print("\nPreprocessing completed.")

    if not images:
        raise RuntimeError("No images could be processed.")

    unique_ids = sorted(set(labels), key=int)
    id_to_index = {cid: i for i, cid in enumerate(unique_ids)}
    y = np.array([id_to_index[c] for c in labels], dtype=np.int64)
    X = np.expand_dims(np.stack(images, axis=0), axis=-1)  # uint8, 0..255
    class_names = [mapping[cid] for cid in unique_ids]
    print(f"Loaded {len(X):,} images, {X.nbytes / 1024**2:.1f} MB")
    return X, y, class_names, image_root, unique_ids


def build_model(num_classes: int, learning_rate: float, dropout: float):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        # The model normalises internally: app.py must feed raw 0..255 pixels.
        tf.keras.layers.Rescaling(1.0 / 255),

        # Light augmentation (active only during training) so hand-drawn canvas
        # strokes generalise better than scanned forms.
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.1),

        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Bangla handwritten character CNN.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--run-name", default="bangla-cnn-run")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else ROOT / args.data_dir
    labels_path = ROOT / "labels.json"
    mapping = json.loads(labels_path.read_text(encoding="utf-8"))["class_to_character"]

    print("\nTensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs:", gpus if gpus else "None detected")

    X, y, class_names, image_root, class_ids = load_data(data_dir, mapping, args.workers)
    print(f"\nDataset: {len(X):,} images, {len(class_names)} classes")

    # Stratified 70 / 15 / 15 split: train / validation / test.
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=y[temp_idx])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    del X, y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(min(len(y_train), 20000)).batch(args.batch_size).prefetch(AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size).prefetch(AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(args.batch_size).prefetch(AUTOTUNE)

    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    mlflow_dir = ROOT / "artifacts" / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment("Bangla-Handwritten-Recognition")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "architecture": "3-block CNN + augmentation",
            "image_size": IMG_SIZE,
            "classes": len(class_names),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "workers": args.workers,
            "split": "70/15/15",
            "dataset": "BanglaLekha-Isolated",
        })

        model = build_model(len(class_names), args.learning_rate, args.dropout)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ROOT / "models" / "best_model.keras"),
                monitor="val_accuracy", save_best_only=True),
        ]

        print("\nStarting model training...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                            callbacks=callbacks, verbose=1)

        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        mlflow.log_metrics({
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_accuracy),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        })

        for epoch, (l, a, vl, va) in enumerate(zip(
                history.history["loss"], history.history["accuracy"],
                history.history["val_loss"], history.history["val_accuracy"]), start=1):
            mlflow.log_metrics({"train_loss": float(l), "train_accuracy": float(a),
                                "val_loss": float(vl), "val_accuracy": float(va)}, step=epoch)

        model.save(ROOT / "models" / "model.keras")

        labels_path.write_text(json.dumps({
            "image_size": IMG_SIZE,
            "class_ids": class_ids,
            "class_names": class_names,
            "class_to_character": mapping,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(labels_path))
        mlflow.log_artifact(str(ROOT / "models" / "model.keras"))

        print("\nTraining completed!")
        print(f"Validation accuracy: {val_accuracy:.4f} | Test accuracy: {test_accuracy:.4f}")
        print(f"Saved model: {ROOT / 'models' / 'model.keras'}")


if __name__ == "__main__":
    main()