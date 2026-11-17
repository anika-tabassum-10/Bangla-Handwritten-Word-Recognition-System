
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

# Prevent OpenCV from creating too many competing threads.
cv2.setNumThreads(1)


def find_image_root(data_dir: Path) -> Path:
    """Find directory containing numeric class folders 1..84."""
    candidates = [data_dir] + [
        p for p in data_dir.rglob("*") if p.is_dir()
    ]

    expected = {str(i) for i in range(1, 85)}

    for folder in candidates:
        numeric = {
            p.name for p in folder.iterdir()
            if p.is_dir() and p.name.isdigit()
        }

        if len(numeric.intersection(expected)) >= 60:
            return folder

    raise FileNotFoundError(
        f"Could not find class folders under {data_dir}"
    )


def prepare_image(path: Path) -> np.ndarray:
    """Preprocess one image and return a compact uint8 array."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unreadable image")

    _, binary = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binary = binary[y:y+h, x:x+w]

    h, w = binary.shape
    side = max(h, w) + 12

    canvas = np.zeros((side, side), dtype=np.uint8)

    y0 = (side - h) // 2
    x0 = (side - w) // 2

    canvas[y0:y0+h, x0:x0+w] = binary

    resized = cv2.resize(
        canvas,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return resized


def process_one(item):
    """Worker function: preprocess an image and return its class."""
    path, class_id = item

    try:
        image = prepare_image(path)
        return image, class_id, None

    except Exception as exc:
        return None, class_id, str(exc)


def load_data(data_dir: Path, mapping: dict, workers: int):
    image_root = find_image_root(data_dir)

    class_ids = sorted(
        [
            p.name for p in image_root.iterdir()
            if p.is_dir() and p.name.isdigit()
        ],
        key=int
    )

    missing = [
        cid for cid in class_ids
        if not mapping.get(cid)
        or "REPLACE" in str(mapping.get(cid))
    ]

    if missing:
        raise ValueError(
            "Complete labels.json first. Missing classes: "
            + ", ".join(missing)
        )

    # Collect image paths first.
    samples = []

    print("Scanning image folders...")

    for class_id in class_ids:
        folder = image_root / class_id

        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                samples.append((path, class_id))

    total = len(samples)

    if total == 0:
        raise RuntimeError("No readable image files found.")

    print(f"Found {total:,} image files.")
    print(f"Starting parallel preprocessing with {workers} workers...")

    images = []
    labels = []
    skipped = 0

    start = time.time()
    chunk_size = 2048

    # Process bounded chunks to avoid creating too many futures.
    with ThreadPoolExecutor(max_workers=workers) as executor:

        for start_index in range(0, total, chunk_size):

            chunk = samples[start_index:start_index + chunk_size]

            results = executor.map(process_one, chunk)

            for image, class_id, error in results:

                if image is None:
                    skipped += 1
                    continue

                images.append(image)
                labels.append(class_id)

            completed = min(start_index + len(chunk), total)

            elapsed = time.time() - start
            speed = completed / max(elapsed, 0.001)

            print(
                f"\rProcessed {completed:,}/{total:,} "
                f"({completed / total * 100:.1f}%) | "
                f"{speed:.0f} images/sec | "
                f"Skipped: {skipped}",
                end="",
                flush=True
            )

    print("\nImage preprocessing completed.")

    if not images:
        raise RuntimeError("No images could be processed.")

    unique_ids = sorted(set(labels), key=int)

    id_to_index = {
        cid: index for index, cid in enumerate(unique_ids)
    }

    y = np.array(
        [id_to_index[cid] for cid in labels],
        dtype=np.int64
    )

    # uint8 uses 1 byte per pixel instead of 4 bytes for float32.
    X = np.stack(images, axis=0)
    X = np.expand_dims(X, axis=-1)

    class_names = [mapping[cid] for cid in unique_ids]

    print(f"Successfully loaded {len(X):,} images.")
    print(f"Skipped images: {skipped}")
    print(f"Dataset memory: {X.nbytes / (1024**2):.1f} MB")

    return X, y, class_names, image_root, unique_ids


def build_model(num_classes: int):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(IMG_SIZE, IMG_SIZE, 1)
        ),

        # Convert uint8 pixels to float values between 0 and 1.
        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(
            32, 3, padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64, 3, padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128, 3, padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(
            num_classes, activation="softmax"
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

    parser = argparse.ArgumentParser(
        description="Train Bangla handwritten character CNN."
    )

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--run-name", default="bangla-cnn-run")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4)
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    data_dir = (
        ROOT / args.data_dir
        if not Path(args.data_dir).is_absolute()
        else Path(args.data_dir)
    )

    labels_path = ROOT / "labels.json"

    mapping = json.loads(
        labels_path.read_text(encoding="utf-8")
    )["class_to_character"]

    print("\nTensorFlow version:", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs:", gpus if gpus else "None detected")

    print("\nLoading dataset...")

    X, y, class_names, image_root, class_ids = load_data(
        data_dir, mapping, args.workers
    )

    print(f"\nDataset: {len(X):,} images")
    print(f"Classes: {len(class_names)}")

    # Stratified train-validation split.
    indices = np.arange(len(y))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    del X, y, indices, train_idx, val_idx

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(min(len(y_train), 20000))
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    mlflow_dir = ROOT / "artifacts" / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment("Bangla-Handwritten-Recognition")

    with mlflow.start_run(run_name=args.run_name):

        mlflow.log_params({
            "architecture": "3-block CNN",
            "image_size": IMG_SIZE,
            "classes": len(class_names),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "workers": args.workers,
            "validation_split": 0.20,
            "dataset": "BanglaLekha-Isolated"
        })

        model = build_model(len(class_names))

        model.optimizer.learning_rate.assign(
            args.learning_rate
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ROOT / "models" / "best_model.keras"),
                monitor="val_accuracy",
                save_best_only=True
            )
        ]

        print("\nStarting model training...")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )

        val_loss, val_accuracy = model.evaluate(
            val_ds, verbose=0
        )

        mlflow.log_metric(
            "validation_loss", float(val_loss)
        )

        mlflow.log_metric(
            "validation_accuracy", float(val_accuracy)
        )

        for epoch, values in enumerate(zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"]
        ), start=1):

            loss, acc, val_loss_epoch, val_acc = values

            mlflow.log_metrics({
                "train_loss": float(loss),
                "train_accuracy": float(acc),
                "val_loss": float(val_loss_epoch),
                "val_accuracy": float(val_acc)
            }, step=epoch)

        model.save(ROOT / "models" / "model.keras")

        output_labels = {
            "image_size": IMG_SIZE,
            "class_ids": class_ids,
            "class_names": class_names,
            "class_to_character": mapping
        }

        labels_path.write_text(
            json.dumps(
                output_labels,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        mlflow.log_artifact(str(labels_path))
        mlflow.log_artifact(
            str(ROOT / "models" / "model.keras")
        )

        print("\nTraining completed!")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Saved model: {ROOT / 'models' / 'model.keras'}")
        print(f"MLflow folder: {mlflow_dir}")


if __name__ == "__main__":
    main()
