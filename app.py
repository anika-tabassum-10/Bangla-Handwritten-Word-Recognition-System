import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.keras"
LABELS_PATH = ROOT / "labels.json"
IMG_SIZE = 64

st.set_page_config(page_title="Bangla Handwritten Word Recognition", page_icon="✍️", layout="centered")
st.title("বাংলা হাতের লেখার শব্দ শনাক্তকরণ")
st.write("Draw one Bangla word in the canvas. The app segments the drawing, predicts each segment, and joins the predictions.")

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, metadata

def prepare_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGBA2GRAY) if crop.ndim == 3 and crop.shape[2] == 4 else (
        cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    )
    # Canvas is white/transparent with dark strokes.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    binary = binary[y:y+h, x:x+w]
    h, w = binary.shape
    side = max(h, w) + 12
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = binary
    square = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return square.astype("float32") / 255.0

def segment_word(rgba):
    """Simple vertical-projection segmentation; connected Bangla matra may prevent splitting."""
    rgb = rgba[:, :, :3].astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    projection = np.sum(ink > 0, axis=0)
    active = projection > 0
    runs = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= 3:
                runs.append((start, i))
            start = None
    if start is not None and len(active) - start >= 3:
        runs.append((start, len(active)))

    # Merge segments separated by very narrow gaps (small writing breaks).
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= 5:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)

    # If matra connects the entire word, projection may yield one segment.
    # Keep it as one crop rather than returning no prediction.
    if not merged:
        coords = cv2.findNonZero(ink)
        if coords is None:
            return []
        x, y, w, h = cv2.boundingRect(coords)
        merged = [(x, x + w)]
    segments = []
    for x1, x2 in merged:
        crop = rgba[:, max(0, x1-3):min(rgba.shape[1], x2+3), :]
        prepared = prepare_crop(crop)
        if prepared is not None:
            segments.append(prepared)
    return segments

def predict_segments(model, metadata, segments):
    class_names = metadata.get("class_names", [])
    if not class_names:
        raise ValueError("class_names missing in labels.json. Retrain the model.")
    batch = np.expand_dims(np.array(segments, dtype=np.float32), axis=-1)
    probabilities = model.predict(batch, verbose=0)
    predictions = []
    for row in probabilities:
        idx = int(np.argmax(row))
        predictions.append((class_names[idx], float(row[idx])))
    return predictions

try:
    model, metadata = load_assets()
except Exception as error:
    st.error(str(error))
    st.stop()

st.subheader("Draw your word")
canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=7,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=220,
    width=650,
    drawing_mode="freedraw",
    key="bangla_word_canvas"
)

if st.button("Recognize word", type="primary"):
    if canvas.image_data is None or np.max(canvas.image_data[:, :, :3]) == 0:
        st.warning("Please draw a word first.")
    else:
        rgba = canvas.image_data.astype(np.uint8)
        parts = segment_word(rgba)
        if not parts:
            st.warning("No handwriting detected. Please draw again.")
        else:
            predictions = predict_segments(model, metadata, parts)
            final_word = "".join(item[0] for item in predictions)
            st.success(f"Predicted word: {final_word}")
            st.caption("Predicted segments and confidence")
            for i, (character, confidence) in enumerate(predictions, start=1):
                st.write(f"{i}. {character} — {confidence:.1%}")
            st.info(
                "This is a character-classifier plus basic segmentation demonstration. "
                "Connected letters, matra, vowel signs and conjuncts may be segmented incorrectly."
            )

st.divider()
st.caption("Model: CNN trained on BanglaLekha-Isolated. Dataset contains isolated character samples, not complete word images.")
