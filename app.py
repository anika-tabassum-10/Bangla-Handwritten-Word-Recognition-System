import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.keras"
LABELS_PATH = ROOT / "labels.json"
IMG_SIZE = 64

st.set_page_config(page_title="Bangla Handwritten Word Recognition", page_icon="✍️", layout="centered")
st.title("বাংলা হাতের লেখার শব্দ শনাক্তকরণ")
st.write("Draw one Bangla word in the canvas. The app removes the matra (headline), "
         "segments the word into characters, predicts each one, and joins them into a word.")


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, metadata


def canvas_to_ink(image_data):
    """Canvas RGBA -> uint8 mask, ink = 255, background = 0 (same polarity as train.py)."""
    arr = image_data.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3:4] / 255.0
        rgb = arr[:, :, :3] * alpha + 255.0 * (1.0 - alpha)  # composite on white
    else:
        rgb = arr[:, :, :3]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return ((gray < 128).astype(np.uint8)) * 255


def prepare_crop(ink):
    """Identical to train.py prepare_image: crop to ink, pad to square, resize to 64x64.
    Returns uint8 0..255. NOT divided by 255: the model already contains a Rescaling layer."""
    coords = cv2.findNonZero(ink)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    ink = ink[y:y + h, x:x + w]
    h, w = ink.shape
    side = max(h, w) + 12
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = ink
    return cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def find_runs(active, min_width, max_gap):
    runs, start = [], None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(active)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_width]


def segment_word(ink):
    """Remove the matra, split on vertical gaps, return 64x64 uint8 character crops (matra kept)."""
    coords = cv2.findNonZero(ink)
    if coords is None:
        return []
    x, y, w, h = cv2.boundingRect(coords)
    ink = ink[y:y + h, x:x + w]
    h, w = ink.shape

    body = ink.copy()
    row_sum = (ink > 0).sum(axis=1).astype(np.float32)
    top = max(1, int(h * 0.5))
    r = int(np.argmax(row_sum[:top]))
    # A headline is a row in the upper half that is inked across most of the word width.
    if w > 40 and row_sum[r] >= 0.6 * w:
        r0 = r1 = r
        thr = 0.6 * row_sum[r]
        while r0 > 0 and row_sum[r0 - 1] >= thr:
            r0 -= 1
        while r1 < h - 1 and row_sum[r1 + 1] >= thr:
            r1 += 1
        body[max(0, r0 - 1):r1 + 2, :] = 0  # cut the matra so letters separate

    active = (body > 0).sum(axis=0) > 0
    runs = find_runs(active, min_width=max(3, int(w * 0.01)), max_gap=max(2, int(w * 0.01)))
    if not runs:
        runs = [(0, w)]

    segments = []
    for x1, x2 in runs:
        crop = ink[:, max(0, x1 - 2):min(w, x2 + 2)]  # original ink, matra included (as in training)
        if cv2.countNonZero(crop) < 15:
            continue
        prepared = prepare_crop(crop)
        if prepared is not None:
            segments.append(prepared)
    return segments


def predict_segments(model, metadata, segments):
    class_names = metadata.get("class_names", [])
    if not class_names:
        raise ValueError("class_names missing in labels.json. Retrain the model.")
    # 0..255 float; the model's Rescaling(1/255) layer normalises it.
    batch = np.expand_dims(np.array(segments, dtype=np.float32), axis=-1)
    probabilities = model.predict(batch, verbose=0)
    return [(class_names[int(np.argmax(row))], float(np.max(row))) for row in probabilities]


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
    key="bangla_word_canvas",
)

if st.button("Recognize word", type="primary"):
    if canvas.image_data is None:
        st.warning("Please draw a word first.")
    else:
        ink = canvas_to_ink(canvas.image_data)
        if cv2.countNonZero(ink) == 0:
            st.warning("No handwriting detected. Please draw again.")
        else:
            parts = segment_word(ink)
            if not parts:
                st.warning("No handwriting detected. Please draw again.")
            else:
                predictions = predict_segments(model, metadata, parts)
                final_word = "".join(p[0] for p in predictions)
                st.success(f"Predicted word: {final_word}")
                st.caption("Segments the model saw, with predicted character and confidence")
                cols = st.columns(min(len(parts), 8))
                for i, (img, (ch, conf)) in enumerate(zip(parts, predictions)):
                    with cols[i % len(cols)]:
                        st.image(img, width=64, clamp=True)
                        st.write(f"{ch}  \n{conf:.0%}")
                st.info("Character classifier plus matra-removal segmentation. Vowel signs, "
                        "conjuncts and touching letters may still be split incorrectly.")

st.divider()
st.caption("Model: CNN trained on BanglaLekha-Isolated (isolated characters, not word images).")