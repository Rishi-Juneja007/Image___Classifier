# ---------------------------------------------------------------
# Image Feature Extraction & Classification App (Final Version)
# ---------------------------------------------------------------
# Developed by: Rishi Juneja
# For NEC Project Evaluation
# ---------------------------------------------------------------

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# -------------------- App Header --------------------
st.set_page_config(page_title="🧠 ImgClf NEC", layout="wide")
st.title("🧠 Image Feature Extraction & Classification App")
st.write("Upload an image to extract features, analyze color histograms, apply transformations, and classify it using a pretrained deep learning model (MobileNetV2).")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

model = load_model()

# -------------------- Helper Function --------------------
def prettify_label(label):
    """Convert technical labels like 'Ibizan_hound' to 'Ibizan Hound'."""
    return label.replace("_", " ").title()

# General label mapping
category_map = {
    "hound": "Dog",
    "retriever": "Dog",
    "terrier": "Dog",
    "shepherd": "Dog",
    "bulldog": "Dog",
    "spaniel": "Dog",
    "cat": "Cat",
    "car": "Vehicle",
    "truck": "Vehicle",
    "airplane": "Vehicle",
    "bus": "Vehicle",
    "train": "Vehicle",
    "ship": "Vehicle",
    "banana": "Fruit",
    "apple": "Fruit",
    "orange": "Fruit",
    "tiger": "Animal",
    "lion": "Animal",
    "elephant": "Animal",
    "zebra": "Animal",
    "bird": "Bird",
}

# -------------------- Upload Image --------------------
uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and show image
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    st.image(img, caption="Uploaded Image", width=500)
    img_array = np.array(img)

    # Save a copy for restoration
    original_img = img_array.copy()

    # -------------------- Transformations --------------------
    st.subheader("🧩 Image Transformations")

    colA, colB, colC = st.columns(3)

    with colA:
        if st.button("🔄 Rotate 90°"):
            img_array = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)

        if st.button("↔️ Flip Horizontally"):
            img_array = cv2.flip(img_array, 1)

    with colB:
        if st.button("⬇️ Flip Vertically"):
            img_array = cv2.flip(img_array, 0)

        if st.button("🌫 Apply Gaussian Blur"):
            img_array = cv2.GaussianBlur(img_array, (15, 15), 0)

    with colC:
        if st.button("🎞 Apply Motion Blur"):
            size = 15
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            img_array = cv2.filter2D(img_array, -1, kernel_motion_blur)

        if st.button("⚫ Binary Threshold"):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            img_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    if st.button("🔁 Restore Original"):
        img_array = original_img.copy()

    # Show transformed image
    st.image(img_array, caption="Transformed Image", width=500)

    # -------------------- Classification --------------------
    img_resized = cv2.resize(img_array, (224, 224))
    img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
    preds = model.predict(img_preprocessed)
    decoded_preds = decode_predictions(preds, top=3)[0]

    st.subheader("🎯 Classification Results:")
    for (_, label, prob) in decoded_preds:
        pretty_label = prettify_label(label)

        general_label = "Unknown"
        for key, value in category_map.items():
            if key.lower() in pretty_label.lower():
                general_label = value
                break

        st.write(f"**{pretty_label}** — {prob*100:.2f}%")
        if general_label != "Unknown":
            st.caption(f"🧩 General Category: {general_label}")

    # -------------------- Feature Extraction --------------------
    st.subheader("📊 Feature Extraction")

    col1, col2, col3 = st.columns(3)

    # 1️⃣ Color Histogram
    with col1:
        st.markdown("**Color Histogram**")
        fig, ax = plt.subplots()
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
            ax.set_xlim([0, 256])
        st.pyplot(fig)

    # 2️⃣ Edge Detection
    with col2:
        st.markdown("**Edge Detection (Canny)**")
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        st.image(edges, caption="Detected Edges", use_container_width=True, channels="GRAY")

    # 3️⃣ Texture (Sobel)
    with col3:
        st.markdown("**Texture using Sobel Filter**")
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_norm = cv2.normalize(sobel_combined, None, 0, 1, cv2.NORM_MINMAX)
        st.image(sobel_norm, caption="Texture (Sobel Gradient)", use_container_width=True, clamp=True)

    # -------------------- Save Results --------------------
    if st.button("💾 Save Classification Result"):
        with open("classification_result.txt", "w") as f:
            f.write("Image Classification Result\n")
            f.write("----------------------------\n")
            for (_, label, prob) in decoded_preds:
                pretty_label = prettify_label(label)
                f.write(f"{pretty_label}: {prob*100:.2f}%\n")
        st.success("✅ Result saved as 'classification_result.txt' in project folder.")

else:
    st.info("Please upload an image file to begin.")
