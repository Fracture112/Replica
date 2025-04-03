# ChatGPT-level fracture reasoning app – code will follow import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
from openai import OpenAI
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import re
from skimage.feature import canny
from scipy.ndimage import center_of_mass

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- UTILS ---
def detect_crack_origin(gray):
    edges = canny(gray, sigma=2)
    y_coords, x_coords = np.nonzero(edges)
    if len(x_coords) == 0:
        return None
    cy, cx = center_of_mass(edges)
    return int(cx), int(cy)

def detect_beach_marks(gray):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    gradient_magnitude = np.mean(np.abs(sobel))
    return gradient_magnitude > 3.0

def analyze_image_features(image):
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    h, w = gray.shape
    top_mean = np.mean(edges[:h//3, :])
    bottom_mean = np.mean(edges[2*h//3:, :])
    symmetry = abs(top_mean - bottom_mean)

    desc = "This is a metal fracture surface.\\n"
    desc += f"Edge symmetry difference: {symmetry:.2f}. "
    desc += "Symmetric → possible bending.\\n" if symmetry < 12 else "Asymmetric → localized overload likely.\\n"

    beach_detected = detect_beach_marks(gray)
    desc += "Beach marks detected.\\n" if beach_detected else "No beach marks visible.\\n"

    origin = detect_crack_origin(gray)
    if origin:
        desc += f"Estimated crack origin at pixel location: {origin}.\\n"
    else:
        desc += "No clear crack origin found.\\n"

    return edges, desc, origin, beach_detected

def generate_gpt_analysis(description):
    prompt = f\"\"\"
You are a metallurgical failure analyst.

Image description:
{description}

Rules:
- If beach marks are visible, classify as fatigue.
- If beach marks are not visible, classify as overload.
- If symmetry is present, likely reversed bending.
- If origin is detected, say how many and where.
- Add confidence % for stress types.

Return this format:

| Feature | Analysis |
|--------|----------|
| Failure Mode | Fatigue / Overload |
| Type of Stress | Most likely |
| Beach Marks | Present / Not visible |
| Chevron Marks | Present / Absent |
| Origin Count | 0 / 1 / 2+ |
| Additional Notes | Surface indicators |

Also give confidence for:
Bending: 40%
Torsional: 30%
Tensile: 20%
Reversed Bending: 5%
Rotating Bending: 5%
\"\"\"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional fracture analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )
    return response.choices[0].message.content

def extract_confidence(text):
    confidence = {}
    for line in text.split("\\n"):
        match = re.match(r"(Bending|Torsional|Tensile|Reversed Bending|Rotating Bending)[:\\s]+(\\d+)%", line.strip(), re.I)
        if match:
            confidence[match.group(1)] = int(match.group(2))
    return confidence

# --- UI ---
st.set_page_config(page_title="ANDALAN FRACTOGRAPHY SOLVER", layout="centered")
st.title("🧠 ANDALAN FRACTOGRAPHY SOLVER – GPT-Replica")

uploaded_file = st.file_uploader("Upload a fracture image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    edges, description, origin, beach_mark_flag = analyze_image_features(image)

    st.subheader("Edge Detection")
    st.image(edges, clamp=True, use_column_width=True)

    st.subheader("GPT Analysis Result")
    with st.spinner("Analyzing..."):
        result = generate_gpt_analysis(description)

    parts = result.split("\\n\\n", 1)
    summary = parts[0]
    table = parts[1] if len(parts) > 1 else ""

    st.markdown("**Summary:**")
    st.markdown(summary)
    st.markdown("### Fracture Table")
    st.markdown(table)

    confidence = extract_confidence(result)
    if confidence:
        st.markdown("### Stress Type Confidence")
        fig, ax = plt.subplots()
        ax.barh(list(confidence.keys()), list(confidence.values()))
        ax.set_xlim(0, 100)
        st.pyplot(fig)

else:
    st.info("👆 Please upload a fracture image to begin.")
shortly.
