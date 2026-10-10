"""
Computer Vision & Image Processing - Streamlit App
Covers:
1. Explanation of Computer Vision vs Image Processing (3 examples each)
2. Load RGB image, inspect dimensions & pixel values, convert to grayscale
3. Two enhancement / filtering operations (Gaussian Blur + CLAHE)
4. Comparison of original vs processed images with explanations

Run locally:  streamlit run app.py
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# st.image width argument differs between Streamlit versions.
# Pick the right one so the app runs on old and new versions.
def _ver(v):
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


_V = _ver(st.__version__)
if _V >= (1, 50):
    IMG_KW = {"width": "stretch"}
else:
    IMG_KW = {"use_container_width": True}

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Computer Vision & Image Processing",
    page_icon="🖼️",
    layout="wide",
)

MAX_WIDTH = 1000  # resize very large images so the app stays fast


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def make_sample_image(width: int = 640, height: int = 420) -> np.ndarray:
    """Create a synthetic RGB sample image (low contrast + noise)
    so the app works even when no image is uploaded."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Soft gradient background
    for x in range(width):
        v = int(90 + 60 * x / width)
        img[:, x] = (v, v - 10, v + 20)

    # Shapes (RGB order)
    cv2.rectangle(img, (60, 60), (240, 220), (150, 110, 110), -1)
    cv2.circle(img, (420, 150), 80, (110, 150, 120), -1)
    cv2.ellipse(img, (330, 320), (140, 50), 0, 0, 360, (120, 120, 170), -1)
    cv2.line(img, (30, 380), (610, 40), (170, 170, 140), 6)
    cv2.putText(img, "OpenCV", (70, 130), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (200, 200, 200), 3, cv2.LINE_AA)

    # Gaussian noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 18, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img  # RGB


def load_image(uploaded_file) -> np.ndarray:
    """Read an uploaded file with OpenCV and return an RGB array."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR -> convert
    h, w = rgb.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        rgb = cv2.resize(rgb, (MAX_WIDTH, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return rgb


def to_png_bytes(img: np.ndarray) -> bytes:
    """Encode an RGB or gray image to PNG bytes for download."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def histogram_df(img: np.ndarray, colour: bool) -> pd.DataFrame:
    """Return histogram data for a line chart."""
    if colour:
        data = {}
        for i, name in enumerate(["Red", "Green", "Blue"]):
            data[name] = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        return pd.DataFrame(data)
    return pd.DataFrame(
        {"Gray": cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()}
    )


def sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian - higher means sharper / more edges (noise too)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_clahe(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


# ----------------------------------------------------------------------
# Sidebar - image input & parameters
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader(
    "Upload an RGB image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

image_rgb = None
if uploaded is not None:
    image_rgb = load_image(uploaded)
    if image_rgb is None:
        st.sidebar.error("Could not read this image. Try another file.")

if image_rgb is None:
    st.sidebar.info("No image uploaded - using a built-in sample image.")
    image_rgb = make_sample_image()

st.sidebar.markdown("---")
st.sidebar.subheader("Operation 1: Gaussian Blur")
blur_k = st.sidebar.slider("Kernel size (odd)", 3, 31, 9, step=2)
blur_sigma = st.sidebar.slider("Sigma", 0.0, 10.0, 2.0, step=0.5)

st.sidebar.subheader("Operation 2: CLAHE")
clahe_clip = st.sidebar.slider("Clip limit", 1.0, 10.0, 3.0, step=0.5)
clahe_tile = st.sidebar.slider("Tile grid size", 2, 16, 8)

# ----------------------------------------------------------------------
# Core processing
# ----------------------------------------------------------------------
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
blurred_rgb = apply_gaussian_blur(image_rgb, blur_k, blur_sigma)
blurred_gray = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2GRAY)
clahe_gray = apply_clahe(gray, clahe_clip, clahe_tile)

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.title("🖼️ Computer Vision & Image Processing")
st.caption("Python + OpenCV + Streamlit")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1️⃣ Concepts",
        "2️⃣ Load & Inspect",
        "3️⃣ Enhancement / Filtering",
        "4️⃣ Comparison",
    ]
)

# ------------------------------ TAB 1 ---------------------------------
with tab1:
    st.header("Computer Vision vs Image Processing")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📷 Image Processing")
        st.markdown(
            """
**Image Processing** means applying mathematical operations on an image to
**improve it or transform it**. The *input is an image* and the *output is
also an image* (a better or modified version). It does **not** try to
understand what is in the picture.

**Three examples**
1. **Noise removal** - cleaning a grainy photo using Gaussian / median filters.
2. **Contrast enhancement** - histogram equalization or CLAHE to brighten a
   dark X-ray.
3. **Image resizing / compression** - shrinking a photo to JPEG for the web.
"""
        )
    with col_b:
        st.subheader("🧠 Computer Vision")
        st.markdown(
            """
**Computer Vision** is a field of AI that enables computers to **understand
and interpret** the visual world. The *input is an image/video* and the
*output is information or a decision* (labels, locations, measurements).
It often *uses* image processing as a first step.

**Three examples**
1. **Face recognition** - unlocking a phone by identifying who the face belongs to.
2. **Object detection in self-driving cars** - finding pedestrians, cars and
   traffic signs.
3. **Medical diagnosis** - detecting tumours in MRI / CT scans automatically.
"""
        )

    st.markdown("### Key differences")
    st.table(
        pd.DataFrame(
            {
                "Aspect": ["Goal", "Input", "Output", "Intelligence", "Typical tools"],
                "Image Processing": [
                    "Enhance / modify the image",
                    "Image",
                    "Image",
                    "Rule-based, low-level",
                    "Filters, FFT, morphology",
                ],
                "Computer Vision": [
                    "Understand the image",
                    "Image / video",
                    "Labels, boxes, decisions",
                    "Learning-based, high-level",
                    "CNNs, YOLO, ViT, SLAM",
                ],
            }
        )
    )

# ------------------------------ TAB 2 ---------------------------------
with tab2:
    st.header("Load an RGB image, inspect it, convert to grayscale")

    h, w, c = image_rgb.shape
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Height (rows)", h)
    m2.metric("Width (columns)", w)
    m3.metric("Channels", c)
    m4.metric("Data type", str(image_rgb.dtype))
    st.write(f"**Shape:** `{image_rgb.shape}`  |  **Total pixels:** `{h * w:,}`  |  "
             f"**Array size:** `{image_rgb.nbytes / 1024:.1f} KB`")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original (RGB)")
        st.image(image_rgb, **IMG_KW)
    with col2:
        st.subheader("Grayscale")
        st.image(gray, **IMG_KW, clamp=True)
        st.write(f"Grayscale shape: `{gray.shape}`  (1 channel)")

    st.markdown("#### 🔍 Inspect pixel values")
    px1, px2 = st.columns(2)
    x = px1.slider("X (column)", 0, w - 1, w // 2)
    y = px2.slider("Y (row)", 0, h - 1, h // 2)

    r, g, b = image_rgb[y, x]
    st.write(
        f"Pixel at (x={x}, y={y}) → **R={r}, G={g}, B={b}**  |  "
        f"Gray value = **{gray[y, x]}**"
    )
    st.markdown(
        f"<div style='width:80px;height:40px;border:1px solid #888;"
        f"background:rgb({r},{g},{b});'></div>",
        unsafe_allow_html=True,
    )

    # 5x5 patch of pixel values around selected point
    y0, y1 = max(0, y - 2), min(h, y + 3)
    x0, x1 = max(0, x - 2), min(w, x + 3)
    patch = gray[y0:y1, x0:x1]
    st.write("Grayscale 5×5 neighbourhood (numeric pixel values):")
    st.dataframe(
        pd.DataFrame(
            patch,
            index=[f"y={i}" for i in range(y0, y1)],
            columns=[f"x={j}" for j in range(x0, x1)],
        )
    )

    st.markdown("#### 📊 Channel statistics")
    stats = pd.DataFrame(
        {
            "Channel": ["Red", "Green", "Blue", "Gray"],
            "Min": [image_rgb[..., i].min() for i in range(3)] + [gray.min()],
            "Max": [image_rgb[..., i].max() for i in range(3)] + [gray.max()],
            "Mean": [round(float(image_rgb[..., i].mean()), 2) for i in range(3)]
            + [round(float(gray.mean()), 2)],
            "Std Dev": [round(float(image_rgb[..., i].std()), 2) for i in range(3)]
            + [round(float(gray.std()), 2)],
        }
    )
    st.dataframe(stats, hide_index=True)

    st.markdown("#### Histograms")
    hc1, hc2 = st.columns(2)
    hc1.write("RGB histogram")
    hc1.line_chart(histogram_df(image_rgb, True))
    hc2.write("Grayscale histogram")
    hc2.line_chart(histogram_df(gray, False))

    with st.expander("How grayscale conversion works"):
        st.markdown(
            """
OpenCV converts RGB → Gray with the weighted (luminosity) formula:

`Gray = 0.299 × R + 0.587 × G + 0.114 × B`

Green gets the highest weight because the human eye is most sensitive to green.
"""
        )

    with st.expander("Code used"):
        st.code(
            """import cv2
img_bgr = cv2.imread("image.jpg")              # OpenCV loads as BGR
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(img_rgb.shape)                            # (height, width, channels)
print(img_rgb[100, 150])                        # [R G B] at row 100, col 150
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
""",
            language="python",
        )

# ------------------------------ TAB 3 ---------------------------------
with tab3:
    st.header("Two enhancement / filtering operations")

    st.subheader("Operation 1 - Gaussian Blur (smoothing filter)")
    c1, c2 = st.columns(2)
    c1.image(image_rgb, caption="Original", **IMG_KW)
    c2.image(
        blurred_rgb,
        caption=f"Gaussian Blur (kernel={blur_k if blur_k % 2 else blur_k + 1}, sigma={blur_sigma})",
        **IMG_KW,
    )
    st.download_button(
        "⬇️ Download blurred image",
        to_png_bytes(blurred_rgb),
        file_name="gaussian_blur.png",
        mime="image/png",
    )

    st.markdown("---")
    st.subheader("Operation 2 - CLAHE (contrast enhancement)")
    c3, c4 = st.columns(2)
    c3.image(gray, caption="Grayscale (original)", **IMG_KW, clamp=True)
    c4.image(
        clahe_gray,
        caption=f"CLAHE (clip={clahe_clip}, tiles={clahe_tile}×{clahe_tile})",
        **IMG_KW,
        clamp=True,
    )
    st.download_button(
        "⬇️ Download CLAHE image",
        to_png_bytes(clahe_gray),
        file_name="clahe_enhanced.png",
        mime="image/png",
    )

    with st.expander("Code used"):
        st.code(
            """blurred = cv2.GaussianBlur(img_rgb, (9, 9), 2.0)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
""",
            language="python",
        )

# ------------------------------ TAB 4 ---------------------------------
with tab4:
    st.header("Original vs Processed - comparison")

    cc1, cc2, cc3 = st.columns(3)
    cc1.image(gray, caption="Original (grayscale)", **IMG_KW, clamp=True)
    cc2.image(blurred_gray, caption="After Gaussian Blur", **IMG_KW, clamp=True)
    cc3.image(clahe_gray, caption="After CLAHE", **IMG_KW, clamp=True)

    st.markdown("#### Difference from original (amplified ×3)")
    d1, d2 = st.columns(2)
    d1.image(
        np.clip(cv2.absdiff(gray, blurred_gray).astype(np.int32) * 3, 0, 255).astype(np.uint8),
        caption="|Original − Blur|",
        **IMG_KW,
        clamp=True,
    )
    d2.image(
        np.clip(cv2.absdiff(gray, clahe_gray).astype(np.int32) * 3, 0, 255).astype(np.uint8),
        caption="|Original − CLAHE|",
        **IMG_KW,
        clamp=True,
    )

    st.markdown("#### Numeric comparison")
    comp = pd.DataFrame(
        {
            "Image": ["Original (gray)", "Gaussian Blur", "CLAHE"],
            "Mean brightness": [round(float(a.mean()), 2) for a in (gray, blurred_gray, clahe_gray)],
            "Contrast (Std Dev)": [round(float(a.std()), 2) for a in (gray, blurred_gray, clahe_gray)],
            "Sharpness (Laplacian var)": [round(sharpness(a), 2) for a in (gray, blurred_gray, clahe_gray)],
        }
    )
    st.dataframe(comp, hide_index=True)

    st.markdown("#### Histogram comparison")
    hist_all = pd.DataFrame(
        {
            "Original": cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten(),
            "Gaussian Blur": cv2.calcHist([blurred_gray], [0], None, [256], [0, 256]).flatten(),
            "CLAHE": cv2.calcHist([clahe_gray], [0], None, [256], [0, 256]).flatten(),
        }
    )
    st.line_chart(hist_all)

    st.markdown("### 📝 Explanation of each operation")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            """
**1. Gaussian Blur**

- A *low-pass* filter: each pixel is replaced by a **weighted average** of its
  neighbours, with weights following a Gaussian (bell-curve) shape.
- **Effect:** reduces noise and fine detail, so the image looks smoother.
  Edges become softer.
- **Parameters:** a larger *kernel* or *sigma* → stronger blur.
- **Result in the table:** the *sharpness* value drops, and the contrast
  (std-dev) stays similar or slightly decreases.
- **Use cases:** noise reduction before edge detection, pre-processing for
  object detection.
"""
        )
    with e2:
        st.markdown(
            """
**2. CLAHE (Contrast Limited Adaptive Histogram Equalization)**

- Divides the image into small **tiles** and equalizes the histogram of each
  tile separately, then blends the borders.
- The **clip limit** stops noise from being over-amplified.
- **Effect:** local contrast improves, so dark and hidden details become
  visible. The histogram spreads over a wider range of intensities.
- **Result in the table:** the *contrast (std-dev)* increases, and
  the sharpness usually increases too.
- **Use cases:** medical imaging (X-ray, MRI), underwater and low-light photos.
"""
        )

    st.info(
        "**Summary:** Gaussian Blur *smooths* the image (reduces noise/detail) "
        "while CLAHE *enhances contrast* (reveals detail). They do opposite "
        "jobs, which is why the sharpness and contrast numbers move in "
        "different directions."
    )

st.markdown("---")
st.caption("Built with Streamlit & OpenCV")
