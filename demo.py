import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# Config
MODEL_PATH = "model/quickdraw_model.keras"
CATEGORIES_PATH = "model/categories.npy"

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    categories = list(np.load(CATEGORIES_PATH, allow_pickle=True))
    return model, categories

#Preprocess: crop to drawing bounds, then resize to 28x28
def preprocess(image_data):
    img_rgba = Image.fromarray(image_data.astype("uint8"), "RGBA")

    # Flatten onto white background then convert to grayscale
    bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
    bg.paste(img_rgba, mask=img_rgba.split()[3])
    gray = bg.convert("L")

    # Invert: model expects white drawing on black
    gray = ImageOps.invert(gray)

    # Crop to drawing bounds
    bbox = gray.getbbox()
    if bbox:
        gray = gray.crop(bbox)

    # Add padding
    w, h = gray.size
    pad = max(int(max(w, h) * 0.15), 4)
    gray = ImageOps.expand(gray, border=pad, fill=0)

    # Resize to 28x28
    gray = gray.resize((28, 28), Image.LANCZOS)

    arr = np.array(gray).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# Page setup 
st.set_page_config(page_title="Animal Doodle Classifier", layout="wide")

# CSS 
st.markdown("""
<style>
    .stApp {
        background-color: #fff8f0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='320'%3E%3C!-- squiggles --%3E%3Cpath d='M5 20 Q15 12 25 20 Q35 28 45 20' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M60 8 Q70 0 80 8 Q90 16 100 8' stroke='%23feca57' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M120 18 Q130 10 140 18 Q150 26 160 18 Q170 10 180 18' stroke='%2348dbfb' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M200 5 Q210 -3 220 5 Q230 13 240 5' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M3 55 Q13 47 23 55 Q33 63 43 55 Q53 47 63 55' stroke='%2354a0ff' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M90 48 Q100 40 110 48 Q120 56 130 48' stroke='%231dd1a1' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M160 60 Q170 52 180 60 Q190 68 200 60 Q210 52 220 60' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M8 100 Q18 92 28 100 Q38 108 48 100' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M70 90 Q80 82 90 90 Q100 98 110 90 Q120 82 130 90' stroke='%2348dbfb' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M155 105 Q165 97 175 105 Q185 113 195 105' stroke='%231dd1a1' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M215 95 Q225 87 235 95 Q245 103 255 95 Q265 87 275 95' stroke='%2354a0ff' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M5 145 Q15 137 25 145 Q35 153 45 145 Q55 137 65 145' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M85 138 Q95 130 105 138 Q115 146 125 138' stroke='%23feca57' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M148 150 Q158 142 168 150 Q178 158 188 150 Q198 142 208 150' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M10 190 Q20 182 30 190 Q40 198 50 190' stroke='%231dd1a1' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M68 200 Q78 192 88 200 Q98 208 108 200 Q118 192 128 200' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M150 188 Q160 180 170 188 Q180 196 190 188' stroke='%23feca57' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M210 198 Q220 190 230 198 Q240 206 250 198 Q260 190 270 198' stroke='%2354a0ff' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M5 238 Q15 230 25 238 Q35 246 45 238 Q55 230 65 238' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M88 245 Q98 237 108 245 Q118 253 128 245' stroke='%2348dbfb' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M150 235 Q160 227 170 235 Q180 243 190 235 Q200 227 210 235' stroke='%231dd1a1' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M240 260 Q250 252 260 260 Q270 268 280 260 Q290 252 300 260' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M5 285 Q15 277 25 285 Q35 293 45 285 Q55 277 65 285' stroke='%23feca57' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M100 275 Q110 267 120 275 Q130 283 140 275' stroke='%2354a0ff' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M270 130 Q280 122 290 130 Q300 138 310 130' stroke='%231dd1a1' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M270 190 Q280 182 290 190 Q300 198 310 190' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3C!-- spirals --%3E%3Cpath d='M52 28 Q57 23 62 28 Q67 35 60 38 Q52 41 49 34 Q47 25 54 21' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M195 70 Q200 65 205 70 Q210 77 203 80 Q195 83 192 76 Q190 67 197 63' stroke='%23feca57' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M52 118 Q57 113 62 118 Q67 125 60 128 Q52 131 49 124 Q47 115 54 111' stroke='%2348dbfb' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M140 165 Q145 160 150 165 Q155 172 148 175 Q140 178 137 171 Q135 162 142 158' stroke='%231dd1a1' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M255 225 Q260 220 265 225 Q270 232 263 235 Q255 238 252 231 Q250 222 257 218' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3Cpath d='M295 55 Q300 50 305 55 Q310 62 303 65 Q295 68 292 61 Q290 52 297 48' stroke='%2354a0ff' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3C!-- arrows --%3E%3Cpath d='M255 25 L272 25 M267 19 L273 25 L267 31' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M30 125 L47 125 M42 119 L48 125 L42 131' stroke='%23feca57' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M170 220 L187 220 M182 214 L188 220 L182 226' stroke='%2348dbfb' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M285 170 L285 187 M279 182 L285 188 L291 182' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C!-- zigzags --%3E%3Cpath d='M108 30 L115 22 L122 30 L129 22 L136 30' stroke='%2354a0ff' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M0 75 L7 67 L14 75 L21 67 L28 75' stroke='%23ff6b6b' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M210 120 L217 112 L224 120 L231 112 L238 120' stroke='%231dd1a1' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M108 175 L115 167 L122 175 L129 167 L136 175' stroke='%23feca57' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M0 260 L7 252 L14 260 L21 252 L28 260' stroke='%2348dbfb' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M290 240 L297 232 L304 240 L311 232 L318 240' stroke='%23ff9ff3' stroke-width='1.8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C!-- stars --%3E%3Cpath d='M230 35 L232 29 L234 35 L240 35 L235 39 L237 45 L232 41 L227 45 L229 39 L224 35 Z' stroke='%23feca57' stroke-width='1.2' fill='%23feca57' opacity='0.5'/%3E%3Cpath d='M78 160 L80 154 L82 160 L88 160 L83 164 L85 170 L80 166 L75 170 L77 164 L72 160 Z' stroke='%23ff9ff3' stroke-width='1.2' fill='%23ff9ff3' opacity='0.5'/%3E%3Cpath d='M175 270 L177 264 L179 270 L185 270 L180 274 L182 280 L177 276 L172 280 L174 274 L169 270 Z' stroke='%2348dbfb' stroke-width='1.2' fill='%2348dbfb' opacity='0.5'/%3E%3Cpath d='M305 105 L307 99 L309 105 L315 105 L310 109 L312 115 L307 111 L302 115 L304 109 L299 105 Z' stroke='%23ff6b6b' stroke-width='1.2' fill='%23ff6b6b' opacity='0.5'/%3E%3Cpath d='M20 200 L22 194 L24 200 L30 200 L25 204 L27 210 L22 206 L17 210 L19 204 L14 200 Z' stroke='%231dd1a1' stroke-width='1.2' fill='%231dd1a1' opacity='0.5'/%3E%3Cpath d='M130 75 L132 69 L134 75 L140 75 L135 79 L137 85 L132 81 L127 85 L129 79 L124 75 Z' stroke='%2354a0ff' stroke-width='1.2' fill='%2354a0ff' opacity='0.5'/%3E%3Cpath d='M280 220 L282 214 L284 220 L290 220 L285 224 L287 230 L282 226 L277 230 L279 224 L274 220 Z' stroke='%23feca57' stroke-width='1.2' fill='%23feca57' opacity='0.5'/%3E%3Cpath d='M50 120 L52 114 L54 120 L60 120 L55 124 L57 130 L52 126 L47 130 L49 124 L44 120 Z' stroke='%23ff6b6b' stroke-width='1.2' fill='%23ff6b6b' opacity='0.4'/%3E%3Cpath d='M200 300 L202 294 L204 300 L210 300 L205 304 L207 310 L202 306 L197 310 L199 304 L194 300 Z' stroke='%23ff9ff3' stroke-width='1.2' fill='%23ff9ff3' opacity='0.5'/%3E%3Cpath d='M310 250 L312 244 L314 250 L320 250 L315 254 L317 260 L312 256 L307 260 L309 254 L304 250 Z' stroke='%231dd1a1' stroke-width='1.2' fill='%231dd1a1' opacity='0.4'/%3E%3C!-- hearts --%3E%3Cpath d='M165 32 Q165 28 169 28 Q173 28 173 32 Q173 36 165 40 Q157 36 157 32 Q157 28 161 28 Q165 28 165 32Z' stroke='%23ff6b6b' stroke-width='1.5' fill='%23ff6b6b' opacity='0.4'/%3E%3Cpath d='M305 295 Q305 291 309 291 Q313 291 313 295 Q313 299 305 303 Q297 299 297 295 Q297 291 301 291 Q305 291 305 295Z' stroke='%23ff9ff3' stroke-width='1.5' fill='%23ff9ff3' opacity='0.4'/%3E%3Cpath d='M45 305 Q45 301 49 301 Q53 301 53 305 Q53 309 45 313 Q37 309 37 305 Q37 301 41 301 Q45 301 45 305Z' stroke='%23feca57' stroke-width='1.5' fill='%23feca57' opacity='0.4'/%3E%3C!-- small circles --%3E%3Ccircle cx='78' cy='30' r='3' fill='none' stroke='%23ff6b6b' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='145' cy='55' r='3' fill='none' stroke='%23feca57' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='265' cy='75' r='3' fill='none' stroke='%2348dbfb' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='35' cy='170' r='3' fill='none' stroke='%23ff9ff3' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='180' cy='135' r='3' fill='none' stroke='%231dd1a1' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='100' cy='220' r='3' fill='none' stroke='%2354a0ff' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='290' cy='15' r='3' fill='none' stroke='%23feca57' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='240' cy='180' r='3' fill='none' stroke='%23ff6b6b' stroke-width='1.5' opacity='0.7'/%3E%3Ccircle cx='55' cy='250' r='3' fill='none' stroke='%231dd1a1' stroke-width='1.5' opacity='0.7'/%3E%3C!-- filled dots --%3E%3Ccircle cx='200' cy='40' r='2' fill='%2354a0ff' opacity='0.6'/%3E%3Ccircle cx='310' cy='70' r='2' fill='%23ff9ff3' opacity='0.6'/%3E%3Ccircle cx='160' cy='130' r='2' fill='%23feca57' opacity='0.6'/%3E%3Ccircle cx='70' cy='210' r='2' fill='%23ff6b6b' opacity='0.6'/%3E%3Ccircle cx='220' cy='270' r='2' fill='%2348dbfb' opacity='0.6'/%3E%3Ccircle cx='315' cy='155' r='2' fill='%231dd1a1' opacity='0.6'/%3E%3C!-- crosses --%3E%3Cpath d='M248 88 L254 94 M254 88 L248 94' stroke='%23ff6b6b' stroke-width='1.8' stroke-linecap='round'/%3E%3Cpath d='M18 143 L24 149 M24 143 L18 149' stroke='%231dd1a1' stroke-width='1.8' stroke-linecap='round'/%3E%3Cpath d='M130 228 L136 234 M136 228 L130 234' stroke='%23feca57' stroke-width='1.8' stroke-linecap='round'/%3E%3Cpath d='M283 278 L289 284 M289 278 L283 284' stroke='%2354a0ff' stroke-width='1.8' stroke-linecap='round'/%3E%3C!-- triangles --%3E%3Cpath d='M295 310 L305 295 L315 310 Z' stroke='%231dd1a1' stroke-width='1.5' fill='none' opacity='0.6'/%3E%3Cpath d='M15 310 L25 295 L35 310 Z' stroke='%23ff6b6b' stroke-width='1.5' fill='none' opacity='0.6'/%3E%3Cpath d='M135 5 L145 -10 L155 5 Z' stroke='%23feca57' stroke-width='1.5' fill='none' opacity='0.6'/%3E%3C!-- wavy double lines --%3E%3Cpath d='M270 45 Q278 40 286 45 Q294 50 302 45' stroke='%23ff9ff3' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3Cpath d='M270 50 Q278 45 286 50 Q294 55 302 50' stroke='%23ff9ff3' stroke-width='1' fill='none' stroke-linecap='round' opacity='0.5'/%3E%3Cpath d='M0 170 Q8 165 16 170 Q24 175 32 170' stroke='%23feca57' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3Cpath d='M0 175 Q8 170 16 175 Q24 180 32 175' stroke='%23feca57' stroke-width='1' fill='none' stroke-linecap='round' opacity='0.5'/%3E%3C/svg%3E");
        background-size: 320px 320px;
        background-repeat: repeat;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.93);
        border-radius: 24px;
        padding: 1.5rem 3rem !important;
        margin: 1rem auto !important;
        max-width: 900px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    .big-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
    .pill-container {
        display: flex; flex-wrap: wrap; gap: 0.3rem;
        justify-content: center; margin: 0.4rem 0;
    }
    .pill {
        background: rgba(255,255,255,0.8);
        border: 1.5px solid #ddd; border-radius: 999px;
        padding: 0.1rem 0.6rem; font-size: 0.75rem;
        color: #555; font-weight: 500;
    }
    .pred-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 1.2rem 1.5rem;
        color: white; text-align: center; margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .pred-animal { font-size: 2rem; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; }
    .pred-conf { font-size: 0.95rem; opacity: 0.85; margin-top: 0.2rem; }
    .bar-row { display: flex; align-items: center; margin: 0.3rem 0; gap: 0.5rem; }
    .bar-label { width: 85px; font-size: 0.82rem; font-weight: 600; color: #444; text-transform: capitalize; }
    .bar-track { flex: 1; background: #e9ecef; border-radius: 999px; height: 13px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; }
    .bar-pct { width: 40px; font-size: 0.78rem; color: #666; text-align: right; }
    .section-header { font-size: 0.8rem; font-weight: 700; color: #888; margin: 0.7rem 0 0.3rem 0; text-transform: uppercase; letter-spacing: 1px; }
    .empty-state { text-align: center; color: #bbb; font-size: 0.95rem; padding: 2.5rem 0; }
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white; border: none; border-radius: 12px;
        font-weight: 700; width: 100%; margin-top: 0.4rem;
    }
    .stButton > button:hover { opacity: 0.85; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Session state 
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Load
model, categories = load_model()

# Header
st.markdown('<div class="big-title">Animal Doodle Guesser</div>', unsafe_allow_html=True)

pills_html = '<div class="pill-container">' + "".join(
    f'<span class="pill">{c}</span>' for c in categories
) + '</div>'
st.markdown(pills_html, unsafe_allow_html=True)
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1])
BAR_COLORS = ["#ff6b6b", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff"]

with col1:
    st.markdown('<div class="section-header">Draw here</div>', unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=8,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
    )
    if st.button("Clear canvas"):
        st.session_state.canvas_key += 1
        st.rerun()

with col2:
    st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

    if canvas.image_data is not None:
        alpha = canvas.image_data[:, :, 3]
        has_drawing = alpha.sum() > 1000

        if has_drawing:
            img = preprocess(canvas.image_data)
            probs = model.predict(img, verbose=0)[0]
            top_idx = np.argmax(probs)
            top_cat = categories[top_idx]
            top_conf = probs[top_idx] * 100

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-animal">{top_cat}</div>
                <div class="pred-conf">Confidence: {top_conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Top 5 guesses</div>', unsafe_allow_html=True)
            top5_idx = np.argsort(probs)[::-1][:5]
            bars_html = ""
            for i, idx in enumerate(top5_idx):
                cat = categories[idx]
                conf = probs[idx] * 100
                bars_html += f"""
                <div class="bar-row">
                    <div class="bar-label">{cat}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{conf:.1f}%;background:{BAR_COLORS[i]}"></div>
                    </div>
                    <div class="bar-pct">{conf:.1f}%</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">Start drawing to see the prediction!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Start drawing to see the prediction!</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Trained on Google Quick Draw dataset · 28 animal categories · CNN model")