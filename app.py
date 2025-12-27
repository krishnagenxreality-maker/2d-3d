import streamlit as st
import os
import tempfile
import shutil
import base64
import streamlit.components.v1 as components
from PIL import Image
import datetime
from core.converter import convert_floorplan_to_3d

current_year = datetime.datetime.now().year

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Floor Plan to 3D Converter",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# STYLING (Optimized for Single Viewport)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Remove padding to fit screen */
.block-container {
    max-width: 1400px;
    padding: 1rem 2rem !important;
}

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
    overflow: hidden; /* Prevent body scroll */
}

#MainMenu, footer, header { visibility: hidden; }

h1 {
    font-size: 2rem !important;
    font-weight: 700;
    text-align: center;
    color: #f8fafc;
    margin: 0 !important;
    padding: 0 !important;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

/* Force specific height for image to prevent vertical jump */
[data-testid="stImage"] img {
    max-height: 250px;
    object-fit: contain;
    border-radius: 12px;
}

section[data-testid="stFileUploader"] {
    background: #020617;
    border: 1px dashed #334155;
    padding: 0.5rem;
    border-radius: 12px;
}

.stButton > button {
    height: 48px;
    font-weight: 600;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    border: none;
}

.stDownloadButton {
    display: flex;
    justify-content: center;
}

.stDownloadButton > button {
    background: #1e293b;
    border: 1px solid #3b82f6;
    color: #93c5fd;
    width: 200px !important;
    height: 50px;
}

.placeholder {
    text-align: center;
    color: #64748b;
    height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed #334155;
    border-radius: 15px;
}

/* Footer Tucking */
.custom-footer {
    position: fixed;
    bottom: 10px;
    width: 100%;
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "converted" not in st.session_state:
    st.session_state.converted = False
if "result" not in st.session_state:
    st.session_state.result = None

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1>Floor Plan to 3D Converter</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a 2D floor plan and instantly generate a realistic 3D model</p>", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown("##### 1️⃣ Upload Floor Plan")
    
    uploaded_file = st.file_uploader(
        "Upload",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)

        if not st.session_state.converted:
            if st.button("🚀 Convert to 3D Model", use_container_width=True):
                with st.spinner("Generating..."):
                    output_dir = tempfile.mkdtemp()
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    texture_dir = os.path.join(script_dir, "textures")

                    try:
                        uploaded_file.seek(0)
                        result = convert_floorplan_to_3d(
                            image_data=uploaded_file.read(),
                            output_dir=output_dir,
                            texture_dir=texture_dir,
                            progress_callback=lambda s, t, m: None
                        )
                        st.session_state.converted = True
                        st.session_state.result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            if st.button("🔄 New Conversion", use_container_width=True):
                st.session_state.converted = False
                st.rerun()

with right_col:
    st.markdown("##### 2️⃣ Preview & Download")

    if st.session_state.converted and st.session_state.result:
        result = st.session_state.result
        glb_path = result["files"]["glb"]

        if os.path.exists(glb_path):
            with open(glb_path, "rb") as f:
                glb_bytes = f.read()
                glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
            
            viewer_html = f"""
                <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
                <model-viewer 
                    src="data:model/gltf-binary;base64,{glb_base64}" 
                    style="width: 100%; height: 380px; background-color: #020617; border-radius: 15px; border: 1px solid #334155;"
                    camera-controls auto-rotate shadow-intensity="2" exposure="0.8">
                </model-viewer>
            """
            components.html(viewer_html, height=390)
            
            # Centered Download Button
            st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label=" Download GLB Model",
                data=glb_bytes,
                file_name="model.glb",
                mime="model/gltf-binary",
                use_container_width=False # CSS handles centering
            )
    else:
        st.markdown(
            "<div class='placeholder'>3D preview will appear here</div>",
            unsafe_allow_html=True
        )

# --------------------------------------------------
# FIXED FOOTER
# --------------------------------------------------
st.markdown(
    f"<div class='custom-footer'>© {current_year} GenXreality</div>",
    unsafe_allow_html=True
)