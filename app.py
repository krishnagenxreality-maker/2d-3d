"""
Floor Plan to 3D Converter
==========================
Streamlit web application for converting 2D floor plan images to 3D models.
Upload a PNG floor plan, click Convert, and download the OBJ/GLB files.
"""

import streamlit as st
import os
import tempfile
import shutil
import zipfile
from io import BytesIO
from PIL import Image
import time

from core.converter import convert_floorplan_to_3d

# Page configuration
st.set_page_config(
    page_title="Floor Plan to 3D Converter",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, aligned design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Container styling */
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px !important;
    }
    
    /* Header section */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .subtitle {
        color: #8892b0;
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .card-icon {
        font-size: 1.5rem;
    }
    
    .card-title {
        color: #e6f1ff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .card-subtitle {
        color: #8892b0;
        font-size: 0.85rem;
        margin: 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 184, 148, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 184, 148, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Image container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #8892b0;
        margin-top: 0.25rem;
    }
    
    /* Success banner */
    .success-banner {
        background: rgba(0, 184, 148, 0.1);
        border: 1px solid rgba(0, 184, 148, 0.3);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .success-icon {
        font-size: 1.5rem;
    }
    
    .success-text h4 {
        color: #00b894;
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
    }
    
    .success-text p {
        color: #8892b0;
        margin: 0.25rem 0 0 0;
        font-size: 0.85rem;
    }
    
    /* Placeholder */
    .placeholder {
        text-align: center;
        padding: 3rem 2rem;
        color: #8892b0;
    }
    
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .placeholder h3 {
        color: #a8b2d1;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .placeholder p {
        font-size: 0.9rem;
    }
    
    /* Progress styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #5a6a8a;
        font-size: 0.85rem;
    }
    
    /* Section header */
    .section-header {
        color: #e6f1ff;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'converted' not in st.session_state:
    st.session_state.converted = False
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None
if 'result' not in st.session_state:
    st.session_state.result = None

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">🏠 Floor Plan to 3D</h1>
    <p class="subtitle">Transform your 2D floor plans into detailed 3D models with AI-powered wall detection</p>
</div>
""", unsafe_allow_html=True)

# Main layout - two equal columns
col1, col2 = st.columns(2, gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN - Upload Section
# ═══════════════════════════════════════════════════════════════════════════════
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-icon">📤</span>
            <div>
                <h3 class="card-title">Upload Floor Plan</h3>
                <p class="card-subtitle">PNG, JPG formats supported</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload floor plan",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear floor plan image. Black lines on white background work best.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display image
        st.image(image, caption="", use_container_width=True)
        
        # Image stats
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{image.size[0]}</div>
                <div class="stat-label">Width (px)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{image.size[1]}</div>
                <div class="stat-label">Height (px)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{image.mode}</div>
                <div class="stat-label">Color Mode</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">📁</div>
            <h3>No file uploaded</h3>
            <p>Drag and drop or click to upload</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN - Convert & Download Section
# ═══════════════════════════════════════════════════════════════════════════════
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-icon">🔄</span>
            <div>
                <h3 class="card-title">Convert & Download</h3>
                <p class="card-subtitle">Generate 3D model with textures</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Convert button
        if not st.session_state.converted:
            if st.button("🚀 Convert to 3D Model", use_container_width=True, type="primary"):
                output_dir = tempfile.mkdtemp(prefix="floorplan_3d_")
                st.session_state.output_dir = output_dir
                
                script_dir = os.path.dirname(os.path.abspath(__file__))
                texture_dir = os.path.join(script_dir, "textures")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(step, total, message):
                    progress_bar.progress(step / total)
                    status_text.markdown(f"**Step {step}/{total}:** {message}")
                
                try:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    result = convert_floorplan_to_3d(
                        image_data=image_bytes,
                        output_dir=output_dir,
                        texture_dir=texture_dir,
                        progress_callback=update_progress
                    )
                    
                    st.session_state.converted = True
                    st.session_state.result = result
                    
                    progress_bar.progress(1.0)
                    status_text.empty()
                    time.sleep(0.3)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.converted = False
        
        # Show results
        if st.session_state.converted and st.session_state.result:
            result = st.session_state.result
            
            # Success message
            st.markdown("""
            <div class="success-banner">
                <span class="success-icon">✅</span>
                <div class="success-text">
                    <h4>Conversion Complete!</h4>
                    <p>Your 3D model is ready for download</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Stats
            st.markdown('<div class="section-header">📊 Detection Results</div>', unsafe_allow_html=True)
            
            stats = result['stats']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Horizontal", stats['horizontal_walls'])
            c2.metric("Vertical", stats['vertical_walls'])
            c3.metric("Diagonal", stats['diagonal_walls'])
            c4.metric("Total", stats['total_walls'])
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Preview
            st.markdown('<div class="section-header">🔍 Wall Detection Preview</div>', unsafe_allow_html=True)
            if os.path.exists(result['files']['preview']):
                preview_img = Image.open(result['files']['preview'])
                st.image(preview_img, use_container_width=True)
                st.caption("🟢 Horizontal  🔵 Vertical  🟡 Diagonal")
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Downloads and 3D Preview Section
            st.markdown('<div class="section-header">🎮 3D Model Preview & Downloads</div>', unsafe_allow_html=True)
            
            # Create two columns: 3D preview on left, downloads on right
            preview_col, download_col = st.columns([3, 2])
            
            with preview_col:
                # 3D GLB Preview using Three.js
                if os.path.exists(result['files']['glb']):
                    import base64
                    with open(result['files']['glb'], 'rb') as f:
                        glb_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Three.js viewer embedded in HTML
                    threejs_viewer = f"""
                    <div id="glb-viewer-container" style="width: 100%; height: 350px; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
                        <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/GLTFLoader.js"></script>
                        <script>
                            (function() {{
                                const container = document.getElementById('glb-viewer-container');
                                const scene = new THREE.Scene();
                                scene.background = new THREE.Color(0x1a1a2e);
                                
                                const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
                                camera.position.set(5, 5, 5);
                                
                                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                                renderer.setSize(container.clientWidth, container.clientHeight);
                                renderer.setPixelRatio(window.devicePixelRatio);
                                renderer.shadowMap.enabled = true;
                                container.appendChild(renderer.domElement);
                                
                                // Lights
                                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                                scene.add(ambientLight);
                                
                                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                                directionalLight.position.set(10, 20, 10);
                                directionalLight.castShadow = true;
                                scene.add(directionalLight);
                                
                                const fillLight = new THREE.DirectionalLight(0x667eea, 0.3);
                                fillLight.position.set(-10, 5, -10);
                                scene.add(fillLight);
                                
                                // Grid helper
                                const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
                                scene.add(gridHelper);
                                
                                // Controls
                                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                                controls.enableDamping = true;
                                controls.dampingFactor = 0.05;
                                controls.minDistance = 2;
                                controls.maxDistance = 50;
                                
                                // Load GLB
                                const loader = new THREE.GLTFLoader();
                                const glbData = '{glb_data}';
                                const binaryData = atob(glbData);
                                const bytes = new Uint8Array(binaryData.length);
                                for (let i = 0; i < binaryData.length; i++) {{
                                    bytes[i] = binaryData.charCodeAt(i);
                                }}
                                
                                loader.parse(bytes.buffer, '', function(gltf) {{
                                    const model = gltf.scene;
                                    
                                    // Center and scale the model
                                    const box = new THREE.Box3().setFromObject(model);
                                    const center = box.getCenter(new THREE.Vector3());
                                    const size = box.getSize(new THREE.Vector3());
                                    
                                    model.position.sub(center);
                                    const maxDim = Math.max(size.x, size.y, size.z);
                                    const scale = 5 / maxDim;
                                    model.scale.multiplyScalar(scale);
                                    
                                    scene.add(model);
                                    
                                    // Adjust camera
                                    camera.position.set(8, 6, 8);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                }}, function(error) {{
                                    console.error('Error loading GLB:', error);
                                }});
                                
                                // Animation loop
                                function animate() {{
                                    requestAnimationFrame(animate);
                                    controls.update();
                                    renderer.render(scene, camera);
                                }}
                                animate();
                                
                                // Handle resize
                                window.addEventListener('resize', function() {{
                                    const width = container.clientWidth;
                                    const height = container.clientHeight;
                                    camera.aspect = width / height;
                                    camera.updateProjectionMatrix();
                                    renderer.setSize(width, height);
                                }});
                            }})();
                        </script>
                    </div>
                    """
                    st.components.v1.html(threejs_viewer, height=370)
                    st.caption("🖱️ Drag to rotate • Scroll to zoom • Right-click to pan")
            
            with download_col:
                st.markdown('<div class="section-header">📥 Download Files</div>', unsafe_allow_html=True)
                
                if os.path.exists(result['files']['obj']):
                    with open(result['files']['obj'], 'rb') as f:
                        st.download_button(
                            "📦 OBJ File",
                            f.read(),
                            "floorplan_3d.obj",
                            "application/octet-stream",
                            use_container_width=True
                        )
                
                if os.path.exists(result['files']['glb']):
                    with open(result['files']['glb'], 'rb') as f:
                        st.download_button(
                            "📦 GLB File",
                            f.read(),
                            "floorplan_3d.glb",
                            "application/octet-stream",
                            use_container_width=True
                        )
                
                # ZIP download
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    output_dir = st.session_state.output_dir
                    for filename in os.listdir(output_dir):
                        filepath = os.path.join(output_dir, filename)
                        if os.path.isfile(filepath):
                            zf.write(filepath, filename)
                zip_buffer.seek(0)
                
                st.download_button(
                    "📁 Download All (ZIP)",
                    zip_buffer.getvalue(),
                    "floorplan_3d_complete.zip",
                    "application/zip",
                    use_container_width=True,
                    type="primary"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Reset
            if st.button("🔄 Convert Another", use_container_width=True):
                if st.session_state.output_dir and os.path.exists(st.session_state.output_dir):
                    shutil.rmtree(st.session_state.output_dir, ignore_errors=True)
                st.session_state.converted = False
                st.session_state.output_dir = None
                st.session_state.result = None
                st.rerun()
    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">⬅️</div>
            <h3>Upload a floor plan</h3>
            <p>Upload an image to start conversion</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    🏗️ AI-Powered Floor Plan to 3D Converter • Built with Streamlit
</div>
""", unsafe_allow_html=True)
