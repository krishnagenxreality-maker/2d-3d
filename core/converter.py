"""
Core Converter Module
=====================
Refactored pipeline using Contour-based wall detection, OCR text removal,
and structural enhancements to correctly reconstruct robust 3D models.
"""

import cv2
import numpy as np
import trimesh
from trimesh.creation import extrude_polygon
from PIL import Image
import os
import io
import warnings
import json

try:
    from shapely.geometry import Polygon, MultiPolygon
except ImportError:
    Polygon = None
    MultiPolygon = None

try:
    import easyocr
except ImportError:
    easyocr = None

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

WALL_HEIGHT = 2.5  # meters
PIXELS_PER_METER = 50

# Default standard colors (Fallback)
WALL_COLOR = [220, 220, 220, 255]
FLOOR_COLOR = [240, 240, 240, 255]

# Filtering Parameters
MIN_WALL_AREA = 500
MIN_WALL_THICKNESS = 4

def convert_floorplan_to_3d(image_data, output_dir, texture_dir="textures", progress_callback=None):
    """
    Convert a floor plan image to robust 3D model using contour detection and morphology.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def update_progress(step, total, message):
        if progress_callback:
            progress_callback(step, total, message)
            
    # ---------------------------------------------------------
    # 1. Load Image and Preprocess
    # ---------------------------------------------------------
    update_progress(1, 10, "Loading and Preprocessing Image...")
    
    if isinstance(image_data, bytes):
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_data, str):
        img = cv2.imread(image_data)
    elif isinstance(image_data, np.ndarray):
        img = image_data
    else:
        raise ValueError("Invalid image_data format")
        
    if img is None:
        raise ValueError("Could not load the image.")

    height, width = img.shape[:2]
    cv2.imwrite(f"{output_dir}/01_original.png", img)
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding for varying lighting, giving a clean binary image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(f"{output_dir}/02_adaptive_thresh.png", binary)
    
    # ---------------------------------------------------------
    # 2. OCR Text Removal
    # ---------------------------------------------------------
    update_progress(2, 10, "Removing Text with OCR...")
    binary_no_text = binary.copy()
    
    if easyocr is not None:
        try:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            results = reader.readtext(img)
            
            for (bbox, text, prob) in results:
                pts = np.array(bbox, np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                pad = 4
                cv2.rectangle(binary_no_text, 
                              (max(0, x-pad), max(0, y-pad)), 
                              (min(width, x+w+pad), min(height, y+h+pad)), 
                              0, -1)
        except Exception as e:
            print(f"OCR Text removal skipped due to error: {e}")
            
    cv2.imwrite(f"{output_dir}/03_no_text.png", binary_no_text)

    # ---------------------------------------------------------
    # 3. Structure Enhancement (Morphological Ops)
    # ---------------------------------------------------------
    update_progress(3, 10, "Enhancing Structures & Removing Noise...")
    
    # Step A: Morphological Opening
    kernel_open = np.ones((4, 4), np.uint8)
    opened = cv2.morphologyEx(binary_no_text, cv2.MORPH_OPEN, kernel_open)
    cv2.imwrite(f"{output_dir}/04_opened_noise_removed.png", opened)
    
    # Step B: Morphological Closing
    kernel_close = np.ones((13, 13), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    cv2.imwrite(f"{output_dir}/04_closed_walls.png", closed)

    # ---------------------------------------------------------
    # 4. Contour-based Wall Detection
    # ---------------------------------------------------------
    update_progress(4, 10, "Detecting Wall Contours...")
    
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return {'success': False, 'error': "No walls detected."}
        
    hierarchy = hierarchy[0]
    valid_polygons = []
    
    update_progress(5, 10, "Filtering Walls and Enhancing Geometry...")
    viz_filtered = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i, cnt in enumerate(contours):
        parent_idx = hierarchy[i][3]
        if parent_idx != -1:
            continue
            
        area = cv2.contourArea(cnt)
        if area < MIN_WALL_AREA:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        thickness_approx = 2 * area / perimeter
        if thickness_approx < MIN_WALL_THICKNESS:
            continue
            
        epsilon = 0.005 * perimeter
        approx_shell = cv2.approxPolyDP(cnt, epsilon, True)
        
        holes = []
        child_idx = hierarchy[i][2]
        while child_idx != -1:
            child_cnt = contours[child_idx]
            child_area = cv2.contourArea(child_cnt)
            
            if child_area > 100:
                child_epsilon = 0.005 * cv2.arcLength(child_cnt, True)
                approx_child = cv2.approxPolyDP(child_cnt, child_epsilon, True)
                holes.append(approx_child)
                
            child_idx = hierarchy[child_idx][0]
            
        valid_polygons.append((approx_shell, holes))
        
        cv2.drawContours(viz_filtered, [approx_shell], -1, (200, 200, 200), cv2.FILLED)
        for hole in holes:
            cv2.drawContours(viz_filtered, [hole], -1, (0, 0, 0), cv2.FILLED)
            
    cv2.imwrite(f"{output_dir}/05_filtered_walls.png", viz_filtered)

    # ---------------------------------------------------------
    # 5. Extract 2D Shapely Polygons
    # ---------------------------------------------------------
    update_progress(6, 10, "Extracting and Building 2D Geometry...")
    
    if Polygon is None:
        raise ImportError("Shapely is required. pip install shapely")
        
    shapely_polys = []
    for (shell_px, holes_px) in valid_polygons:
        shell_m = [(p[0][0] / PIXELS_PER_METER, (height - p[0][1]) / PIXELS_PER_METER) for p in shell_px]
        
        if len(shell_m) < 3:
            continue
            
        holes_m = []
        for h_px in holes_px:
            h_m = [(p[0][0] / PIXELS_PER_METER, (height - p[0][1]) / PIXELS_PER_METER) for p in h_px]
            if len(h_m) >= 3:
                holes_m.append(h_m)
                
        poly = Polygon(shell=shell_m, holes=holes_m)
        
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # buffer(0) may return a MultiPolygon — decompose into individual Polygons
        if MultiPolygon is not None and isinstance(poly, MultiPolygon):
            geoms = list(poly.geoms)
        else:
            geoms = [poly]
            
        for g in geoms:
            if g.is_empty or g.area < 0.05:
                continue
            shapely_polys.append(g)
        
    # ---------------------------------------------------------
    # 6. Generate 3D Extrusions
    # ---------------------------------------------------------
    update_progress(7, 10, "Generating 3D Extrusions for Walls...")
    
    wall_meshes = []
    for i, poly in enumerate(shapely_polys):
        # Ensure we only ever extrude a simple Polygon (no MultiPolygon)
        if MultiPolygon is not None and isinstance(poly, MultiPolygon):
            sub_polys = list(poly.geoms)
        else:
            sub_polys = [poly]
            
        for sub_poly in sub_polys:
            if sub_poly.is_empty or sub_poly.area < 0.05:
                continue
            try:
                mesh = extrude_polygon(sub_poly, height=WALL_HEIGHT)
                mesh.visual.face_colors = WALL_COLOR
                wall_meshes.append(mesh)
            except Exception as e:
                print(f"Warning: Failed to extrude sub-polygon from group {i}: {e}")

    # ---------------------------------------------------------
    # 7. Generate Floor Mesh
    # ---------------------------------------------------------
    update_progress(8, 10, "Generating Floor Mesh...")
    
    floor_mesh = None
    if wall_meshes:
        all_verts = np.vstack([m.vertices for m in wall_meshes])
        min_xy = all_verts[:, :2].min(axis=0)
        max_xy = all_verts[:, :2].max(axis=0)
        
        pad = 0.5 
        floor_poly = Polygon([
            (min_xy[0]-pad, min_xy[1]-pad),
            (max_xy[0]+pad, min_xy[1]-pad),
            (max_xy[0]+pad, max_xy[1]+pad),
            (min_xy[0]-pad, max_xy[1]+pad)
        ])
        
        floor_mesh = extrude_polygon(floor_poly, height=0.1)
        floor_mesh.vertices[:, 2] -= 0.1
        floor_mesh.visual.face_colors = FLOOR_COLOR

    # ---------------------------------------------------------
    # 8. Assemble Scene
    # ---------------------------------------------------------
    update_progress(9, 10, "Centering and Assembling Scene...")
    
    scene = trimesh.Scene()
    
    if wall_meshes:
        center = all_verts.mean(axis=0)
        center[2] = 0
        
        if floor_mesh is not None:
            floor_mesh.vertices -= center
            scene.add_geometry(floor_mesh, node_name="Floor")
            
        for i, m in enumerate(wall_meshes):
            m.vertices -= center
            scene.add_geometry(m, node_name=f"Wall_{i}")

    # ---------------------------------------------------------
    # 9. Export GLB & OBJ
    # ---------------------------------------------------------
    update_progress(10, 10, "Exporting 3D Models...")
    
    glb_path = f"{output_dir}/model.glb"
    obj_path = f"{output_dir}/model.obj"
    
    scene.export(glb_path)
    scene.export(obj_path)
    
    if wall_meshes:
        combined_walls = trimesh.util.concatenate(wall_meshes)
        combined_walls.export(f"{output_dir}/walls.obj")
        
    if floor_mesh is not None:
        floor_mesh.export(f"{output_dir}/floor.obj")
        
    stats = {
        'total_polygons': len(shapely_polys),
        'image_size': f"{width}x{height}",
        'robust_pipeline': True
    }
    
    with open(f"{output_dir}/parsed_geometry.json", 'w') as f:
        json.dump(stats, f, indent=2)

    return {
        'success': True,
        'output_dir': output_dir,
        'files': {
            'glb': glb_path,
            'obj': obj_path,
            'floor_obj': f"{output_dir}/floor.obj" if floor_mesh else None,
            'walls_obj': f"{output_dir}/walls.obj" if wall_meshes else None,
            'preview': f"{output_dir}/05_filtered_walls.png"
        },
        'stats': stats
    }
