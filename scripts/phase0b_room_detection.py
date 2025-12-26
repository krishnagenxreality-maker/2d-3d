"""
PHASE 0B: ROOM DETECTION & FURNITURE PLACEMENT
===============================================
- Uses EasyOCR to detect ALL room labels
- Loads YOUR FBX/GLB/OBJ furniture models from furniture/ folder
- Places furniture in correct room locations
- Exports combined model with walls + furniture

FURNITURE FOLDER STRUCTURE:
furniture/
├── bed.fbx           (or .glb, .obj)
├── wardrobe.fbx
├── sofa.fbx
├── dining_table.fbx
├── toilet.fbx
├── sink.fbx
├── kitchen_counter.fbx
├── stove.fbx
└── desk.fbx
"""

import cv2
import numpy as np
from PIL import Image
import json
import os
import subprocess
import trimesh

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_IMAGE = "input/floorplan.png"
OUTPUT_DIR = "output/phase0_parsed"
FURNITURE_DIR = "furniture"

# Room type keywords - EXPANDED for better detection (including common OCR errors)
ROOM_KEYWORDS = {
    'bedroom': ['BED', 'BEDROOM', 'MASTER', 'GUEST', 'ROOM', 'HOOU', 'BEDR', 'BDRM', 'BR', 'IEDR', 'BEDRM'],
    'kitchen': ['KITCHEN', 'KITCHENETTE', 'COOKING', 'KITCH', 'KIT', 'KTCHN'],
    'bathroom': ['TOILET', 'BATH', 'BATHROOM', 'WC', 'RESTROOM', 'TOIL', 'WASHROOM', 'LATRINE', 'LAVATORY'],
    'living': ['LIVING', 'HALL', 'LOUNGE', 'DRAWING', 'HAL', 'SITTING', 'LIV', 'LVNG'],
    'dining': ['DINING', 'DINNER', 'DIN', 'DNNG'],
    'balcony': ['BALCONY', 'TERRACE', 'PATIO', 'BALC', 'VERANDAH', 'VERANDA'],
    'service': ['SERVICE', 'UTILITY', 'LAUNDRY', 'STORE', 'AREA', 'STORAGE', 'SERV'],
    'wardrobe': ['WARDROBE', 'CLOSET', 'DRESSING', 'ROBE', 'WARD', 'DRESS'],
    'pooja': ['POOJA', 'PUJA', 'PRAYER', 'TEMPLE'],
    'study': ['STUDY', 'OFFICE', 'WORK', 'STDY'],
}

# Furniture for each room type (matching your GLB file names)
ROOM_FURNITURE = {
    'bedroom': ['bed', 'wardrobe'],
    'kitchen': ['kitchen_counter', 'stove', 'sink'],
    'bathroom': ['toilet', 'sink'],
    'living': ['sofa', 'dining_table'],
    'dining': ['dining_table'],
    'balcony': [],
    'service': [],
    'wardrobe': ['wardrobe'],
    'pooja': [],
    'study': [],
}

# Default furniture sizes (used if GLB model not found)
FURNITURE_SIZES = {
    'bed': (2.0, 1.6, 0.5),
    'wardrobe': (1.5, 0.6, 2.0),
    'desk': (1.2, 0.6, 0.75),
    'kitchen_counter': (2.0, 0.6, 0.9),
    'stove': (0.6, 0.6, 0.9),
    'sink': (0.5, 0.5, 0.85),
    'toilet': (0.6, 0.7, 0.4),
    'shower': (1.0, 1.0, 2.1),
    'sofa': (2.0, 0.9, 0.85),
    'coffee_table': (1.0, 0.6, 0.4),
    'dining_table': (1.5, 0.9, 0.75),
}

# Furniture colors (for fallback boxes)
FURNITURE_COLORS = {
    'bed': [200, 180, 160, 255],
    'wardrobe': [139, 90, 43, 255],
    'desk': [205, 170, 125, 255],
    'kitchen_counter': [180, 180, 180, 255],
    'stove': [50, 50, 50, 255],
    'sink': [220, 220, 230, 255],
    'toilet': [255, 255, 255, 255],
    'shower': [200, 220, 240, 255],
    'sofa': [100, 80, 60, 255],
    'coffee_table': [139, 90, 43, 255],
    'dining_table': [180, 140, 100, 255],
}

PIXELS_PER_METER = 50
FURNITURE_SCALE = 0.002  # Scale factor for imported models (small to fit inside rooms)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 0B: ROOM DETECTION & FURNITURE PLACEMENT")
print("=" * 70)

# ============================================================
# STEP 1: LOAD IMAGE
# ============================================================

print(f"\n[1/6] Loading: {INPUT_IMAGE}")

if not os.path.exists(INPUT_IMAGE):
    print(f"ERROR: File not found!")
    exit(1)

img = cv2.imread(INPUT_IMAGE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

print(f"   Size: {width} x {height}")

# ============================================================
# STEP 2: CHECK FURNITURE FOLDER
# ============================================================

print(f"\n[2/6] Checking furniture folder: {FURNITURE_DIR}/")

available_furniture = {}
if os.path.exists(FURNITURE_DIR):
    for fname in os.listdir(FURNITURE_DIR):
        # Support FBX, GLB, and OBJ files
        if fname.lower().endswith(('.fbx', '.glb', '.obj', '.gltf')):
            name = os.path.splitext(fname)[0].lower()
            available_furniture[name] = os.path.join(FURNITURE_DIR, fname)
            print(f"   Found: {fname}")

if not available_furniture:
    print("   No furniture files found! Will use placeholder boxes.")
    print(f"   Place .fbx, .glb, or .obj files in: {FURNITURE_DIR}/")
else:
    print(f"   {len(available_furniture)} furniture models available")

# ============================================================
# STEP 2.5: CONVERT FBX TO GLB (if needed)
# ============================================================

GLB_CACHE_DIR = os.path.join(OUTPUT_DIR, "furniture_glb_cache")
os.makedirs(GLB_CACHE_DIR, exist_ok=True)

def find_blender():
    """Find Blender executable"""
    import glob
    paths = [
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender*\blender.exe",
    ]
    for pattern in paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def convert_fbx_to_glb(fbx_path, glb_path):
    """Convert FBX to GLB using Blender"""
    blender = find_blender()
    if not blender:
        return False
    
    script_path = os.path.join(os.path.dirname(__file__), "convert_fbx_to_glb.py")
    if not os.path.exists(script_path):
        return False
    
    try:
        result = subprocess.run(
            [blender, "--background", "--python", script_path, "--", fbx_path, glb_path],
            capture_output=True,
            timeout=300
        )
        return os.path.exists(glb_path)
    except Exception as e:
        print(f"      Conversion error: {e}")
        return False

# Convert FBX files to GLB
converted_furniture = {}
for name, path in available_furniture.items():
    if path.lower().endswith('.fbx'):
        glb_path = os.path.join(GLB_CACHE_DIR, f"{name}.glb")
        if not os.path.exists(glb_path):
            print(f"   Converting {name}.fbx to GLB...")
            if convert_fbx_to_glb(path, glb_path):
                print(f"   ✓ Converted: {name}.glb")
                converted_furniture[name] = glb_path
            else:
                print(f"   ✗ Failed to convert: {name}.fbx")
        else:
            print(f"   Using cached: {name}.glb")
            converted_furniture[name] = glb_path
    else:
        converted_furniture[name] = path

available_furniture = converted_furniture

# ============================================================
# STEP 3: OCR - DETECT ALL TEXT
# ============================================================

print("\n[3/6] Running OCR to detect room labels...")

try:
    import easyocr
    
    # Preprocess image for better OCR
    print("   Preprocessing image for better OCR...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Threshold to make text clearer
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to RGB for EasyOCR
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    # Run OCR on both original and processed images
    results_orig = reader.readtext(img_rgb)
    results_proc = reader.readtext(processed)
    
    # Combine results from both
    all_results = results_orig + results_proc
    
    detected_texts = []
    seen_positions = set()
    
    for (bbox, text, conf) in all_results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        
        center_x = int(sum(x_coords) / 4)
        center_y = int(sum(y_coords) / 4)
        
        # Skip duplicate positions
        pos_key = (center_x // 20, center_y // 20)
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)
        
        detected_texts.append({
            'text': text.upper().strip(),
            'confidence': round(conf * 100),
            'center': {'x': center_x, 'y': center_y}
        })
    
    print(f"   Detected {len(detected_texts)} text elements")
    for t in detected_texts:
        print(f"      '{t['text']}' (conf: {t['confidence']}%)")
    
except ImportError:
    print("   ERROR: easyocr not installed! Run: pip install easyocr")
    detected_texts = []
except Exception as e:
    print(f"   ERROR: OCR failed - {e}")
    detected_texts = []

# ============================================================
# STEP 4: IDENTIFY ALL ROOMS
# ============================================================

print("\n[4/6] Identifying ALL rooms from text...")

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def classify_room(text):
    """Classify text as a room type - with fuzzy matching"""
    text = text.upper().strip()
    
    # Skip very short or very long texts
    if len(text) < 2 or len(text) > 20:
        return None
    
    best_match = None
    best_score = float('inf')
    
    # Check each room type
    for room_type, keywords in ROOM_KEYWORDS.items():
        for keyword in keywords:
            # Exact substring match
            if keyword in text or text in keyword:
                return room_type
            
            # Check if text starts with keyword prefix
            if len(text) >= 3 and text[:3] == keyword[:3]:
                return room_type
            
            # Fuzzy match using Levenshtein distance
            # Allow up to 2 character differences for words >= 4 chars
            if len(keyword) >= 4 and len(text) >= 4:
                dist = levenshtein_distance(text, keyword)
                max_allowed = min(2, len(keyword) // 3)
                if dist <= max_allowed and dist < best_score:
                    best_score = dist
                    best_match = room_type
    
    return best_match

rooms = []
used_positions = []

for item in detected_texts:
    text = item['text']
    room_type = classify_room(text)
    
    if room_type:
        center_x = item['center']['x']
        center_y = item['center']['y']
        
        # Skip if too close to existing room (avoid duplicates)
        too_close = False
        for px, py in used_positions:
            if abs(center_x - px) < 50 and abs(center_y - py) < 50:
                too_close = True
                break
        
        if too_close:
            continue
        
        # Convert pixel position to meters (matching phase0 coordinate system)
        # phase0 uses: y = (height - pixel_y) / PPM (Y IS INVERTED)
        mx = center_x / PIXELS_PER_METER
        my = (height - center_y) / PIXELS_PER_METER
        
        # Clamp to building bounds to ensure furniture is inside
        mx = max(1.0, min(mx, 5.5))  # Keep away from outer walls
        my = max(1.0, min(my, 5.0))
        
        rooms.append({
            'type': room_type,
            'label': text,
            'position_px': {'x': center_x, 'y': center_y},
            'position_m': {'x': round(mx, 2), 'y': round(my, 2)},
            'furniture': ROOM_FURNITURE.get(room_type, [])
        })
        
        used_positions.append((center_x, center_y))

print(f"   Found {len(rooms)} rooms:")
for room in rooms:
    print(f"      - {room['type'].upper()}: '{room['label']}' at ({room['position_m']['x']}, {room['position_m']['y']})")

# ============================================================
# STEP 5: LOAD/CREATE FURNITURE MESHES
# ============================================================

print("\n[5/6] Creating furniture for each room...")

def load_furniture_model(name, position, scale=FURNITURE_SCALE, rotation_z=0):
    """Load GLB model or create fallback box"""
    x, y = position
    
    # Try to load GLB file
    if name in available_furniture:
        try:
            mesh = trimesh.load(available_furniture[name])
            
            # Handle scene vs mesh
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Convert from Y-up (OBJ standard) to Z-up (Blender/scene standard)
            # Rotate +90 degrees around X axis to flip correctly
            rotation_x = trimesh.transformations.rotation_matrix(
                np.radians(90), [1, 0, 0]
            )
            mesh.apply_transform(rotation_x)
            
            # Scale the model
            mesh.apply_scale(scale)
            
            # Get bounds after scaling
            bounds = mesh.bounds
            
            # Center XY at origin, place bottom on floor (z=0)
            center_x = (bounds[0][0] + bounds[1][0]) / 2
            center_y = (bounds[0][1] + bounds[1][1]) / 2
            min_z = bounds[0][2]  # Bottom of mesh
            
            mesh.apply_translation([-center_x, -center_y, -min_z])
            
            # Apply rotation around Z axis (in radians)
            if rotation_z != 0:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    np.radians(rotation_z), [0, 0, 1]
                )
                mesh.apply_transform(rotation_matrix)
            
            # Place at final position
            mesh.apply_translation([x, y, 0])
            
            # Debug: print mesh size
            size = bounds[1] - bounds[0]
            print(f"      Loaded: {name} at ({x:.1f}, {y:.1f}), size: {size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f}")
            return mesh
        except Exception as e:
            print(f"      Error loading {name}: {e}")
    
    # Fallback: create simple box
    if name in FURNITURE_SIZES:
        w, d, h = FURNITURE_SIZES[name]
        color = FURNITURE_COLORS.get(name, [150, 150, 150, 255])
        
        box = trimesh.creation.box([w, d, h])
        box.apply_translation([x, y, h/2])
        box.visual.face_colors = color
        
        print(f"      Created box: {name}")
        return box
    
    return None

furniture_meshes = []
furniture_data = []

for room in rooms:
    room_x = room['position_m']['x']
    room_y = room['position_m']['y']
    
    for i, furn_name in enumerate(room['furniture']):
        # DEBUG: Test placement - move bed further from walls
        # Building bounds: X(0.36-6.74), Y(0.12-6.12)
        if furn_name == 'bed':
            pos = (1.5, 2.0)  # Further from walls
            rotation = 0
        elif furn_name == 'wardrobe':
            pos = (2.5, 2.0)  # Offset from bed
            rotation = 0
        else:
            pos = (2.0, 2.0)
            rotation = 0
        
        mesh = load_furniture_model(furn_name, pos, rotation_z=rotation)
        
        if mesh is not None:
            furniture_meshes.append((f"{room['type']}_{furn_name}_{len(furniture_meshes)}", mesh))
            
            furniture_data.append({
                'name': furn_name,
                'room': room['type'],
                'position': {'x': round(pos[0], 2), 'y': round(pos[1], 2), 'z': 0}
            })

print(f"\n   Total furniture items: {len(furniture_meshes)}")

# ============================================================
# STEP 6: EXPORT EVERYTHING
# ============================================================

print("\n[6/6] Exporting results...")

# Save room detection data
room_data = {
    'rooms': rooms,
    'furniture': furniture_data,
    'image_info': {'width': width, 'height': height},
    'detected_texts': [{'text': t['text'], 'conf': t['confidence']} for t in detected_texts]
}

with open(f"{OUTPUT_DIR}/room_detection.json", 'w') as f:
    json.dump(room_data, f, indent=2)
print(f"   Saved: room_detection.json")

# Export furniture as separate GLB
if furniture_meshes:
    scene = trimesh.Scene()
    for name, mesh in furniture_meshes:
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    
    scene.export(f"{OUTPUT_DIR}/furniture.glb")
    scene.export(f"{OUTPUT_DIR}/furniture.obj")
    print(f"   Saved: furniture.glb + furniture.obj")

# Try to combine with existing walls
walls_model = f"{OUTPUT_DIR}/model.glb"
if os.path.exists(walls_model) and furniture_meshes:
    try:
        # Load walls
        walls_scene = trimesh.load(walls_model)
        
        # Add furniture to scene
        combined = trimesh.Scene()
        
        # Add walls
        for name, geom in walls_scene.geometry.items():
            combined.add_geometry(geom, node_name=name, geom_name=name)
        
        # Add furniture
        for name, mesh in furniture_meshes:
            combined.add_geometry(mesh, node_name=name, geom_name=name)
        
        combined.export(f"{OUTPUT_DIR}/complete_model.glb")
        combined.export(f"{OUTPUT_DIR}/complete_model.obj")
        print(f"   Saved: complete_model.glb (walls + furniture)")
    except Exception as e:
        print(f"   Could not combine: {e}")

# Create visualization
viz = img.copy()
colors_viz = {
    'bedroom': (0, 255, 0),
    'bathroom': (255, 0, 255),
    'kitchen': (0, 255, 255),
    'living': (255, 255, 0),
    'dining': (255, 128, 0),
    'balcony': (128, 128, 255),
    'service': (128, 128, 128),
    'wardrobe': (200, 100, 50),
}

for room in rooms:
    x = room['position_px']['x']
    y = room['position_px']['y']
    color = colors_viz.get(room['type'], (0, 255, 0))
    
    cv2.circle(viz, (x, y), 15, color, 3)
    cv2.putText(viz, room['type'].upper(), (x-40, y-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite(f"{OUTPUT_DIR}/room_detection_viz.png", viz)
print(f"   Saved: room_detection_viz.png")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 0B COMPLETE!")
print("=" * 70)
print(f"\n   Rooms detected: {len(rooms)}")
for room in rooms:
    print(f"      - {room['type']}: {len(room['furniture'])} furniture")
print(f"\n   Furniture items: {len(furniture_meshes)}")
print(f"\nOutput files:")
print(f"   - room_detection.json")
print(f"   - room_detection_viz.png")
print(f"   - furniture.glb / furniture.obj")
if os.path.exists(f"{OUTPUT_DIR}/complete_model.glb"):
    print(f"   - complete_model.glb (WALLS + FURNITURE combined!)")
print("=" * 70)
