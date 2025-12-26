"""
PHASE 0: WALL DETECTION WITH OPENINGS PRESERVED
================================================
Detects walls while PRESERVING door/window openings.
The gaps in wall lines represent doors and windows - we must NOT merge across them.

Key: Detect wall segments exactly as they appear, don't merge across gaps.
Wall thickness: 2cm (paper thin)
"""

import cv2
import numpy as np
import trimesh
from PIL import Image
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_IMAGE = "input/floorplan.png"
OUTPUT_DIR = "output/phase0_parsed"

# Wall dimensions (3D export)
WALL_HEIGHT = 1.5  # meters (reduced for better proportions)
WALL_THICKNESS = 0.02  # meters (2cm - paper thin!)
PIXELS_PER_METER = 50

# Detection parameters - PRESERVE OPENINGS
ANGLE_TOLERANCE = 8  # Degrees for H/V classification
MIN_WALL_LENGTH = 20  # Minimum segment length in pixels
MAX_MERGE_GAP = 5  # VERY SMALL gap to merge (preserve door openings which are larger)
MIN_DIAGONAL_LENGTH = 60  # Minimum length for diagonal walls (longer to filter text)
ENABLE_DIAGONAL_WALLS = True  # Enable detection of diagonal walls

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 0: WALL DETECTION WITH OPENINGS PRESERVED")
print("=" * 70)
print("Doors and windows (gaps in walls) will be preserved!")
print("=" * 70)

# ============================================================
# LOAD IMAGE
# ============================================================

print(f"\n[1/7] Loading: {INPUT_IMAGE}")

if not os.path.exists(INPUT_IMAGE):
    print(f"ERROR: File not found!")
    exit(1)

img = cv2.imread(INPUT_IMAGE)
if img is None:
    print("ERROR: Cannot read image!")
    exit(1)

height, width = img.shape[:2]
print(f"   Size: {width} x {height}")
cv2.imwrite(f"{OUTPUT_DIR}/01_original.png", img)

# ============================================================
# DETECT BLACK LINES (WALLS)
# ============================================================

print("\n[2/7] Detecting wall pixels...")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Simple threshold - walls are dark/black
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Detect and remove text regions before wall detection
print("   Removing text regions...")

# Find contours that look like text (small, clustered)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create mask for text regions to exclude
text_mask = np.zeros_like(binary)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect_ratio = max(w, h) / (min(w, h) + 1)
    
    # Text characteristics: small-medium size, roughly square or slightly elongated
    is_text = False
    
    # Small isolated characters
    if area < 200 and w < 30 and h < 30:
        is_text = True
    
    # Text regions - clustered small elements with aspect ratio
    if 5 < area < 500 and aspect_ratio < 5:
        is_text = True
    
    # Dimension text (numbers) - typically small
    if area < 100 and w < 20 and h < 20:
        is_text = True
    
    if is_text:
        # Expand the region slightly to cover the text
        pad = 3
        cv2.rectangle(text_mask, (max(0, x-pad), max(0, y-pad)), 
                     (min(width, x+w+pad), min(height, y+h+pad)), 255, -1)

# Remove text regions from binary image
binary_no_text = cv2.bitwise_and(binary, cv2.bitwise_not(text_mask))

# Minimal cleanup - don't over-process
kernel = np.ones((2, 2), np.uint8)
binary_no_text = cv2.morphologyEx(binary_no_text, cv2.MORPH_CLOSE, kernel)

cv2.imwrite(f"{OUTPUT_DIR}/02_binary.png", binary_no_text)
print("   Binary threshold done (text removed)")

# ============================================================
# DETECT LINE SEGMENTS - PRESERVING GAPS
# ============================================================

print("\n[3/7] Detecting line segments (preserving openings)...")

# Use Canny edge detection
edges = cv2.Canny(binary_no_text, 50, 150)
cv2.imwrite(f"{OUTPUT_DIR}/03_edges.png", edges)

# Detect lines with parameters that preserve gaps
# LOW max_gap ensures we don't connect across door openings
lines = cv2.HoughLinesP(
    edges, 
    rho=1, 
    theta=np.pi/180, 
    threshold=20,
    minLineLength=MIN_WALL_LENGTH,
    maxLineGap=MAX_MERGE_GAP  # Very small! Don't connect across openings
)

if lines is None:
    print("   Trying alternate detection...")
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, 15, MIN_WALL_LENGTH, MAX_MERGE_GAP)

if lines is None:
    print("ERROR: No lines detected!")
    exit(1)

raw_lines = [tuple(l[0]) for l in lines]
print(f"   Detected {len(raw_lines)} line segments")

# ============================================================
# FILTER TO HORIZONTAL/VERTICAL ONLY
# ============================================================

print("\n[4/7] Filtering to H/V lines only...")

def get_angle(x1, y1, x2, y2):
    """Angle in degrees (0=horizontal, 90=vertical)"""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx == 0:
        return 90
    return np.arctan(dy / dx) * 180 / np.pi

def line_length(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

h_lines = []  # Horizontal
v_lines = []  # Vertical
d_lines = []  # Diagonal walls

for x1, y1, x2, y2 in raw_lines:
    length = line_length(x1, y1, x2, y2)
    if length < MIN_WALL_LENGTH:
        continue
    
    angle = get_angle(x1, y1, x2, y2)
    
    if angle <= ANGLE_TOLERANCE:  # Horizontal
        avg_y = (y1 + y2) // 2
        h_lines.append((min(x1, x2), avg_y, max(x1, x2), avg_y))
    elif angle >= (90 - ANGLE_TOLERANCE):  # Vertical
        avg_x = (x1 + x2) // 2
        v_lines.append((avg_x, min(y1, y2), avg_x, max(y1, y2)))
    elif ENABLE_DIAGONAL_WALLS and length >= MIN_DIAGONAL_LENGTH:
        # Diagonal wall - only keep if long enough (not text)
        d_lines.append((x1, y1, x2, y2))

print(f"   Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}, Diagonal: {len(d_lines)}")

# ============================================================
# CAREFUL MERGING - ONLY MERGE OVERLAPPING, NOT ACROSS GAPS
# ============================================================

print("\n[5/7] Careful merging (preserving openings)...")

def merge_overlapping_only(lines, is_horizontal, tolerance=5, min_gap_for_opening=15):
    """
    Merge lines that OVERLAP, but preserve gaps that represent openings.
    min_gap_for_opening: If gap is larger than this, it's a door/window - don't merge
    """
    if not lines:
        return []
    
    # Group by position
    groups = defaultdict(list)
    for line in lines:
        if is_horizontal:
            key = line[1]  # Y coordinate
        else:
            key = line[0]  # X coordinate
        groups[key].append(line)
    
    # Merge groups that are at same position
    merged_groups = defaultdict(list)
    
    for pos, segs in groups.items():
        found_group = None
        for existing_pos in merged_groups.keys():
            if abs(pos - existing_pos) <= tolerance:
                found_group = existing_pos
                break
        
        if found_group is not None:
            merged_groups[found_group].extend(segs)
        else:
            merged_groups[pos] = segs
    
    # Now merge segments within each group - BUT preserve openings
    result = []
    
    for pos, segments in merged_groups.items():
        if is_horizontal:
            # Sort by X
            intervals = [(min(s[0], s[2]), max(s[0], s[2])) for s in segments]
        else:
            # Sort by Y
            intervals = [(min(s[1], s[3]), max(s[1], s[3])) for s in segments]
        
        intervals.sort()
        
        # Merge only if overlapping or very small gap
        merged_intervals = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            gap = start - current_end
            
            if gap <= min_gap_for_opening:
                # Small gap - merge
                current_end = max(current_end, end)
            else:
                # Large gap - this is a door/window opening! Keep separate
                if current_end - current_start >= MIN_WALL_LENGTH:
                    merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        
        if current_end - current_start >= MIN_WALL_LENGTH:
            merged_intervals.append((current_start, current_end))
        
        # Convert back to line format
        avg_pos = int(np.mean([pos for _ in merged_intervals]))
        for start, end in merged_intervals:
            if is_horizontal:
                result.append((int(start), avg_pos, int(end), avg_pos))
            else:
                result.append((avg_pos, int(start), avg_pos, int(end)))
    
    return result

h_merged = merge_overlapping_only(h_lines, is_horizontal=True)
v_merged = merge_overlapping_only(v_lines, is_horizontal=False)

print(f"   After merging: H={len(h_merged)}, V={len(v_merged)}")

# Combine all wall types
all_walls = [(x1, y1, x2, y2, 'H') for x1, y1, x2, y2 in h_merged]
all_walls.extend([(x1, y1, x2, y2, 'V') for x1, y1, x2, y2 in v_merged])
all_walls.extend([(x1, y1, x2, y2, 'D') for x1, y1, x2, y2 in d_lines])  # Diagonal walls

print(f"   TOTAL: {len(all_walls)} wall segments (H={len(h_merged)}, V={len(v_merged)}, D={len(d_lines)})")

# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\n[6/7] Removing duplicates...")

def remove_duplicates(walls, tolerance=8):
    unique = []
    for wall in walls:
        x1, y1, x2, y2, wtype = wall
        is_dup = False
        
        for ux1, uy1, ux2, uy2, utype in unique:
            if wtype != utype:
                continue
            
            if wtype == 'H':
                if abs(y1 - uy1) <= tolerance:
                    # Same row - check X overlap
                    if max(min(x1,x2), min(ux1,ux2)) < min(max(x1,x2), max(ux1,ux2)):
                        is_dup = True
                        break
            elif wtype == 'V':
                if abs(x1 - ux1) <= tolerance:
                    # Same column - check Y overlap
                    if max(min(y1,y2), min(uy1,uy2)) < min(max(y1,y2), max(uy1,uy2)):
                        is_dup = True
                        break
            else:  # Diagonal
                # Check if endpoints are close
                dist1 = np.sqrt((x1-ux1)**2 + (y1-uy1)**2)
                dist2 = np.sqrt((x2-ux2)**2 + (y2-uy2)**2)
                if (dist1 < tolerance and dist2 < tolerance) or (dist1 < tolerance and dist2 < tolerance):
                    is_dup = True
                    break
        
        if not is_dup:
            unique.append(wall)
    
    return unique

final_walls = remove_duplicates(all_walls)
h_count = sum(1 for w in final_walls if w[4] == 'H')
v_count = sum(1 for w in final_walls if w[4] == 'V')
d_count = sum(1 for w in final_walls if w[4] == 'D')

print(f"   Final: {len(final_walls)} walls ({h_count} H + {v_count} V + {d_count} D)")

# ============================================================
# VISUALIZE AND EXPORT
# ============================================================

print("\n[7/7] Creating outputs...")

# Visualization
viz = img.copy()

for x1, y1, x2, y2, wtype in final_walls:
    if wtype == 'H':
        color = (0, 255, 0)  # Green for horizontal
    elif wtype == 'V':
        color = (255, 0, 0)  # Blue for vertical
    else:
        color = (0, 255, 255)  # Yellow for diagonal
    cv2.line(viz, (x1, y1), (x2, y2), color, 2)
    cv2.circle(viz, (x1, y1), 3, (0, 0, 255), -1)
    cv2.circle(viz, (x2, y2), 3, (0, 0, 255), -1)

cv2.imwrite(f"{OUTPUT_DIR}/05_detected_walls.png", viz)

# Clean visualization
clean_viz = np.zeros((height, width, 3), dtype=np.uint8)
for x1, y1, x2, y2, wtype in final_walls:
    if wtype == 'H':
        color = (0, 255, 0)
    elif wtype == 'V':
        color = (255, 0, 0)
    else:
        color = (0, 255, 255)
    cv2.line(clean_viz, (x1, y1), (x2, y2), color, 2)
cv2.imwrite(f"{OUTPUT_DIR}/06_walls_only.png", clean_viz)

# Convert to 3D
def pixel_to_meters(px, py):
    return px / PIXELS_PER_METER, (height - py) / PIXELS_PER_METER

# MATERIALS / TEXTURES
TEXTURE_DIR = "textures"
WALL_TEXTURE = f"{TEXTURE_DIR}/wall_texture.png"
FLOOR_TEXTURE = f"{TEXTURE_DIR}/floor_texture.png"

# Fallback colors if textures not found
WALL_COLOR = [255, 240, 245, 255]   # Very light pink
FLOOR_COLOR = [255, 250, 240, 255]  # Cream-white marble

def load_texture(texture_path):
    """Load texture image and return as PIL Image"""
    if os.path.exists(texture_path):
        from PIL import Image as PILImage
        return PILImage.open(texture_path)
    return None

def create_uv_coords_wall(vertices, wall_height, wall_length):
    """Create UV coordinates for a wall mesh (box-shaped)"""
    tile_size = 1.0  # 1 meter per texture tile
    u_scale = wall_length / tile_size
    v_scale = wall_height / tile_size
    
    uv = np.array([
        [0, 0], [0, 0], [u_scale, 0], [u_scale, 0],
        [0, v_scale], [0, v_scale], [u_scale, v_scale], [u_scale, v_scale],
    ])
    return uv

def create_uv_coords_floor(floor_width, floor_depth):
    """Create UV coordinates for floor mesh"""
    tile_size = 1.0
    u_scale = floor_width / tile_size
    v_scale = floor_depth / tile_size
    
    uv = np.array([
        [0, 0], [u_scale, 0], [u_scale, v_scale], [0, v_scale],
        [0, 0], [u_scale, 0], [u_scale, v_scale], [0, v_scale],
    ])
    return uv

def apply_texture_to_mesh(mesh, texture_image, uv_coords):
    """Apply texture with UV mapping to a mesh - proper GLB compatible"""
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    
    # Create PBR material (works better with GLB/Blender)
    material = PBRMaterial(
        baseColorTexture=texture_image,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    
    # Create texture visual with proper UV mapping
    visual = TextureVisuals(uv=uv_coords, material=material)
    mesh.visual = visual
    
    return mesh

def apply_color_to_mesh(mesh, color_rgba):
    """Apply solid color to mesh - for fallback"""
    mesh.visual.face_colors = color_rgba
    return mesh

# Load textures
print(f"   Loading textures from {TEXTURE_DIR}/...")

wall_texture = None
floor_texture = None

if os.path.exists(WALL_TEXTURE):
    from PIL import Image as PILImage
    wall_texture = PILImage.open(WALL_TEXTURE).convert('RGBA')
    print(f"   ✓ Wall texture loaded: {WALL_TEXTURE} ({wall_texture.size})")
else:
    print(f"   ✗ Wall texture not found, using solid color")

if os.path.exists(FLOOR_TEXTURE):
    from PIL import Image as PILImage
    floor_texture = PILImage.open(FLOOR_TEXTURE).convert('RGBA')
    print(f"   ✓ Floor texture loaded: {FLOOR_TEXTURE} ({floor_texture.size})")
else:
    print(f"   ✗ Floor texture not found, using solid color")

walls_data = []
wall_meshes = []  # Keep walls separate

for i, (x1, y1, x2, y2, _) in enumerate(final_walls):
    mx1, my1 = pixel_to_meters(x1, y1)
    mx2, my2 = pixel_to_meters(x2, y2)
    
    walls_data.append({
        'start': [round(mx1, 2), round(my1, 2), 0],
        'end': [round(mx2, 2), round(my2, 2), 0],
        'thickness': WALL_THICKNESS
    })
    
    # 3D mesh
    direction = np.array([mx2 - mx1, my2 - my1, 0])
    length_m = np.linalg.norm(direction[:2])
    
    if length_m < 0.1:
        continue
    
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
    perp = np.array([-direction_norm[1], direction_norm[0], 0]) * WALL_THICKNESS / 2
    
    start = np.array([mx1, my1, 0])
    end = np.array([mx2, my2, 0])
    
    vertices = np.array([
        start + perp, start - perp, end - perp, end + perp,
        start + perp + [0,0,WALL_HEIGHT], start - perp + [0,0,WALL_HEIGHT],
        end - perp + [0,0,WALL_HEIGHT], end + perp + [0,0,WALL_HEIGHT],
    ])
    
    faces = np.array([
        [0,1,2], [0,2,3], [4,6,5], [4,7,6],
        [0,4,5], [0,5,1], [1,5,6], [1,6,2],
        [2,6,7], [2,7,3], [3,7,4], [3,4,0],
    ])
    
    wall_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Apply texture or color
    if wall_texture:
        uv = create_uv_coords_wall(vertices, WALL_HEIGHT, length_m)
        wall_mesh = apply_texture_to_mesh(wall_mesh, wall_texture, uv)
    else:
        wall_mesh.visual.face_colors = WALL_COLOR
    
    wall_meshes.append((f"Wall_{i+1}", wall_mesh))

# Create floor as SEPARATE component
floor_mesh = None
if len(wall_meshes) > 0:
    all_verts = np.vstack([m.vertices for _, m in wall_meshes])
    min_xy = all_verts[:, :2].min(axis=0)
    max_xy = all_verts[:, :2].max(axis=0)
    
    floor_width = max_xy[0] - min_xy[0] + 0.6
    floor_depth = max_xy[1] - min_xy[1] + 0.6
    
    floor_verts = np.array([
        [min_xy[0]-0.3, min_xy[1]-0.3, -0.1],
        [max_xy[0]+0.3, min_xy[1]-0.3, -0.1],
        [max_xy[0]+0.3, max_xy[1]+0.3, -0.1],
        [min_xy[0]-0.3, max_xy[1]+0.3, -0.1],
        [min_xy[0]-0.3, min_xy[1]-0.3, 0],
        [max_xy[0]+0.3, min_xy[1]-0.3, 0],
        [max_xy[0]+0.3, max_xy[1]+0.3, 0],
        [min_xy[0]-0.3, max_xy[1]+0.3, 0],
    ])
    floor_faces = np.array([
        [0,1,2], [0,2,3], [4,6,5], [4,7,6],
        [0,4,5], [0,5,1], [1,5,6], [1,6,2],
        [2,6,7], [2,7,3], [3,7,4], [3,4,0],
    ])
    
    floor_mesh = trimesh.Trimesh(vertices=floor_verts, faces=floor_faces)
    
    # Apply texture or color
    if floor_texture:
        uv = create_uv_coords_floor(floor_width, floor_depth)
        floor_mesh = apply_texture_to_mesh(floor_mesh, floor_texture, uv)
    else:
        floor_mesh.visual.face_colors = FLOOR_COLOR

# Export as SEPARATE COMPONENTS using Scene
print("   Creating scene with separate components...")

# Create scene with named objects
scene = trimesh.Scene()

# Center calculation
if len(wall_meshes) > 0:
    all_verts = np.vstack([m.vertices for _, m in wall_meshes])
    if floor_mesh is not None:
        all_verts = np.vstack([all_verts, floor_mesh.vertices])
    center = all_verts.mean(axis=0)
    
    # Add floor first (separate component)
    if floor_mesh is not None:
        floor_mesh.vertices -= center
        scene.add_geometry(floor_mesh, node_name="Floor", geom_name="Floor")
        print(f"   Floor component {'(TEXTURED)' if floor_texture else '(COLOR)'}")
    
    # Add each wall as separate component
    for name, wall_mesh in wall_meshes:
        wall_mesh.vertices -= center
        scene.add_geometry(wall_mesh, node_name=name, geom_name=name)
    
    print(f"   {len(wall_meshes)} wall components {'(TEXTURED)' if wall_texture else '(COLOR)'}")

# Export scene with separate objects
scene.export(f"{OUTPUT_DIR}/model.glb")  # GLB preserves materials and textures
scene.export(f"{OUTPUT_DIR}/model.obj")  # OBJ with MTL for materials

# Also export individual OBJ files for maximum compatibility
if floor_mesh is not None:
    floor_mesh.export(f"{OUTPUT_DIR}/floor.obj")
    print(f"   Exported: floor.obj")

# Export combined walls (but still separate from floor)
if len(wall_meshes) > 0:
    walls_combined = trimesh.util.concatenate([m for _, m in wall_meshes])
    if not wall_texture:
        walls_combined.visual.face_colors = WALL_COLOR
    walls_combined.export(f"{OUTPUT_DIR}/walls.obj")
    print(f"   Exported: walls.obj")

print("   3D models exported with textures/materials!")

# Save JSON
parsed_data = {
    'walls': walls_data,
    'image_info': {'width': width, 'height': height, 'pixels_per_meter': PIXELS_PER_METER},
    'stats': {
        'horizontal_walls': h_count,
        'vertical_walls': v_count,
        'total_walls': len(final_walls),
        'wall_thickness_cm': WALL_THICKNESS * 100
    }
}

with open(f"{OUTPUT_DIR}/parsed_geometry.json", 'w') as f:
    json.dump(parsed_data, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 0 COMPLETE - OPENINGS PRESERVED!")
print("=" * 70)
print(f"\n   Horizontal walls: {h_count}")
print(f"   Vertical walls:   {v_count}")
print(f"   TOTAL segments:   {len(final_walls)}")
print(f"   Wall thickness:   {WALL_THICKNESS*100}cm (paper thin)")
print(f"\n   Door/window openings should now be VISIBLE as gaps!")
print(f"\nCheck: {OUTPUT_DIR}/05_detected_walls.png")
print("=" * 70)