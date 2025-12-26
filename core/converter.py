"""
Core Converter Module
=====================
Refactored wall detection and 3D model generation logic from phase0_image_to_walls.py.
Provides a clean API for the Streamlit app to convert floor plan images to 3D models.
"""

import cv2
import numpy as np
import trimesh
from PIL import Image
import os
from collections import defaultdict
import warnings
import tempfile
import io

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

WALL_HEIGHT = 1.5  # meters
WALL_THICKNESS = 0.02  # meters (2cm) for 3D model
PIXELS_PER_METER = 50

# Detection parameters
ANGLE_TOLERANCE = 8
MIN_WALL_LENGTH = 20
MAX_MERGE_GAP = 5
MIN_DIAGONAL_LENGTH = 100
ENABLE_DIAGONAL_WALLS = True

# Minimum thickness to be considered a wall (filters out arrows, dimension lines)
MIN_WALL_THICKNESS_PIXELS = 4  # Lowered to detect thinner internal walls

# Colors (fallback if textures not found)
WALL_COLOR = [255, 240, 245, 255]
FLOOR_COLOR = [255, 250, 240, 255]


def get_angle(x1, y1, x2, y2):
    """Angle in degrees (0=horizontal, 90=vertical)"""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx == 0:
        return 90
    return np.arctan(dy / dx) * 180 / np.pi


def line_length(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def measure_wall_thickness(binary_img, x1, y1, x2, y2, wall_type):
    """
    Measure the thickness of a wall by sampling perpendicular to the wall direction.
    Returns the average thickness in pixels.
    """
    height, width = binary_img.shape[:2]
    
    if wall_type == 'H':  # Horizontal wall - measure thickness vertically
        # Sample at multiple points along the wall
        num_samples = max(3, min(10, abs(x2 - x1) // 20))
        thicknesses = []
        
        for i in range(num_samples):
            x = x1 + (x2 - x1) * i // (num_samples - 1) if num_samples > 1 else (x1 + x2) // 2
            x = max(0, min(width - 1, x))
            
            # Measure upward and downward from the detected line position
            thickness = 0
            # Go up
            for dy in range(50):
                check_y = y1 - dy
                if check_y < 0 or binary_img[check_y, x] == 0:
                    break
                thickness += 1
            # Go down
            for dy in range(1, 50):
                check_y = y1 + dy
                if check_y >= height or binary_img[check_y, x] == 0:
                    break
                thickness += 1
            
            if thickness > 0:
                thicknesses.append(thickness)
        
        return int(np.median(thicknesses)) if thicknesses else 2
    
    elif wall_type == 'V':  # Vertical wall - measure thickness horizontally
        num_samples = max(3, min(10, abs(y2 - y1) // 20))
        thicknesses = []
        
        for i in range(num_samples):
            y = y1 + (y2 - y1) * i // (num_samples - 1) if num_samples > 1 else (y1 + y2) // 2
            y = max(0, min(height - 1, y))
            
            # Measure left and right from the detected line position
            thickness = 0
            # Go left
            for dx in range(50):
                check_x = x1 - dx
                if check_x < 0 or binary_img[y, check_x] == 0:
                    break
                thickness += 1
            # Go right
            for dx in range(1, 50):
                check_x = x1 + dx
                if check_x >= width or binary_img[y, check_x] == 0:
                    break
                thickness += 1
            
            if thickness > 0:
                thicknesses.append(thickness)
        
        return int(np.median(thicknesses)) if thicknesses else 2
    
    else:  # Diagonal - use perpendicular measurement
        return 8  # Default for diagonal walls


def merge_parallel_to_centerline(lines, binary_img, is_horizontal, thickness_tolerance=20):
    """
    Merge parallel lines (edges of same thick wall) into single centerline.
    Also measures and returns the wall thickness.
    Returns list of (x1, y1, x2, y2, thickness) tuples.
    """
    if not lines:
        return []
    
    height, width = binary_img.shape[:2]
    
    # Sort by position
    if is_horizontal:
        sorted_lines = sorted(lines, key=lambda l: l[1])
    else:
        sorted_lines = sorted(lines, key=lambda l: l[0])
    
    result = []
    used = set()
    
    for i, line1 in enumerate(sorted_lines):
        if i in used:
            continue
        
        x1_1, y1_1, x2_1, y2_1 = line1
        parallel_group = [(line1, i)]
        
        for j, line2 in enumerate(sorted_lines[i+1:], start=i+1):
            if j in used:
                continue
            
            x1_2, y1_2, x2_2, y2_2 = line2
            
            if is_horizontal:
                pos_diff = abs(y1_1 - y1_2)
                # Check if X ranges overlap
                overlap_start = max(min(x1_1, x2_1), min(x1_2, x2_2))
                overlap_end = min(max(x1_1, x2_1), max(x1_2, x2_2))
                has_overlap = overlap_end > overlap_start
            else:
                pos_diff = abs(x1_1 - x1_2)
                # Check if Y ranges overlap
                overlap_start = max(min(y1_1, y2_1), min(y1_2, y2_2))
                overlap_end = min(max(y1_1, y2_1), max(y1_2, y2_2))
                has_overlap = overlap_end > overlap_start
            
            # Merge if within thickness tolerance and overlapping
            if pos_diff <= thickness_tolerance and has_overlap:
                parallel_group.append((line2, j))
                used.add(j)
        
        used.add(i)
        
        # Calculate centerline and thickness for the group
        if len(parallel_group) > 1:
            if is_horizontal:
                # Get min and max Y positions (the two edges)
                y_positions = [l[0][1] for l in parallel_group]
                min_y, max_y = min(y_positions), max(y_positions)
                center_y = (min_y + max_y) // 2
                thickness = max_y - min_y
                # Union of X ranges
                min_x = min(min(l[0][0], l[0][2]) for l in parallel_group)
                max_x = max(max(l[0][0], l[0][2]) for l in parallel_group)
                result.append((min_x, center_y, max_x, center_y, max(thickness, 2)))
            else:
                # Get min and max X positions (the two edges)
                x_positions = [l[0][0] for l in parallel_group]
                min_x, max_x = min(x_positions), max(x_positions)
                center_x = (min_x + max_x) // 2
                thickness = max_x - min_x
                # Union of Y ranges
                min_y = min(min(l[0][1], l[0][3]) for l in parallel_group)
                max_y = max(max(l[0][1], l[0][3]) for l in parallel_group)
                result.append((center_x, min_y, center_x, max_y, max(thickness, 2)))
        else:
            # Single line - measure thickness from binary image
            x1, y1, x2, y2 = line1
            wall_type = 'H' if is_horizontal else 'V'
            thickness = measure_wall_thickness(binary_img, x1, y1, x2, y2, wall_type)
            result.append((x1, y1, x2, y2, thickness))
    
    return result


def merge_overlapping_only(lines, is_horizontal, tolerance=5, min_gap_for_opening=15):
    """Merge lines that OVERLAP, but preserve gaps that represent openings."""
    if not lines:
        return []
    
    groups = defaultdict(list)
    for line in lines:
        if is_horizontal:
            key = line[1]
        else:
            key = line[0]
        groups[key].append(line)
    
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
    
    result = []
    
    for pos, segments in merged_groups.items():
        if is_horizontal:
            intervals = [(min(s[0], s[2]), max(s[0], s[2])) for s in segments]
        else:
            intervals = [(min(s[1], s[3]), max(s[1], s[3])) for s in segments]
        
        intervals.sort()
        
        merged_intervals = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            gap = start - current_end
            
            if gap <= min_gap_for_opening:
                current_end = max(current_end, end)
            else:
                if current_end - current_start >= MIN_WALL_LENGTH:
                    merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        
        if current_end - current_start >= MIN_WALL_LENGTH:
            merged_intervals.append((current_start, current_end))
        
        avg_pos = int(np.mean([pos for _ in merged_intervals]))
        for start, end in merged_intervals:
            if is_horizontal:
                result.append((int(start), avg_pos, int(end), avg_pos))
            else:
                result.append((avg_pos, int(start), avg_pos, int(end)))
    
    return result


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
                    if max(min(x1,x2), min(ux1,ux2)) < min(max(x1,x2), max(ux1,ux2)):
                        is_dup = True
                        break
            elif wtype == 'V':
                if abs(x1 - ux1) <= tolerance:
                    if max(min(y1,y2), min(uy1,uy2)) < min(max(y1,y2), max(uy1,uy2)):
                        is_dup = True
                        break
            else:
                dist1 = np.sqrt((x1-ux1)**2 + (y1-uy1)**2)
                dist2 = np.sqrt((x2-ux2)**2 + (y2-uy2)**2)
                if (dist1 < tolerance and dist2 < tolerance):
                    is_dup = True
                    break
        
        if not is_dup:
            unique.append(wall)
    
    return unique


def create_uv_coords_wall(vertices, wall_height, wall_length):
    """Create UV coordinates for a wall mesh"""
    tile_size = 1.0
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
    """Apply texture with UV mapping to a mesh"""
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    
    material = PBRMaterial(
        baseColorTexture=texture_image,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    
    visual = TextureVisuals(uv=uv_coords, material=material)
    mesh.visual = visual
    
    return mesh


def convert_floorplan_to_3d(image_data, output_dir, texture_dir="textures", progress_callback=None):
    """
    Convert a floor plan image to 3D model.
    
    Args:
        image_data: Either bytes, file path, or numpy array of the image
        output_dir: Directory to save output files
        texture_dir: Directory containing wall_texture.png and floor_texture.png
        progress_callback: Optional function(step, total, message) for progress updates
    
    Returns:
        dict with paths to generated files and stats
    """
    
    def update_progress(step, total, message):
        if progress_callback:
            progress_callback(step, total, message)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load image
    update_progress(1, 7, "Loading image...")
    
    if isinstance(image_data, bytes):
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_data, str):
        img = cv2.imread(image_data)
    elif isinstance(image_data, np.ndarray):
        img = image_data
    else:
        raise ValueError("image_data must be bytes, file path, or numpy array")
    
    if img is None:
        raise ValueError("Could not load image")
    
    height, width = img.shape[:2]
    cv2.imwrite(f"{output_dir}/01_original.png", img)
    
    # Step 2: Detect black lines (walls)
    update_progress(2, 7, "Detecting wall pixels...")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Remove text regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_mask = np.zeros_like(binary)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1)
        
        is_text = False
        if area < 200 and w < 30 and h < 30:
            is_text = True
        if 5 < area < 500 and aspect_ratio < 5:
            is_text = True
        if area < 100 and w < 20 and h < 20:
            is_text = True
        
        if is_text:
            pad = 3
            cv2.rectangle(text_mask, (max(0, x-pad), max(0, y-pad)), 
                         (min(width, x+w+pad), min(height, y+h+pad)), 255, -1)
    
    binary_no_text = cv2.bitwise_and(binary, cv2.bitwise_not(text_mask))
    kernel = np.ones((2, 2), np.uint8)
    binary_no_text = cv2.morphologyEx(binary_no_text, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{output_dir}/02_binary.png", binary_no_text)
    
    # Step 3: Detect line segments
    update_progress(3, 7, "Detecting line segments...")
    
    edges = cv2.Canny(binary_no_text, 50, 150)
    cv2.imwrite(f"{output_dir}/03_edges.png", edges)
    
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=20,
        minLineLength=MIN_WALL_LENGTH, maxLineGap=MAX_MERGE_GAP
    )
    
    if lines is None:
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, 15, MIN_WALL_LENGTH, MAX_MERGE_GAP)
    
    if lines is None:
        raise ValueError("No lines detected in image")
    
    raw_lines = [tuple(l[0]) for l in lines]
    
    # Step 4: Filter to H/V lines
    update_progress(4, 7, "Classifying wall segments...")
    
    h_lines = []
    v_lines = []
    d_lines = []
    
    for x1, y1, x2, y2 in raw_lines:
        length = line_length(x1, y1, x2, y2)
        if length < MIN_WALL_LENGTH:
            continue
        
        angle = get_angle(x1, y1, x2, y2)
        
        if angle <= ANGLE_TOLERANCE:
            avg_y = (y1 + y2) // 2
            h_lines.append((min(x1, x2), avg_y, max(x1, x2), avg_y))
        elif angle >= (90 - ANGLE_TOLERANCE):
            avg_x = (x1 + x2) // 2
            v_lines.append((avg_x, min(y1, y2), avg_x, max(y1, y2)))
        elif ENABLE_DIAGONAL_WALLS and length >= MIN_DIAGONAL_LENGTH:
            d_lines.append((x1, y1, x2, y2))
    
    # Step 5: Merge to centerlines with thickness
    update_progress(5, 7, "Detecting wall thickness and centerlines...")
    
    # Merge parallel edge lines to centerlines and measure thickness
    h_with_thickness = merge_parallel_to_centerline(h_lines, binary_no_text, is_horizontal=True)
    v_with_thickness = merge_parallel_to_centerline(v_lines, binary_no_text, is_horizontal=False)
    
    # Merge overlapping segments (preserving thickness)
    def merge_overlapping_with_thickness(lines_with_thick, is_horizontal, tolerance=5, min_gap=15):
        """Merge overlapping segments while preserving thickness info."""
        if not lines_with_thick:
            return []
        
        # Extract lines without thickness for grouping
        lines_only = [(x1, y1, x2, y2) for x1, y1, x2, y2, t in lines_with_thick]
        thickness_map = {(x1, y1, x2, y2): t for x1, y1, x2, y2, t in lines_with_thick}
        
        groups = defaultdict(list)
        for line in lines_only:
            if is_horizontal:
                key = line[1]
            else:
                key = line[0]
            groups[key].append(line)
        
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
        
        result = []
        for pos, segments in merged_groups.items():
            if is_horizontal:
                intervals = [(min(s[0], s[2]), max(s[0], s[2]), s) for s in segments]
            else:
                intervals = [(min(s[1], s[3]), max(s[1], s[3]), s) for s in segments]
            
            intervals.sort()
            
            merged_intervals = []
            current_start, current_end, current_seg = intervals[0]
            current_thickness = thickness_map.get(current_seg, 2)
            
            for start, end, seg in intervals[1:]:
                gap = start - current_end
                seg_thickness = thickness_map.get(seg, 2)
                
                if gap <= min_gap:
                    current_end = max(current_end, end)
                    current_thickness = max(current_thickness, seg_thickness)
                else:
                    if current_end - current_start >= MIN_WALL_LENGTH:
                        merged_intervals.append((current_start, current_end, current_thickness))
                    current_start, current_end = start, end
                    current_thickness = seg_thickness
            
            if current_end - current_start >= MIN_WALL_LENGTH:
                merged_intervals.append((current_start, current_end, current_thickness))
            
            avg_pos = int(np.mean([pos for _ in merged_intervals]))
            for start, end, thick in merged_intervals:
                if is_horizontal:
                    result.append((int(start), avg_pos, int(end), avg_pos, thick))
                else:
                    result.append((avg_pos, int(start), avg_pos, int(end), thick))
        
        return result
    
    h_merged = merge_overlapping_with_thickness(h_with_thickness, is_horizontal=True, min_gap=25)
    v_merged = merge_overlapping_with_thickness(v_with_thickness, is_horizontal=False, min_gap=25)
    
    # Add type info and diagonal walls
    all_walls = [(x1, y1, x2, y2, 'H', thick) for x1, y1, x2, y2, thick in h_merged]
    all_walls.extend([(x1, y1, x2, y2, 'V', thick) for x1, y1, x2, y2, thick in v_merged])
    
    # Measure thickness for diagonal walls
    for x1, y1, x2, y2 in d_lines:
        thick = measure_wall_thickness(binary_no_text, x1, y1, x2, y2, 'D')
        all_walls.append((x1, y1, x2, y2, 'D', thick))
    
    # Filter out thin lines (arrows, dimension marks)
    all_walls = [(x1, y1, x2, y2, wtype, thick) for x1, y1, x2, y2, wtype, thick in all_walls 
                 if thick >= MIN_WALL_THICKNESS_PIXELS]
    
    # Step 6: Remove duplicates (updated for new format)
    update_progress(6, 7, "Removing duplicates...")
    
    def remove_duplicates_with_thickness(walls, tolerance=8):
        unique = []
        for wall in walls:
            x1, y1, x2, y2, wtype, thick = wall
            is_dup = False
            
            for idx, (ux1, uy1, ux2, uy2, utype, uthick) in enumerate(unique):
                if wtype != utype:
                    continue
                
                if wtype == 'H':
                    if abs(y1 - uy1) <= tolerance:
                        if max(min(x1,x2), min(ux1,ux2)) < min(max(x1,x2), max(ux1,ux2)):
                            # Keep the one with larger thickness
                            if thick > uthick:
                                unique[idx] = wall
                            is_dup = True
                            break
                elif wtype == 'V':
                    if abs(x1 - ux1) <= tolerance:
                        if max(min(y1,y2), min(uy1,uy2)) < min(max(y1,y2), max(uy1,uy2)):
                            if thick > uthick:
                                unique[idx] = wall
                            is_dup = True
                            break
                else:
                    dist1 = np.sqrt((x1-ux1)**2 + (y1-uy1)**2)
                    dist2 = np.sqrt((x2-ux2)**2 + (y2-uy2)**2)
                    if (dist1 < tolerance and dist2 < tolerance):
                        if thick > uthick:
                            unique[idx] = wall
                        is_dup = True
                        break
            
            if not is_dup:
                unique.append(wall)
        
        return unique
    
    final_walls = remove_duplicates_with_thickness(all_walls)
    h_count = sum(1 for w in final_walls if w[4] == 'H')
    v_count = sum(1 for w in final_walls if w[4] == 'V')
    d_count = sum(1 for w in final_walls if w[4] == 'D')
    
    # Step 7: Create 3D model
    update_progress(7, 7, "Generating 3D model...")
    
    # Visualization with thickness
    viz = img.copy()
    for x1, y1, x2, y2, wtype, thickness in final_walls:
        if wtype == 'H':
            color = (0, 255, 0)  # Green for horizontal
        elif wtype == 'V':
            color = (255, 0, 0)  # Blue for vertical
        else:
            color = (0, 255, 255)  # Yellow for diagonal
        
        # Draw line with measured thickness
        line_thickness = max(2, min(thickness, 30))  # Clamp between 2 and 30 pixels
        cv2.line(viz, (x1, y1), (x2, y2), color, line_thickness)
    
    cv2.imwrite(f"{output_dir}/05_detected_walls.png", viz)
    
    # Load textures
    wall_texture = None
    floor_texture = None
    
    wall_texture_path = os.path.join(texture_dir, "wall_texture.png")
    floor_texture_path = os.path.join(texture_dir, "floor_texture.png")
    
    if os.path.exists(wall_texture_path):
        wall_texture = Image.open(wall_texture_path).convert('RGBA')
    
    if os.path.exists(floor_texture_path):
        floor_texture = Image.open(floor_texture_path).convert('RGBA')
    
    def pixel_to_meters(px, py):
        return px / PIXELS_PER_METER, (height - py) / PIXELS_PER_METER
    
    wall_meshes = []
    
    for i, (x1, y1, x2, y2, wtype, thickness_px) in enumerate(final_walls):
        mx1, my1 = pixel_to_meters(x1, y1)
        mx2, my2 = pixel_to_meters(x2, y2)
        
        direction = np.array([mx2 - mx1, my2 - my1, 0])
        length_m = np.linalg.norm(direction[:2])
        
        if length_m < 0.1:
            continue
        
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        # Use measured thickness converted to meters
        wall_thickness_m = max(thickness_px / PIXELS_PER_METER, 0.02)  # Minimum 2cm
        perp = np.array([-direction_norm[1], direction_norm[0], 0]) * wall_thickness_m / 2
        
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
        
        if wall_texture:
            uv = create_uv_coords_wall(vertices, WALL_HEIGHT, length_m)
            wall_mesh = apply_texture_to_mesh(wall_mesh, wall_texture, uv)
        else:
            wall_mesh.visual.face_colors = WALL_COLOR
        
        wall_meshes.append((f"Wall_{i+1}", wall_mesh))
    
    # Create floor
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
        
        if floor_texture:
            uv = create_uv_coords_floor(floor_width, floor_depth)
            floor_mesh = apply_texture_to_mesh(floor_mesh, floor_texture, uv)
        else:
            floor_mesh.visual.face_colors = FLOOR_COLOR
    
    # Create scene and export
    scene = trimesh.Scene()
    
    if len(wall_meshes) > 0:
        all_verts = np.vstack([m.vertices for _, m in wall_meshes])
        if floor_mesh is not None:
            all_verts = np.vstack([all_verts, floor_mesh.vertices])
        center = all_verts.mean(axis=0)
        
        if floor_mesh is not None:
            floor_mesh.vertices -= center
            scene.add_geometry(floor_mesh, node_name="Floor", geom_name="Floor")
        
        for name, wall_mesh in wall_meshes:
            wall_mesh.vertices -= center
            scene.add_geometry(wall_mesh, node_name=name, geom_name=name)
    
    # Export files
    glb_path = f"{output_dir}/model.glb"
    obj_path = f"{output_dir}/model.obj"
    
    scene.export(glb_path)
    scene.export(obj_path)
    
    # Export individual files
    if floor_mesh is not None:
        floor_mesh.export(f"{output_dir}/floor.obj")
    
    if len(wall_meshes) > 0:
        walls_combined = trimesh.util.concatenate([m for _, m in wall_meshes])
        if not wall_texture:
            walls_combined.visual.face_colors = WALL_COLOR
        walls_combined.export(f"{output_dir}/walls.obj")
    
    return {
        'success': True,
        'output_dir': output_dir,
        'files': {
            'glb': glb_path,
            'obj': obj_path,
            'floor_obj': f"{output_dir}/floor.obj",
            'walls_obj': f"{output_dir}/walls.obj",
            'preview': f"{output_dir}/05_detected_walls.png"
        },
        'stats': {
            'horizontal_walls': h_count,
            'vertical_walls': v_count,
            'diagonal_walls': d_count,
            'total_walls': len(final_walls),
            'image_size': f"{width}x{height}"
        }
    }
