"""
PHASE 2: SEMANTIC EXTRACTION
Extracts structured data from AI-generated design
"""

import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np
import json
import cv2
import os

# ⚠️ EDIT THIS: Choose which design option you liked
SELECTED_IMAGE = "output/phase1_proposals/option_1_modern_minimalist.png"

OUTPUT_DIR = "output/phase2_semantic"
PARSED_GEOMETRY = "output/phase0_parsed/parsed_geometry.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 2: SEMANTIC EXTRACTION")
print("="*60)

# Load image
print(f"\n📂 Loading: {SELECTED_IMAGE}")
if not os.path.exists(SELECTED_IMAGE):
    print(f"❌ ERROR: File not found!")
    print(f"\n💡 Options:")
    print("   1. Make sure Phase 1 completed successfully")
    print("   2. Update SELECTED_IMAGE variable to your chosen design")
    exit(1)

image = Image.open(SELECTED_IMAGE)
print("✅ Image loaded")

# Load AI model
print("\n🤖 Loading segmentation model (first time takes 2-3 min)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-ade-semantic"
).to(device)

print("✅ Model loaded")

# Run segmentation
print("\n🔍 Analyzing image...")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = [image.size[::-1]]
results = processor.post_process_semantic_segmentation(
    outputs, target_sizes=target_sizes
)[0]

segmentation_map = results.cpu().numpy()
print("✅ Segmentation complete")

# Save visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(image)
plt.title("Original")
plt.axis('off')

plt.subplot(132)
plt.imshow(segmentation_map, cmap='tab20')
plt.title("Segmentation")
plt.axis('off')

plt.subplot(133)
overlay = np.array(image)
mask_colored = plt.cm.tab20(segmentation_map / (segmentation_map.max() + 1))[:, :, :3]
overlay = (overlay * 0.5 + mask_colored * 255 * 0.5).astype(np.uint8)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/segmentation_vis.png", dpi=150, bbox_inches='tight')
print(f"✅ Visualization saved")

# Extract objects
print("\n📦 Extracting objects...")

ADE_LABELS = {
    3: 'floor', 5: 'ceiling', 0: 'wall', 
    7: 'bed', 15: 'table', 19: 'sofa', 23: 'chair'
}

def extract_regions(seg_map, label_id, min_area=100):
    mask = (seg_map == label_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            regions.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'polygon': approx.reshape(-1, 2).tolist(),
                'area': float(area)
            })
    
    return regions

extracted_objects = {}

for label_id, label_name in ADE_LABELS.items():
    regions = extract_regions(segmentation_map, label_id)
    if regions:
        extracted_objects[label_name] = regions
        print(f"   ✅ {label_name}: {len(regions)}")

# Build scene graph
print("\n🏗️  Building scene graph...")

with open(PARSED_GEOMETRY, 'r') as f:
    original_geometry = json.load(f)

scene_graph = {
    'rooms': [],
    'objects': [],
    'walls': original_geometry['walls']
}

# Rooms
if 'floor' in extracted_objects:
    for i, floor_region in enumerate(extracted_objects['floor']):
        scene_graph['rooms'].append({
            'id': f'room_{i}',
            'boundary': floor_region['polygon'],
            'area_pixels': floor_region['area']
        })

# Objects
for obj_type in ['bed', 'table', 'sofa', 'chair']:
    if obj_type in extracted_objects:
        for obj in extracted_objects[obj_type]:
            scene_graph['objects'].append({
                'type': obj_type,
                'bbox': obj['bbox'],
                'center': [
                    obj['bbox'][0] + obj['bbox'][2] / 2,
                    obj['bbox'][1] + obj['bbox'][3] / 2
                ]
            })

# Save
with open(f"{OUTPUT_DIR}/scene_graph.json", 'w') as f:
    json.dump(scene_graph, f, indent=2)

print(f"\n✅ Scene graph saved:")
print(f"   Rooms: {len(scene_graph['rooms'])}")
print(f"   Objects: {len(scene_graph['objects'])}")
print(f"   Walls: {len(scene_graph['walls'])}")

print("\n" + "="*60)
print("✅ PHASE 2 COMPLETE!")
print("="*60)
print(f"\n📁 Outputs:")
print(f"   {OUTPUT_DIR}/scene_graph.json")
print(f"   {OUTPUT_DIR}/segmentation_vis.png")
print("\n▶️  NEXT: Run phase3_bim_reconstruction.py")
print("="*60)