"""
PHASE 1: AI DESIGN PROPOSALS
Generates 3 design options using AI
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw
import numpy as np
import json
import os

OUTPUT_DIR = "output/phase1_proposals"
PARSED_JSON = "output/phase0_parsed/parsed_geometry.json"

STYLES = ["modern minimalist", "luxury elegant", "scandinavian cozy"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 1: AI DESIGN PROPOSALS")
print("="*60)

# Load parsed geometry
print(f"\n📂 Loading: {PARSED_JSON}")

if not os.path.exists(PARSED_JSON):
    print(f"❌ ERROR: File not found!")
    print(f"\n💡 Run phase0_image_to_walls.py first!")
    exit(1)

with open(PARSED_JSON, 'r') as f:
    geometry = json.load(f)

walls = geometry['walls']
print(f"✅ Loaded {len(walls)} walls")

# Create control image
print("\n🎨 Creating layout control image...")

def create_layout_control(walls, size=(512, 512)):
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    all_points = []
    for wall in walls:
        all_points.extend(wall['start'][:2])
        all_points.extend(wall['end'][:2])
    
    all_points = np.array(all_points).reshape(-1, 2)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    
    scale = min(size[0] / (max_xy[0] - min_xy[0]), 
                size[1] / (max_xy[1] - min_xy[1])) * 0.8
    
    offset = np.array([size[0]/2, size[1]/2]) - (min_xy + max_xy) / 2 * scale
    
    for wall in walls:
        start = np.array(wall['start'][:2]) * scale + offset
        end = np.array(wall['end'][:2]) * scale + offset
        draw.line([tuple(start), tuple(end)], fill='black', width=5)
    
    return img

control_image = create_layout_control(walls)
control_image.save(f"{OUTPUT_DIR}/layout_control.png")
print("✅ Control image saved")

# Load AI models
print("\n🤖 Loading AI models...")
print("   ⏳ First time: 5-10 minutes download")
print("   ⏳ Next times: ~30 seconds")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"   Device: {device}")

try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=dtype
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    
    print("✅ Models loaded")
    
except Exception as e:
    import traceback
    print(f"❌ ERROR loading models:")
    print(f"   {type(e).__name__}: {e}")
    print(f"\n📋 Full error traceback:")
    traceback.print_exc()
    print(f"\n💡 Possible solutions:")
    print(f"   1. Check internet connection")
    print(f"   2. Try: pip install --upgrade diffusers transformers")
    print(f"   3. Clear cache: Delete ~/.cache/huggingface/")
    exit(1)

# Generate proposals
print("\n🎨 Generating design proposals...")
print(f"   Creating {len(STYLES)} design options")

control_edges = np.array(control_image.convert('L'))
control_edges = Image.fromarray(control_edges)

proposals = []

for i, style in enumerate(STYLES):
    print(f"\n▶️  Option {i+1}/{len(STYLES)}: {style}")
    
    prompt = f"interior design {style} living room bedroom kitchen realistic lighting professional photography 8k detailed"
    negative_prompt = "outdoor people blurry distorted low quality"
    
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_edges,
            num_inference_steps=30,
            controlnet_conditioning_scale=0.7,
            guidance_scale=7.5
        ).images[0]
        
        filename = f"option_{i+1}_{style.replace(' ', '_')}.png"
        filepath = f"{OUTPUT_DIR}/{filename}"
        result.save(filepath)
        
        proposals.append({
            'id': i + 1,
            'style': style,
            'image': filepath
        })
        
        print(f"   ✅ Saved: {filename}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        continue

if len(proposals) == 0:
    print("\n❌ ERROR: No proposals generated!")
    exit(1)

# Save metadata
with open(f"{OUTPUT_DIR}/proposals.json", 'w') as f:
    json.dump(proposals, f, indent=2)

print("\n" + "="*60)
print("✅ PHASE 1 COMPLETE!")
print("="*60)
print(f"\n📁 Generated {len(proposals)} design options:")
for p in proposals:
    print(f"   - {os.path.basename(p['image'])}")
print(f"\n💡 Next steps:")
print(f"   1. Open folder: {OUTPUT_DIR}")
print(f"   2. View all 3 design images")
print(f"   3. Pick your favorite!")
print(f"   4. Edit phase2_semantic_extraction.py")
print(f"      Update line 18: SELECTED_IMAGE = 'your_choice'")
print(f"   5. Run phase2_semantic_extraction.py")
print("="*60)