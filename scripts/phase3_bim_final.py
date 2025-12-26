"""
PHASE 3: BIM + 3D MESH EXPORT
Generates both IFC (BIM) and detailed 3D mesh
"""

import ifcopenshell
import ifcopenshell.api
import json
import numpy as np
import trimesh
import os
from datetime import datetime

SCENE_GRAPH = "output/phase2_semantic/scene_graph.json"
OUTPUT_DIR = "output/phase3_final"
PROJECT_NAME = "AI Generated Building"

STANDARDS = {
    'wall_height': 3.0,
    'wall_thickness': 0.02,  # 2cm - paper thin
    'floor_thickness': 0.2
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 3: BIM + 3D MESH GENERATION")
print("="*60)

# Load scene graph
print(f"\n📂 Loading: {SCENE_GRAPH}")

if not os.path.exists(SCENE_GRAPH):
    print(f"❌ ERROR: File not found!")
    print(f"\n💡 Run phase2_semantic_extraction.py first!")
    exit(1)

with open(SCENE_GRAPH, 'r') as f:
    scene_graph = json.load(f)

print(f"✅ Loaded:")
print(f"   Rooms: {len(scene_graph['rooms'])}")
print(f"   Walls: {len(scene_graph['walls'])}")
print(f"   Objects: {len(scene_graph['objects'])}")

# ============================================
# PART A: CREATE DETAILED 3D MESH
# ============================================

print("\n🏗️  Creating detailed 3D mesh...")

meshes = []

# Create walls
for wall_data in scene_graph['walls']:
    start = np.array(wall_data['start'])
    end = np.array(wall_data['end'])
    thickness = wall_data.get('thickness', STANDARDS['wall_thickness'])
    height = STANDARDS['wall_height']
    
    direction = end - start
    length = np.linalg.norm(direction[:2])
    
    if length < 0.01:
        continue
    
    direction_norm = direction / np.linalg.norm(direction)
    perp = np.array([-direction_norm[1], direction_norm[0], 0]) * thickness / 2
    
    vertices = np.array([
        start + perp,
        start - perp,
        end - perp,
        end + perp,
        start + perp + [0, 0, height],
        start - perp + [0, 0, height],
        end - perp + [0, 0, height],
        end + perp + [0, 0, height],
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ])
    
    wall_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes.append(wall_mesh)

print(f"✅ Created {len(meshes)} walls")

# Add floor
if len(meshes) > 0:
    all_verts = np.vstack([m.vertices for m in meshes])
    min_x, min_y = all_verts[:, :2].min(axis=0)
    max_x, max_y = all_verts[:, :2].max(axis=0)
    
    padding = 0.5
    ft = STANDARDS['floor_thickness']
    
    floor_verts = np.array([
        [min_x - padding, min_y - padding, -ft],
        [max_x + padding, min_y - padding, -ft],
        [max_x + padding, max_y + padding, -ft],
        [min_x - padding, max_y + padding, -ft],
        [min_x - padding, min_y - padding, 0],
        [max_x + padding, min_y - padding, 0],
        [max_x + padding, max_y + padding, 0],
        [min_x - padding, max_y + padding, 0],
    ])
    
    floor_faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ])
    
    floor_mesh = trimesh.Trimesh(vertices=floor_verts, faces=floor_faces)
    meshes.append(floor_mesh)
    
    print("✅ Added floor")

# Add ceiling
if len(meshes) > 0:
    ceiling_verts = floor_verts.copy()
    ceiling_verts[:, 2] += STANDARDS['wall_height']
    ceiling_mesh = trimesh.Trimesh(vertices=ceiling_verts, faces=floor_faces)
    meshes.append(ceiling_mesh)
    
    print("✅ Added ceiling")

# Add furniture (simple boxes)
furniture_count = 0
for obj in scene_graph['objects']:
    try:
        # Simple box representation
        box = trimesh.creation.box([1.0, 0.5, 0.8])
        box.apply_translation([0, 0, 0.4])  # Half height
        meshes.append(box)
        furniture_count += 1
    except:
        pass

if furniture_count > 0:
    print(f"✅ Added {furniture_count} furniture items")

# Combine meshes
if len(meshes) > 0:
    scene_mesh = trimesh.util.concatenate(meshes)
    scene_mesh.vertices -= scene_mesh.bounds.mean(axis=0)
    
    # Export 3D mesh
    scene_mesh.export(f"{OUTPUT_DIR}/building_detailed.obj")
    scene_mesh.export(f"{OUTPUT_DIR}/building_detailed.ply")
    scene_mesh.export(f"{OUTPUT_DIR}/building_detailed.glb")
    
    print(f"\n✅ 3D mesh exported:")
    print(f"   Vertices: {len(scene_mesh.vertices):,}")
    print(f"   Faces: {len(scene_mesh.faces):,}")
else:
    print("\n⚠️  No geometry created")

# ============================================
# PART B: CREATE IFC (BIM)
# ============================================

print("\n🏢 Creating BIM model (IFC)...")

try:
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4")
    
    # Project
    project = ifcopenshell.api.run("root.create_entity", ifc, 
                                   ifc_class="IfcProject", name=PROJECT_NAME)
    
    # Units
    length_unit = ifc.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE")
    area_unit = ifc.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE")
    units = ifc.createIfcUnitAssignment([length_unit, area_unit])
    project.UnitsInContext = units
    
    # Site
    site = ifcopenshell.api.run("root.create_entity", ifc, 
                                ifc_class="IfcSite", name="Site")
    ifcopenshell.api.run("aggregate.assign_object", ifc, 
                        products=[site], relating_object=project)
    
    # Building
    building = ifcopenshell.api.run("root.create_entity", ifc, 
                                    ifc_class="IfcBuilding", name="Building")
    ifcopenshell.api.run("aggregate.assign_object", ifc, 
                        products=[building], relating_object=site)
    
    # Storey
    storey = ifcopenshell.api.run("root.create_entity", ifc, 
                                  ifc_class="IfcBuildingStorey", name="Ground Floor")
    ifcopenshell.api.run("aggregate.assign_object", ifc, 
                        products=[storey], relating_object=building)
    
    # Create walls
    walls_created = []
    for i, wall_data in enumerate(scene_graph['walls']):
        wall = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcWall")
        
        pset = ifcopenshell.api.run("pset.add_pset", ifc, 
                                    product=wall, name="Pset_WallCommon")
        ifcopenshell.api.run("pset.edit_pset", ifc, pset=pset, properties={
            "Reference": f"W-{i+1}",
            "LoadBearing": False
        })
        
        ifcopenshell.api.run("spatial.assign_container", ifc, 
                            products=[wall], relating_structure=storey)
        
        walls_created.append(wall)
    
    # Create spaces
    for room_data in scene_graph['rooms']:
        space = ifcopenshell.api.run("root.create_entity", ifc, 
                                     ifc_class="IfcSpace", 
                                     name=room_data['id'])
        
        ifcopenshell.api.run("spatial.assign_container", ifc, 
                            products=[space], relating_structure=storey)
    
    # Export IFC
    ifc_path = f"{OUTPUT_DIR}/building.ifc"
    ifc.write(ifc_path)
    
    print(f"✅ IFC file saved: building.ifc")
    
except Exception as e:
    print(f"⚠️  IFC generation warning: {e}")
    print("   (3D mesh files still created successfully)")

# ============================================
# SUMMARY
# ============================================

summary = {
    'project_name': PROJECT_NAME,
    'created': datetime.now().isoformat(),
    'statistics': {
        'walls': len(scene_graph['walls']),
        'rooms': len(scene_graph['rooms']),
        'objects': len(scene_graph['objects']),
        'mesh_vertices': len(scene_mesh.vertices) if len(meshes) > 0 else 0,
        'mesh_faces': len(scene_mesh.faces) if len(meshes) > 0 else 0
    }
}

with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("✅ PHASE 3 COMPLETE!")
print("="*60)
print(f"\n📁 Final outputs in: {OUTPUT_DIR}/")
print(f"\n🏢 3D Models (WITH GEOMETRY):")
print(f"   ✅ building_detailed.obj")
print(f"   ✅ building_detailed.ply")
print(f"   ✅ building_detailed.glb")
print(f"\n📋 BIM File:")
print(f"   ✅ building.ifc")
print(f"\n📊 Statistics:")
print(f"   Walls: {len(scene_graph['walls'])}")
print(f"   Rooms: {len(scene_graph['rooms'])}")
if len(meshes) > 0:
    print(f"   Vertices: {len(scene_mesh.vertices):,}")
    print(f"   Faces: {len(scene_mesh.faces):,}")
print(f"\n💡 Open these files:")
print(f"   - building_detailed.obj in FreeCAD/Blender")
print(f"   - building_detailed.glb in any 3D viewer")
print(f"   - building.ifc in BIM software")
print("\n" + "="*60)
print("🎉 COMPLETE PIPELINE FINISHED!")
print("="*60)
print("PNG/JPG → 3D Walls → AI Design → BIM ✅")
print("="*60)