"""
Blender script to convert FBX to GLB
Run with: blender --background --python convert_fbx_to_glb.py -- input.fbx output.glb
"""
import bpy
import sys

# Get command line arguments after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) < 2:
    print("Usage: blender --background --python convert_fbx_to_glb.py -- input.fbx output.glb")
    sys.exit(1)

input_fbx = argv[0]
output_glb = argv[1]

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import FBX
bpy.ops.import_scene.fbx(filepath=input_fbx)

# Export as GLB
bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')

print(f"Converted: {input_fbx} -> {output_glb}")
