# Floor Plan to 3D Converter

A comprehensive system for converting 2D floor plans into fully navigable 3D interior models. Supports multiple input formats and exports to various 3D engines and libraries.

## Features

### Input Formats Supported
- **JSON/Structured Data**: Rooms, walls, doors, windows with coordinates
- **SVG Vector Graphics**: Scalable vector floor plans
- **DXF CAD Files**: Industry-standard CAD drawings
- **Raster Images**: PNG/JPG floor plan images with automatic detection
- **Grid-based Layouts**: Simple grid representations for quick prototyping

### Output Formats Supported
- **Blender** (Python/bpy): Full Python scripts for Blender 3D
- **Three.js** (JavaScript): Interactive web-based 3D scenes
- **Babylon.js**: WebGL-based 3D engine with advanced features
- **Unity** (C#): Game engine integration (planned)
- **OpenGL/WebGL**: Low-level graphics API implementations

### Core Capabilities
- **Wall Detection**: Automatic wall segmentation and thickness measurement
- **Room Recognition**: Intelligent room type classification
- **Door/Window Detection**: Automatic opening identification
- **Furniture Placement**: Procedural furniture generation based on room types
- **Material System**: Realistic materials and textures
- **Scale Normalization**: Automatic unit conversion and scaling

## Installation

### Requirements
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Trimesh (`pip install trimesh`)
- Pillow (`pip install pillow`)

### Install Package
```bash
# Clone or download the repository
cd floorplan_3d_converter

# Install dependencies
pip install -r requirements.txt

# Run examples
python -m floorplan_3d_converter.examples
```

## Quick Start

### Basic Usage

```python
from floorplan_3d_converter import convert_floorplan

# Convert JSON floor plan to Blender script
result = convert_floorplan(
    input_data='floorplan.json',
    input_format='json',
    output_format='blender',
    output_path='my_house.py'
)

if result['success']:
    print(f"Conversion successful! Output: {result['output_path']}")
```

### Input Format Examples

#### JSON Format
```json
{
  "dimensions": {"real_width": 10.0, "real_height": 8.0},
  "rooms": [
    {
      "id": "living_room",
      "name": "Living Room",
      "type": "living",
      "vertices": [[0, 0], [5, 0], [5, 4], [0, 4]],
      "center": [2.5, 2.0]
    }
  ],
  "walls": [
    {"start": [0, 0], "end": [5, 0], "thickness": 0.2, "height": 2.5}
  ],
  "doors": [
    {"position": [2.5, 0], "width": 0.9, "height": 2.1}
  ]
}
```

#### Grid Format
```json
{
  "grid": [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 2, 0],
    [0, 1, 1, 2, 0],
    [0, 3, 3, 2, 0],
    [0, 0, 0, 0, 0]
  ],
  "cell_size": 2.0
}
```

#### Image Processing
```python
# Process floor plan image
result = convert_floorplan(
    input_data='floorplan.png',
    input_format='image',
    output_format='threejs',
    output_path='floorplan_3d.html'
)
```

## API Reference

### Main Functions

#### `convert_floorplan(input_data, input_format='auto', output_format='blender', output_path='output.py', **kwargs)`

Main conversion function.

**Parameters:**
- `input_data`: Input floor plan data (file path, string, dict, or numpy array)
- `input_format`: Input format ('json', 'svg', 'image', 'grid', 'auto')
- `output_format`: Output format ('blender', 'threejs', 'babylonjs')
- `output_path`: Output file path
- `**kwargs`: Additional format-specific options

**Returns:** Dictionary with conversion results and statistics

#### `FloorPlanConverter` Class

Advanced usage with more control:

```python
from floorplan_3d_converter import FloorPlanConverter

converter = FloorPlanConverter()
result = converter.convert(input_data, input_format='json', output_format='blender')

# Access intermediate data
floorplan_data = converter.get_floorplan_data()
geometry = converter.get_geometry()
```

## Output Formats

### Blender (Python)
- **File**: `.py` script
- **Usage**: Run in Blender's text editor or import as module
- **Features**: Full 3D scene with materials, lighting, and camera
- **Dependencies**: Blender 3.0+

### Three.js (JavaScript)
- **File**: `.html` with embedded JavaScript
- **Usage**: Open in web browser
- **Features**: Interactive 3D scene with orbit controls
- **Dependencies**: Modern web browser with WebGL

### Babylon.js
- **File**: `.html` with embedded JavaScript
- **Usage**: Open in web browser
- **Features**: Advanced 3D engine with physics support
- **Dependencies**: Modern web browser with WebGL

## Room Types & Furniture

The system automatically places furniture based on room types:

- **Bedroom**: Bed, wardrobe
- **Bathroom**: Toilet, sink
- **Kitchen**: Kitchen counter, stove, sink
- **Living Room**: Sofa, dining table
- **Office**: Desk, wardrobe

## Architecture

```
floorplan_3d_converter/
├── __init__.py              # Package initialization
├── input_parser.py          # Input format parsers
├── geometry_generator.py    # 3D geometry creation
├── output_backends.py       # Output format exporters
├── converter.py             # Main conversion logic
└── examples.py              # Usage examples
```

### Input Parser
Handles multiple input formats and normalizes them into a unified `FloorPlanData` structure.

### Geometry Generator
Converts 2D floor plan data into 3D meshes:
- Wall extrusion
- Floor/ceiling generation
- Door/window cutouts
- Furniture placement

### Output Backends
Export 3D geometry to target formats with appropriate materials and scene setup.

## Examples

Run the included examples:

```bash
python -m floorplan_3d_converter.examples
```

This creates:
- `examples/simple_house_blender.py` - Basic house in Blender
- `examples/apartment_threejs.html` - Apartment in Three.js
- `examples/office_building_babylon.html` - Office building in Babylon.js

## Advanced Usage

### Custom Materials
```python
# Define custom materials in output backends
materials = {
    'custom_wall': create_material('Custom Wall', (0.8, 0.8, 0.9)),
    'glass': create_transparent_material('Glass', (0.9, 0.95, 1.0), 0.3)
}
```

### Room Type Inference
The system automatically infers room types from names:
- "bed", "bedroom" → bedroom
- "bath", "toilet" → bathroom
- "kitchen", "cook" → kitchen
- "living", "lounge" → living room

### Scale Handling
- **JSON/SVG**: Uses provided dimensions or assumes 1 unit = 1 meter
- **Images**: Automatically detects scale from wall measurements
- **Grid**: Uses specified cell_size parameter

## Troubleshooting

### Common Issues

1. **"No lines detected in image"**
   - Ensure the floor plan image has clear, dark lines on a light background
   - Try adjusting image contrast or using a different image

2. **"Unsupported output format"**
   - Check available formats: 'blender', 'threejs', 'babylonjs'
   - Unity and OpenGL backends are planned for future releases

3. **Poor geometry quality**
   - Ensure input data has reasonable scale (not too small or large)
   - Check that wall coordinates form valid polygons

### Performance Tips

- **Large floor plans**: Break into smaller sections
- **High detail**: Reduce mesh complexity for web exports
- **Memory usage**: Process one floor plan at a time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Adding New Output Formats

1. Create a new backend class inheriting from `OutputBackend`
2. Implement the `export()` method
3. Add to the `BACKENDS` dictionary
4. Update documentation

### Adding New Input Formats

1. Create a new parser class inheriting from `InputParser`
2. Implement the `parse()` method
3. Update the main `parse_floorplan()` function
4. Add format detection logic

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the examples and documentation
- Review the API reference

## Roadmap

- [ ] Unity C# backend
- [ ] OpenGL/WebGL backends
- [ ] DXF file support
- [ ] Advanced room recognition with ML
- [ ] Texture mapping system
- [ ] Physics simulation
- [ ] VR/AR export formats
