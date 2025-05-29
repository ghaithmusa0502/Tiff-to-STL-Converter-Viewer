# TIFF to STL Converter & Viewer

A powerful PyQt5-based application for converting 2D images into 3D STL models and viewing them in an interactive 3D environment. Perfect for creating heightmaps, topographical models, and converting grayscale images into printable 3D objects.

## Features

###  Image Processing
- **Multi-format Support:** Load PNG, JPG, JPEG, BMP, GIF, TIF, and TIFF images
- **Height Mapping:** Convert grayscale values to 3D height data
- **Invert Height Option:** Make darker pixels taller or shorter
- **Configurable Scale:** Set pixels-per-unit for precise dimensional control
- **Adjustable Max Height:** Control the maximum Z-height of your 3D model

###  Advanced Visualisation
- **Interactive 3D Viewer:** Real-time PyVista-powered 3D visualization
- **Dual Colour Modes:**
  - **Height-based Gradient:** Customizable 3-colour gradient (low/mid/high)
  - **Solid Color:** Uniform color with custom selection
- **Camera Controls:** Pan, zoom, rotate with reset-to-initial functionality
- **Grid & Axes:** Optional coordinate system display

###  Export & Save Options
- **STL Export:** Save as industry-standard STL files for 3D printing
- **Screenshot Capture:** Export high-quality PNG images of your 3D model
- **Smart Naming:** Automatic filename suggestions based on source files

###  Dual Workflow Support
- **Image → STL:** Convert 2D images to 3D models
- **STL Viewer:** Load and visualise existing STL files with enhanced colouring

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Dependencies

Install all required dependencies:

```bash
pip install PyQt5 Pillow numpy numpy-stl pyvista pyvistaqt
```

### Individual Package Details:
- **PyQt5**: GUI framework
- **Pillow (PIL)**: Image processing
- **numpy**: Numerical computations
- **numpy-stl**: STL file handling
- **pyvista**: 3D visualization and mesh processing
- **pyvistaqt**: PyVista-Qt integration

## Usage

### Getting Started

1. **Launch the Application:**
   ```bash
   python "Tiff_to_STL_converter & viewer.py"
   ```

2. **Choose Your Workflow:**
   - **For Image Conversion:** Use "Browse Image for Conversion..."
   - **For STL Viewing:** Use "Load Existing STL..."

### Image to STL Conversion Workflow

#### Step 1: Load Image
- Click **"Browse Image for Conversion..."**
- Select your image file (PNG, JPG, TIFF, etc.)
- Preview appears in the left panel

#### Step 2: Adjust Conversion Settings
- **Max Height (mm):** Maximum Z-dimension of your 3D model
- **Pixels per Unit (mm):** Scale factor (e.g., 0.1 = 10 pixels per mm)
- **Invert Height:** Check to make darker areas taller

#### Step 3: Convert & Customise
- Click **"Convert Image to STL"**
- Model appears in 3D viewer
- Adjust colours using the colour controls
- Use camera controls to examine your model

#### Step 4: Export
- **Save STL:** Export for 3D printing
- **Save Screenshot:** Capture current view as PNG
- **Reset View:** Return to initial camera position

### STL Viewing Workflow

1. Click **"Load Existing STL..."**
2. Select your STL file
3. Model loads with automatic height-based colouring
4. Use colour controls to customise appearance
5. Export screenshots or save modified STL

## Interface Overview

### Control Panel (Left Side)

#### 1. Load Image or STL
- File loading buttons
- Image preview area

#### 2. Adjust Conversion Settings
- Height and scale parameters
- Invert height checkbox

#### 3. Convert & Save
- Conversion trigger
- Export options (STL, Screenshot)
- View reset

#### 4. Change Model Colour
- **Colour Mode Selection:**
  - Original (Height-based gradient)
  - Solid Colour
- **Height Gradient Colours:**
  - Low Height Colour (typically black/dark)
  - Mid Height Colour (typically gold/yellow)
  - High Height Colour (typically red/bright)
- **Custom Solid Color:** Color picker for uniform coloring

### 3D Viewer (Right Side)
- Interactive PyVista visualisation
- Mouse controls: rotate, pan, zoom
- Grid and axes display
- Real-time colour updates

## Technical Architecture

### Core Classes

#### `ImageToSTLApp` (Main Application)
- PyQt5 QMainWindow-based GUI
- Coordinates all functionality
- Manages UI state and user interactions

#### Key Methods:
- `load_image()`: Image file loading and preview
- `load_stl()`: STL file loading with format validation
- `convert_image_to_stl()`: Core conversion algorithm
- `apply_color_to_mesh()`: Dynamic colour application
- `save_stl()`: STL export functionality

### Conversion Algorithm

1. **Image Processing:**
   - Convert to grayscale using PIL
   - Normalise pixel values (0-255 → 0-1)
   - Apply height scaling and inversion

2. **Mesh Generation:**
   - Create a vertex grid from pixel positions
   - Generate triangular faces for the surface
   - Build PyVista PolyData structure

3. **Dual Format Support:**
   - PyVista mesh for visualisation
   - numpy-stl mesh for file export

### Colour System

#### Height-based Gradient
- Three-point gradient interpolation
- Customizable low/mid/high colours
- Real-time colour computation based on Z-values

#### Solid Colour Mode
- Uniform colour application
- Custom colour picker integration
- Preserved colour preferences

## File Formats

### Supported Input Formats
- **Images:** PNG, JPG, JPEG, BMP, GIF, TIF, TIFF
- **3D Models:** STL (binary and ASCII)

### Output Formats
- **STL:** Industry-standard 3D printing format
- **PNG:** High-resolution screenshots

## Tips & Best Practices

### For Best Results:
1. **Use High-Contrast Images:** Better height differentiation
2. **Consider Scale:** Match pixels-per-unit to your intended print size
3. **Test Height Settings:** Start with moderate max heights (5-20mm)
4. **Preview Before Export:** Use 3D viewer to verify model appearance

## System Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python:** 3.7+
- **Memory:** 4GB+ RAM recommended for large images
- **Graphics:** OpenGL-compatible graphics card for 3D visualization

## Acknowledgments

- **PyVista:** Excellent 3D visualization capabilities
- **numpy-stl:** Robust STL file handling
- **PyQt5:** Comprehensive GUI framework

