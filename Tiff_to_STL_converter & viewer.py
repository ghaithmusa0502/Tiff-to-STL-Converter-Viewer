import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QLabel, QLineEdit, QCheckBox, QMessageBox, QFrame,
    QHBoxLayout, QSizePolicy, QComboBox, QColorDialog, QProgressBar,
    QSpinBox, QGroupBox, QTextEdit, QSplitter, QTabWidget
)
from PyQt5.QtGui import QPixmap, QIcon, QColor, QFont
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PIL import Image, ImageFilter
import numpy as np
from stl import mesh
from scipy import ndimage
from scipy.spatial.distance import cdist
import gc

import pyvista as pv
from pyvistaqt import QtInteractor

class ProcessingThread(QThread):
    """Dedicated thread for heavy processing tasks to prevent UI freezing"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    processing_complete = pyqtSignal(object, object)  # stl_mesh, pyvista_mesh
    processing_error = pyqtSignal(str)
    
    def __init__(self, image_path, settings):
        super().__init__()
        self.image_path = image_path
        self.settings = settings
        
    def run(self):
        try:
            self.status_updated.emit("Loading and preprocessing image...")
            self.progress_updated.emit(10)
            
            # Load and preprocess image
            img = Image.open(self.image_path).convert('L')
            
            # Apply preprocessing filters if enabled
            if self.settings.get('gaussian_blur', 0) > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=self.settings['gaussian_blur']))
                self.progress_updated.emit(20)
                
            if self.settings.get('edge_smoothing', False):
                img_array = np.array(img)
                img_array = ndimage.gaussian_filter(img_array, sigma=1)
                img = Image.fromarray(img_array.astype(np.uint8))
                self.progress_updated.emit(30)
            
            # Check image size and downsample if necessary
            original_size = img.size
            max_dimension = self.settings.get('max_dimension', 2000)
            
            if max(img.size) > max_dimension:
                self.status_updated.emit(f"Large image detected ({original_size[0]}x{original_size[1]}). Resizing to optimize performance...")
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.progress_updated.emit(40)
                
            self.status_updated.emit("Converting image to height map...")
            img_array = np.array(img)
            
            max_height = self.settings['max_height']
            unit_per_pixel = self.settings['unit_per_pixel']
            invert_height = self.settings['invert_height']
            
            height_map = img_array / 255.0 * max_height
            if invert_height:
                height_map = max_height - height_map
                
            self.progress_updated.emit(50)
            
            # Generate mesh with chunked processing for large images
            self.status_updated.emit("Generating 3D mesh...")
            rows, cols = height_map.shape
            
            # Use chunked processing for very large meshes
            # The chunk size from settings will override the default if available
            chunk_size = self.settings.get('chunk_size', min(500, rows // 4) if rows > 1000 else rows)
            
            vertices = []
            faces = []
            
            # Process in chunks to manage memory
            for chunk_start in range(0, rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, rows)
                chunk_progress = 50 + int((chunk_start / rows) * 40)
                self.progress_updated.emit(chunk_progress)
                
                # Generate vertices for this chunk
                chunk_vertices = []
                for r in range(chunk_start, chunk_end):
                    for c in range(cols):
                        x = c * unit_per_pixel
                        y = r * unit_per_pixel
                        z = height_map[r, c]
                        chunk_vertices.append([x, y, z])
                        
                vertices.extend(chunk_vertices)
                
            pass # Faces will be generated after all vertices are collected below

            # All vertices collected, now generate all faces
            # This ensures correct indexing across the entire height map.
            self.status_updated.emit("Generating faces for 3D mesh...")
            for r in range(rows - 1):
                for c in range(cols - 1):
                    p1 = r * cols + c
                    p2 = r * cols + (c + 1)
                    p3 = (r + 1) * cols + c
                    p4 = (r + 1) * cols + (c + 1)
                    
                    faces.append([p1, p3, p2])
                    faces.append([p2, p3, p4])
            
            # Force garbage collection to manage memory before final array conversion
            if self.settings.get('auto_gc', True):
                gc.collect()
                    
            self.progress_updated.emit(90)
            self.status_updated.emit("Finalizing mesh data structures...")
            
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)
            
            # Validate mesh integrity
            if vertices.size == 0 or faces.size == 0:
                raise ValueError("Image conversion resulted in no vertices or faces.")
                
            # Check for invalid face indices
            max_vertex_index = len(vertices) - 1
            if np.any(faces > max_vertex_index) or np.any(faces < 0):
                raise ValueError("Invalid face indices detected in mesh.")
                
            # Create PyVista mesh with proper validation
            # PyVista's faces array format: [N, p1, p2, ..., pN, M, p1, p2, ..., pM, ...]
            # For triangles, N is always 3.
            pyvista_faces = np.hstack((np.full((len(faces), 1), 3), faces)).flatten()
            pyvista_mesh = pv.PolyData(vertices, pyvista_faces)
            
            # Validate PyVista mesh
            if not pyvista_mesh.faces.any() or pyvista_mesh.n_cells == 0:
                raise ValueError("Generated PyVista mesh has no faces or cells.")
            
            # Apply Level of Detail (LOD) if threshold exceeded
            lod_threshold = self.settings.get('lod_threshold', 5000)
            if pyvista_mesh.n_cells > lod_threshold:
                self.status_updated.emit(f"Mesh has {pyvista_mesh.n_cells} cells, applying decimation...")
                # Reduce by 50% for high poly count
                pyvista_mesh = pyvista_mesh.decimate(target_reduction=0.5) 
                self.status_updated.emit(f"Mesh reduced to {pyvista_mesh.n_cells} cells.")

            # Create numpy-stl mesh with error checking
            # stl.mesh expects `vectors` which is an array of (3, 3) arrays representing triangles.
            # Each (3, 3) array contains 3 vertices (each a 3-element coordinate).
            stl_mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                try:
                    # f contains vertex indices (p1, p2, p3)
                    stl_mesh_data.vectors[i] = [vertices[f[0]], vertices[f[1]], vertices[f[2]]]
                except IndexError as e:
                    raise ValueError(f"Face index error at triangle {i}, face indices {f}: {e}")
                    
            self.progress_updated.emit(100)
            self.status_updated.emit("Mesh generation complete!")
            self.processing_complete.emit(stl_mesh_data, pyvista_mesh)
            
        except Exception as e:
            self.processing_error.emit(str(e))

class MeshValidator:
    """Comprehensive STL mesh validation utilities"""
    
    @staticmethod
    def validate_stl_structure(mesh_data):
        """Validate STL mesh structure and return detailed report"""
        issues = []
        warnings = []
        
        if mesh_data is None:
            issues.append("Mesh data is None")
            return issues, warnings
            
        try:
            # Check for empty mesh
            if hasattr(mesh_data, 'vectors') and len(mesh_data.vectors) == 0:
                issues.append("Mesh contains no triangles")
                
            # Check for degenerate triangles
            if hasattr(mesh_data, 'vectors'):
                degenerate_count = 0
                for i, triangle in enumerate(mesh_data.vectors):
                    # Check if all three vertices are collinear or identical
                    v0, v1, v2 = triangle
                    # Calculate edge vectors
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    # Check if cross product is zero vector (collinear)
                    if np.allclose(np.cross(edge1, edge2), [0, 0, 0]):
                        degenerate_count += 1
                        
                if degenerate_count > 0:
                    warnings.append(f"Found {degenerate_count} degenerate triangles.")
                    if degenerate_count == len(mesh_data.vectors):
                        issues.append("All triangles are degenerate.")
                    
            # Check for NaN or Inf coordinates
            if hasattr(mesh_data, 'vectors'):
                if np.any(np.isnan(mesh_data.vectors)) or np.any(np.isinf(mesh_data.vectors)):
                    issues.append("Mesh contains NaN or Infinite coordinate values.")
                    
        except Exception as e:
            issues.append(f"Validation error (numpy-stl): {str(e)}")
            
        return issues, warnings
    
    @staticmethod
    def validate_pyvista_mesh(pv_mesh):
        """Validate PyVista mesh structure"""
        issues = []
        warnings = []
        
        if pv_mesh is None:
            issues.append("PyVista mesh is None")
            return issues, warnings
            
        try:
            # Check basic structure
            if pv_mesh.n_points == 0:
                issues.append("Mesh has no vertices.")
            if pv_mesh.n_cells == 0:
                issues.append("Mesh has no faces.")
                
            # Check face array format (simplistic check for triangles)
            # PyVista's faces array is a 1D array where every 4th element is the number of points (3 for triangle)
            if hasattr(pv_mesh, 'faces') and len(pv_mesh.faces) > 0:
                if len(pv_mesh.faces) % 4 != 0:
                    warnings.append("PyVista face array format may be non-standard for triangles (expected multiple of 4).")
                
                # Check for valid face indices relative to n_points
                # This is more robust check. The `faces` array has the structure [3, i0, i1, i2, 3, i3, i4, i5, ...]
                # So we can extract the vertex indices by slicing.
                face_indices = pv_mesh.faces[1::4] # First index of each triangle
                face_indices = np.concatenate((face_indices, pv_mesh.faces[2::4], pv_mesh.faces[3::4]))
                
                if np.any(face_indices >= pv_mesh.n_points) or np.any(face_indices < 0):
                    issues.append("PyVista mesh contains out-of-bounds face indices.")

            # Check for valid bounds
            if pv_mesh.n_points > 0:
                bounds = pv_mesh.bounds
                if any(np.isnan(bounds)) or any(np.isinf(bounds)):
                    issues.append("PyVista mesh contains invalid coordinate values (NaN/Inf) in bounds.")
                    
        except Exception as e:
            issues.append(f"PyVista validation error: {str(e)}")
            
        return issues, warnings

class PerformanceMonitor:
    """Monitor and optimize rendering performance"""

    def __init__(self):
        self.frame_times = []
        self.last_frame_time = time.time()

    def update_frame_time(self):
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)

        # Keep only last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        self.last_frame_time = current_time

    def get_average_fps(self):  # <--- Ensure 'plotter' argument is NOT here
        if not self.frame_times:
            return 0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    def is_performance_good(self):
        return self.get_average_fps() > 15  # Consider 15+ FPS as good

class ImageToSTLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.stl_mesh_data = None
        self.pyvista_mesh = None
        self.loaded_stl_path = None
        self.initial_camera_position = None
        self.current_mesh_actor = None
        self.processing_thread = None

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_stats)
        # Start the timer AFTER the plotter is initialized in init_ui
        # self.performance_timer.start(1000)  # Update every second

        # Color settings
        self.last_applied_color_type = 'original'
        self.last_custom_color = (1.0, 0.843, 0.0)  # Default solid color: Goldenrod
        self.low_height_color = (0.0, 0.0, 0.0)    # Default gradient start: Black
        self.mid_height_color = (1.0, 0.843, 0.0)  # Default gradient mid: Goldenrod
        self.high_height_color = (1.0, 0.0, 0.0)   # Default gradient end: Red

        # Viewer and validation
        self.viewer_widget = None
        self.plotter = None
        self.mesh_validator = MeshValidator()

        self.init_ui()
        # Now that self.plotter is initialized, start the timer
        self.performance_timer.start(1000) # Update every second

    def init_ui(self):
        self.setWindowTitle('Image to STL Converter & Viewer')
        # Use a more flexible initial size, allow scaling by user
        self.setGeometry(100, 100, 1400, 1000)
        self.setWindowIcon(QIcon(self.get_resource_path('icon.png')))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create tabbed interface for better organization
        tab_widget = QTabWidget()
        central_widget_layout = QVBoxLayout()
        central_widget.setLayout(central_widget_layout)
        central_widget_layout.addWidget(tab_widget)

        # Main conversion tab
        main_tab = QWidget()
        tab_widget.addTab(main_tab, "Converter & Viewer")

        # Advanced settings tab
        advanced_tab = QWidget()
        tab_widget.addTab(advanced_tab, "Advanced Settings")

        # Diagnostics tab
        diagnostics_tab = QWidget()
        tab_widget.addTab(diagnostics_tab, "Diagnostics")

        self.setup_main_tab(main_tab)
        self.setup_advanced_tab(advanced_tab)
        self.setup_diagnostics_tab(diagnostics_tab)

        # Status bar with performance info
        self.status_label = QLabel('Ready: Load an image or STL file.')
        self.performance_label = QLabel('FPS: 0')
        self.memory_label = QLabel('Memory: 0 MB')
        
        self.statusBar().addWidget(self.status_label)
        self.statusBar().addPermanentWidget(self.performance_label)
        self.statusBar().addPermanentWidget(self.memory_label)

        # Removed: self.plotter.on_show_event(self.performance_monitor.update_frame_time)
        # This line caused the error. FPS will now be polled by the QTimer.


    def setup_main_tab(self, main_tab):
        main_h_layout = QHBoxLayout()
        main_tab.setLayout(main_h_layout)

        # Controls panel
        controls_widget = QWidget()
        controls_widget.setFixedWidth(450)
        controls_v_layout = QVBoxLayout()
        controls_widget.setLayout(controls_v_layout)
        main_h_layout.addWidget(controls_widget)

        # Image/STL Loading Section
        image_section = QGroupBox("1. Load Image or STL")
        image_layout = QVBoxLayout()
        image_section.setLayout(image_layout)
        controls_v_layout.addWidget(image_section)

        self.load_stl_btn = QPushButton('Load Existing STL...')
        self.load_stl_btn.clicked.connect(self.load_stl)
        image_layout.addWidget(self.load_stl_btn)

        self.load_image_btn = QPushButton('Browse Image for Conversion...')
        self.load_image_btn.clicked.connect(self.load_image)
        image_layout.addWidget(self.load_image_btn)

        self.image_display_label = QLabel('No image loaded.')
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setFixedSize(300, 200) # Fixed size for preview
        self.image_display_label.setScaledContents(True)
        self.image_display_label.setStyleSheet("border: 1px solid gray;")
        image_layout.addWidget(self.image_display_label, alignment=Qt.AlignCenter)

        # Conversion Settings Section
        settings_section = QGroupBox("2. Conversion Settings")
        settings_layout = QVBoxLayout()
        settings_section.setLayout(settings_layout)
        controls_v_layout.addWidget(settings_section)

        # Max Height
        max_height_layout = QHBoxLayout()
        max_height_layout.addWidget(QLabel("Max Height (mm):"))
        self.max_height_input = QLineEdit('10.0')
        max_height_layout.addWidget(self.max_height_input)
        settings_layout.addLayout(max_height_layout)

        # Scale
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Unit per Pixel (mm):")) # Changed label for clarity
        self.scale_input = QLineEdit('0.1')
        scale_layout.addWidget(self.scale_input)
        settings_layout.addLayout(scale_layout)

        # Max dimension for large images
        max_dim_layout = QHBoxLayout()
        max_dim_layout.addWidget(QLabel("Max Image Dimension:"))
        self.max_dimension_input = QSpinBox()
        self.max_dimension_input.setRange(500, 5000)
        self.max_dimension_input.setSingleStep(100) # Step by 100
        self.max_dimension_input.setValue(2000)
        self.max_dimension_input.setToolTip("Large images will be resized for performance. Set to 0 to disable resizing.")
        max_dim_layout.addWidget(self.max_dimension_input)
        settings_layout.addLayout(max_dim_layout)

        self.invert_height_checkbox = QCheckBox('Invert Height (darker = taller)')
        settings_layout.addWidget(self.invert_height_checkbox)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        settings_layout.addWidget(self.progress_bar)

        # Convert & Save Section
        actions_section = QGroupBox("3. Convert & Save")
        actions_layout = QVBoxLayout()
        actions_section.setLayout(actions_layout)
        controls_v_layout.addWidget(actions_section)

        self.convert_btn = QPushButton('Convert Image to STL')
        self.convert_btn.clicked.connect(self.convert_image_to_stl)
        self.convert_btn.setEnabled(False)
        actions_layout.addWidget(self.convert_btn)

        self.save_stl_btn = QPushButton('Save Current STL File...')
        self.save_stl_btn.clicked.connect(self.save_stl)
        self.save_stl_btn.setEnabled(False)
        actions_layout.addWidget(self.save_stl_btn)

        self.save_screenshot_btn = QPushButton('Save Screenshot (PNG)')
        self.save_screenshot_btn.clicked.connect(self.save_screenshot)
        self.save_screenshot_btn.setEnabled(False)
        actions_layout.addWidget(self.save_screenshot_btn)

        self.reset_view_btn = QPushButton('Reset View to Initial Position')
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setEnabled(False)
        actions_layout.addWidget(self.reset_view_btn)

        # Color Settings Section
        color_section = QGroupBox("4. Model Appearance")
        color_layout = QVBoxLayout()
        color_section.setLayout(color_layout)
        controls_v_layout.addWidget(color_section)

        self.color_combo_box = QComboBox()
        self.color_combo_box.addItem("Original (Height-based)")
        self.color_combo_box.addItem("Solid Color")
        self.color_combo_box.currentIndexChanged.connect(self.on_color_selection_changed)
        self.color_combo_box.setEnabled(False)
        color_layout.addWidget(self.color_combo_box)

        self.select_custom_color_btn = QPushButton('Select Custom Solid Color...')
        self.select_custom_color_btn.clicked.connect(self.select_custom_color)
        self.select_custom_color_btn.setEnabled(False) # Default disabled
        color_layout.addWidget(self.select_custom_color_btn)

        # Gradient colors
        color_layout.addWidget(QLabel("Height Gradient Colors:"))
        
        gradient_buttons_layout = QVBoxLayout()
        
        self.select_low_color_btn = QPushButton('Low Height Color')
        self.select_low_color_btn.clicked.connect(lambda: self._select_gradient_color('low'))
        self.select_low_color_btn.setEnabled(False) # Default disabled

        self.select_mid_color_btn = QPushButton('Mid Height Color')
        self.select_mid_color_btn.clicked.connect(lambda: self._select_gradient_color('mid'))
        self.select_mid_color_btn.setEnabled(False) # Default disabled

        self.select_high_color_btn = QPushButton('High Height Color')
        self.select_high_color_btn.clicked.connect(lambda: self._select_gradient_color('high'))
        self.select_high_color_btn.setEnabled(False) # Default disabled
        
        gradient_buttons_layout.addWidget(self.select_low_color_btn)
        gradient_buttons_layout.addWidget(self.select_mid_color_btn)
        gradient_buttons_layout.addWidget(self.select_high_color_btn)
        color_layout.addLayout(gradient_buttons_layout)


        controls_v_layout.addStretch()

        # 3D Viewer
        self.viewer_widget = QtInteractor(self)
        self.plotter = self.viewer_widget.interactor
        self.plotter.set_background('white')
        self.plotter.show_axes()
        self.plotter.show_grid()
        main_h_layout.addWidget(self.viewer_widget)

    def setup_advanced_tab(self, advanced_tab):
        advanced_layout = QVBoxLayout()
        advanced_tab.setLayout(advanced_layout)

        # Image preprocessing section
        preprocessing_section = QGroupBox("Image Preprocessing")
        preprocessing_layout = QVBoxLayout()
        preprocessing_section.setLayout(preprocessing_layout)
        advanced_layout.addWidget(preprocessing_section)

        # Gaussian blur
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Gaussian Blur Radius:"))
        self.gaussian_blur_input = QSpinBox()
        self.gaussian_blur_input.setRange(0, 10)
        self.gaussian_blur_input.setValue(0)
        self.gaussian_blur_input.setToolTip("Apply Gaussian blur to smooth image before conversion (0 for no blur)")
        blur_layout.addWidget(self.gaussian_blur_input)
        preprocessing_layout.addLayout(blur_layout)

        self.edge_smoothing_checkbox = QCheckBox('Enable Edge Smoothing')
        self.edge_smoothing_checkbox.setToolTip("Applies an additional Gaussian filter to the height map to reduce sharp edges")
        preprocessing_layout.addWidget(self.edge_smoothing_checkbox)

        # Performance settings
        performance_section = QGroupBox("Performance Optimization")
        performance_layout = QVBoxLayout()
        performance_section.setLayout(performance_layout)
        advanced_layout.addWidget(performance_section)

        # Memory management
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(QLabel("Processing Chunk Size (rows):"))
        self.chunk_size_input = QSpinBox()
        self.chunk_size_input.setRange(100, 2000) # Increased max range
        self.chunk_size_input.setSingleStep(50)
        self.chunk_size_input.setValue(500)
        self.chunk_size_input.setToolTip("Smaller values use less memory but might increase processing time for very large images.")
        memory_layout.addWidget(self.chunk_size_input)
        performance_layout.addLayout(memory_layout)

        self.auto_gc_checkbox = QCheckBox('Automatic Garbage Collection')
        self.auto_gc_checkbox.setChecked(True)
        self.auto_gc_checkbox.setToolTip("Enable to automatically free memory during heavy processing stages.")
        performance_layout.addWidget(self.auto_gc_checkbox)

        # LOD settings
        lod_layout = QHBoxLayout()
        lod_layout.addWidget(QLabel("Mesh Decimation Threshold (faces):"))
        self.lod_threshold_input = QSpinBox()
        self.lod_threshold_input.setRange(0, 500000) # Up to 500k faces
        self.lod_threshold_input.setSingleStep(1000)
        self.lod_threshold_input.setValue(50000) # Default to 50k faces before decimation
        self.lod_threshold_input.setToolTip("If the generated mesh has more faces than this, it will be automatically decimated. Set to 0 to disable.")
        lod_layout.addWidget(self.lod_threshold_input)
        performance_layout.addLayout(lod_layout)

        advanced_layout.addStretch()

    def setup_diagnostics_tab(self, diagnostics_tab):
        diagnostics_layout = QVBoxLayout()
        diagnostics_tab.setLayout(diagnostics_layout)

        # Mesh validation section
        validation_section = QGroupBox("Mesh Validation")
        validation_layout = QVBoxLayout()
        validation_section.setLayout(validation_layout)
        diagnostics_layout.addWidget(validation_section)

        self.validate_mesh_btn = QPushButton('Validate Current Mesh')
        self.validate_mesh_btn.clicked.connect(self.validate_current_mesh)
        self.validate_mesh_btn.setEnabled(False)
        validation_layout.addWidget(self.validate_mesh_btn)

        self.validation_results = QTextEdit()
        self.validation_results.setMaximumHeight(200)
        self.validation_results.setReadOnly(True)
        self.validation_results.setFont(QFont("Monospace", 9)) # Better for reports
        validation_layout.addWidget(self.validation_results)

        # Performance metrics
        performance_section = QGroupBox("Performance Metrics")
        performance_layout = QVBoxLayout()
        performance_section.setLayout(performance_layout)
        diagnostics_layout.addWidget(performance_section)

        self.performance_display = QTextEdit()
        self.performance_display.setMaximumHeight(150)
        self.performance_display.setReadOnly(True)
        self.performance_display.setFont(QFont("Monospace", 9))
        performance_layout.addWidget(self.performance_display)

        # System information
        system_section = QGroupBox("System Information")
        system_layout = QVBoxLayout()
        system_section.setLayout(system_layout)
        diagnostics_layout.addWidget(system_section)

        self.system_info_display = QTextEdit()
        self.system_info_display.setMaximumHeight(150)
        self.system_info_display.setReadOnly(True)
        self.system_info_display.setFont(QFont("Monospace", 9))
        self.update_system_info() # Populate on startup
        system_layout.addWidget(self.system_info_display)

        diagnostics_layout.addStretch()

    def get_resource_path(self, relative_path):
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def _reset_viewer_state(self):
        """Enhanced state reset with proper cleanup"""
        # Stop any running processing
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate() # Terminate the thread if it's still running
            self.processing_thread.wait() # Wait for it to finish
            self.processing_thread = None # Clear reference
            
        self.stl_mesh_data = None
        self.pyvista_mesh = None
        self.loaded_stl_path = None
        self.initial_camera_position = None
        
        # Clear viewer and remove actors
        self.plotter.clear_actors() # Remove all actors
        self.plotter.reset_camera()
        
        # Reset UI elements
        self.convert_btn.setEnabled(False)
        self.save_stl_btn.setEnabled(False)
        self.save_screenshot_btn.setEnabled(False)
        self.reset_view_btn.setEnabled(False)
        self.validate_mesh_btn.setEnabled(False)
        
        self.color_combo_box.setEnabled(False)
        self.select_custom_color_btn.setEnabled(False)
        self.select_low_color_btn.setEnabled(False)
        self.select_mid_color_btn.setEnabled(False)
        self.select_high_color_btn.setEnabled(False)
        self.color_combo_box.setCurrentIndex(0) # Reset to Original
        self.last_applied_color_type = 'original' # Reset tracked color type
        
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText('Ready: Load an image or STL file.')
        self.validation_results.clear()
        self.image_display_label.setPixmap(QPixmap()) # Clear image preview
        self.image_display_label.setText('No image loaded.')
        
        # Force garbage collection
        gc.collect()

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Open Image', '',
            'Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff)'
        )
        if file_path:
            try:
                # Validate image can be opened
                test_img = Image.open(file_path)
                test_img.verify() # Checks if the file is a valid image
                test_img.close() # Close image handle

                self._reset_viewer_state() # Reset state before loading new image
                self.image_path = file_path
                self.status_label.setText(f'Loaded Image: {os.path.basename(file_path)}')
                
                # Display image preview
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(
                    self.image_display_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_display_label.setPixmap(scaled_pixmap)

                self.convert_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Image Load Error", f"Failed to load image: {e}")
                self.status_label.setText(f'Error loading image: {e}')
                self._reset_viewer_state() # Ensure clean state on error

    def load_stl(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Open STL File', '',
            'STL Files (*.stl)'
        )
        if file_path:
            try:
                self._reset_viewer_state()

                # Load with PyVista
                temp_pv_mesh = pv.read(file_path)
                
                # Basic check for empty mesh from file
                if temp_pv_mesh.n_points == 0 or temp_pv_mesh.n_cells == 0:
                    raise ValueError("Loaded STL file contains no vertices or faces.")

                self.pyvista_mesh = temp_pv_mesh
                self.loaded_stl_path = file_path

                # Comprehensive validation for the loaded PyVista mesh
                pv_issues, pv_warnings = self.mesh_validator.validate_pyvista_mesh(self.pyvista_mesh)
                
                report_message = ""
                if pv_issues:
                    report_message += "--- PyVista Mesh Issues ---\n" + "\n".join(pv_issues) + "\n"
                    # If critical issues, consider raising an error
                    QMessageBox.critical(self, "STL Load Error", f"Critical issues detected in PyVista mesh: {'; '.join(pv_issues)}")
                    self.status_label.setText(f'Error loading STL: {"; ".join(pv_issues)}')
                    self._reset_viewer_state()
                    return # Stop processing if critical issues

                if pv_warnings:
                    report_message += "--- PyVista Mesh Warnings ---\n" + "\n".join(pv_warnings) + "\n"
                    QMessageBox.warning(self, "STL Load Warnings", f"Warnings detected in PyVista mesh: {'; '.join(pv_warnings)}")
                
                self.validation_results.setText(report_message if report_message else "No significant issues or warnings detected for PyVista mesh.")


                # Attempt to convert PyVista mesh to numpy-stl mesh for saving functionality
                # This part is crucial for the 'Save Current STL File' button to work after loading.
                try:
                    # PyVista's faces array format: [N, p1, p2, ..., pN, M, p1, p2, ..., pM, ...]
                    # For triangles, N is always 3. So, we slice to get the vertex indices.
                    # We need to reshape the faces array to (num_faces, 3) where each row is (idx0, idx1, idx2)
                    # The `faces` array from pyvista.PolyData is `[3, v0, v1, v2, 3, v3, v4, v5, ...]`
                    # We need to extract `[v0, v1, v2]` from this.
                    pv_faces_reshaped = self.pyvista_mesh.faces.reshape(-1, 4)[:, 1:] # Each row is now (3, i0, i1, i2)
                    
                    self.stl_mesh_data = mesh.Mesh(np.zeros(pv_faces_reshaped.shape[0], dtype=mesh.Mesh.dtype))
                    vertices_array = self.pyvista_mesh.points # numpy array of vertices
                    
                    for i, face_indices in enumerate(pv_faces_reshaped):
                        # Ensure indices are within bounds of vertices_array
                        if not (np.all(face_indices >= 0) and np.all(face_indices < len(vertices_array))):
                            raise IndexError(f"Face indices {face_indices} out of bounds for vertices array of size {len(vertices_array)}")
                        
                        self.stl_mesh_data.vectors[i] = [
                            vertices_array[face_indices[0]],
                            vertices_array[face_indices[1]],
                            vertices_array[face_indices[2]]
                        ]

                    # Validate the generated numpy-stl mesh too
                    stl_issues, stl_warnings = self.mesh_validator.validate_stl_structure(self.stl_mesh_data)
                    stl_report = ""
                    if stl_issues:
                        stl_report += "\n--- NumPy-STL Mesh Issues (for saving) ---\n" + "\n".join(stl_issues) + "\n"
                        QMessageBox.warning(self, "STL Conversion Warning", f"Issues detected when converting for saving: {'; '.join(stl_issues)}")
                    if stl_warnings:
                        stl_report += "\n--- NumPy-STL Mesh Warnings (for saving) ---\n" + "\n".join(stl_warnings) + "\n"
                        QMessageBox.information(self, "STL Conversion Information", f"Warnings detected when converting for saving: {'; '.join(stl_warnings)}")
                    
                    if stl_report:
                        self.validation_results.append(stl_report)

                except Exception as e:
                    QMessageBox.warning(self, "STL Conversion for Save Error", f"Could not create savable STL mesh from loaded file: {e}. Saving functionality may be limited.")
                    self.stl_mesh_data = None # Cannot create savable mesh
                    self.validation_results.append(f"\nCould not create savable STL mesh: {e}\n")

                self.apply_color_to_mesh('original') # Apply initial height-based coloring
                self.plotter.reset_camera()
                self.initial_camera_position = self.plotter.camera_position # Save camera position
                self.plotter.show_grid()

                self.status_label.setText(f'Loaded STL: {os.path.basename(file_path)}')
                self.save_stl_btn.setEnabled(True if self.stl_mesh_data is not None else False)
                self.save_screenshot_btn.setEnabled(True)
                self.reset_view_btn.setEnabled(True)
                self.validate_mesh_btn.setEnabled(True)
                
                self.color_combo_box.setEnabled(True)
                self.color_combo_box.setCurrentIndex(0) # Ensure 'Original' is selected
                self.select_custom_color_btn.setEnabled(False) # Disabled for 'Original'
                self.select_low_color_btn.setEnabled(True) # Enabled for 'Original'
                self.select_mid_color_btn.setEnabled(True)
                self.select_high_color_btn.setEnabled(True)

            except ValueError as ve:
                QMessageBox.critical(self, "Load STL Error", f"Failed to process STL mesh data. It might be malformed or empty. Details: {ve}")
                self.status_label.setText(f'Error loading STL: {ve}')
                self._reset_viewer_state() # Reset state on error
            except Exception as e:
                QMessageBox.critical(self, "Load STL Error", f"An error occurred while loading the STL file: {e}")
                self.status_label.setText(f'Error loading STL: {e}')
                self._reset_viewer_state() # Reset state on error

    def convert_image_to_stl(self):
        if not self.image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        # Disable buttons during processing
        self.convert_btn.setEnabled(False)
        self.load_image_btn.setEnabled(False)
        self.load_stl_btn.setEnabled(False)
        self.save_stl_btn.setEnabled(False)
        self.save_screenshot_btn.setEnabled(False)
        self.reset_view_btn.setEnabled(False)
        self.validate_mesh_btn.setEnabled(False)

        self.color_combo_box.setEnabled(False)
        self.select_custom_color_btn.setEnabled(False)
        self.select_low_color_btn.setEnabled(False)
        self.select_mid_color_btn.setEnabled(False)
        self.select_high_color_btn.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting conversion process...")
        self.validation_results.clear()

        try:
            # Gather all settings for the processing thread
            settings = {
                'max_height': float(self.max_height_input.text().strip()),
                'unit_per_pixel': float(self.scale_input.text().strip()),
                'invert_height': self.invert_height_checkbox.isChecked(),
                'gaussian_blur': self.gaussian_blur_input.value(),
                'edge_smoothing': self.edge_smoothing_checkbox.isChecked(),
                'max_dimension': self.max_dimension_input.value() if self.max_dimension_input.value() > 0 else float('inf'),
                'chunk_size': self.chunk_size_input.value(),
                'auto_gc': self.auto_gc_checkbox.isChecked(),
                'lod_threshold': self.lod_threshold_input.value()
            }

            self._reset_viewer_state() # Clear previous model from viewer

            self.processing_thread = ProcessingThread(self.image_path, settings)
            self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
            self.processing_thread.status_updated.connect(self.status_label.setText)
            self.processing_thread.processing_complete.connect(self.on_processing_complete)
            self.processing_thread.processing_error.connect(self.on_processing_error)
            self.processing_thread.start()

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Error in input values: {e}\nPlease ensure all numerical fields are valid.")
            self.status_label.setText(f'Error: {e}')
            self._reenable_ui_after_processing()
        except Exception as e:
            QMessageBox.critical(self, "Conversion Setup Error", f"An unexpected error occurred during conversion setup: {e}")
            self.status_label.setText(f'Error during setup: {e}')
            self._reenable_ui_after_processing()

    def on_processing_complete(self, stl_mesh_data, pyvista_mesh):
        self.stl_mesh_data = stl_mesh_data
        self.pyvista_mesh = pyvista_mesh
        self.loaded_stl_path = None # This was a conversion, not a loaded STL file

        # Perform comprehensive validation on the generated meshes
        pv_issues, pv_warnings = self.mesh_validator.validate_pyvista_mesh(self.pyvista_mesh)
        stl_issues, stl_warnings = self.mesh_validator.validate_stl_structure(self.stl_mesh_data)

        report = "--- Mesh Generation Report ---\n"
        if pv_issues or pv_warnings or stl_issues or stl_warnings:
            report += "Issues/Warnings detected:\n"
            if pv_issues: report += "PyVista Mesh Issues:\n" + "\n".join(pv_issues) + "\n"
            if pv_warnings: report += "PyVista Mesh Warnings:\n" + "\n".join(pv_warnings) + "\n"
            if stl_issues: report += "NumPy-STL Mesh Issues:\n" + "\n".join(stl_issues) + "\n"
            if stl_warnings: report += "NumPy-STL Mesh Warnings:\n" + "\n".join(stl_warnings) + "\n"
        else:
            report += "No significant issues or warnings detected. Mesh appears healthy.\n"

        self.validation_results.setText(report)
        QMessageBox.information(self, "Conversion Complete", "Image to STL conversion finished successfully! Check Diagnostics tab for mesh validation results.")

        # Display the mesh
        self.apply_color_to_mesh('original')
        self.plotter.reset_camera()
        self.initial_camera_position = self.plotter.camera_position
        self.plotter.show_grid()

        self.status_label.setText('Conversion complete! Mesh displayed. You can now save the STL.')
        self._reenable_ui_after_processing()
        self.save_stl_btn.setEnabled(True)
        self.save_screenshot_btn.setEnabled(True)
        self.reset_view_btn.setEnabled(True)
        self.validate_mesh_btn.setEnabled(True)
        
        self.color_combo_box.setEnabled(True)
        self.color_combo_box.setCurrentIndex(0) # Ensure 'Original' is selected
        self.select_custom_color_btn.setEnabled(False) # Disabled for 'Original'
        self.select_low_color_btn.setEnabled(True) # Enabled for 'Original'
        self.select_mid_color_btn.setEnabled(True)
        self.select_high_color_btn.setEnabled(True)

    def on_processing_error(self, message):
        QMessageBox.critical(self, "Conversion Error", f"An error occurred during conversion: {message}")
        self.status_label.setText(f'Conversion failed: {message}')
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self._reset_viewer_state() # Ensure a clean state after an error
        self._reenable_ui_after_processing()

    def _reenable_ui_after_processing(self):
        """Re-enables UI elements after a processing task completes or errors."""
        self.convert_btn.setEnabled(True) # Always allow re-conversion if image is loaded
        self.load_image_btn.setEnabled(True)
        self.load_stl_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        # Other buttons (save, reset, validate) are enabled based on mesh existence in on_processing_complete/load_stl

    def apply_color_to_mesh(self, color_type, custom_rgb=None):
        """
        Applies color to the current PyVista mesh based on the specified type.
        'original' for height-based gradient, 'solid' for a uniform color.
        """
        if self.pyvista_mesh is None:
            return

        # Remove existing actor if present to prevent overlapping meshes
        if self.current_mesh_actor:
            self.plotter.remove_actor(self.current_mesh_actor)
            self.current_mesh_actor = None # Clear the reference

        if color_type == 'original':
            # Re-calculate height-based colors using user-defined gradient colors
            if self.pyvista_mesh.points.size > 0:
                z_values = self.pyvista_mesh.points[:, 2]
                min_z = np.min(z_values)
                max_z = np.max(z_values)

                # Use instance variables for gradient colors (normalized 0-1)
                LOW_COLOR = np.array(self.low_height_color)
                MID_COLOR = np.array(self.mid_height_color)
                HIGH_COLOR = np.array(self.high_height_color)

                colors = np.zeros((len(z_values), 3)) # Initialize color array

                if max_z == min_z: # Flat surface
                    colors[:] = MID_COLOR
                else:
                    mid_z = (min_z + max_z) / 2.0
                    for i, height in enumerate(z_values):
                        if height <= mid_z:
                            if mid_z == min_z: # Avoid division by zero if lower half is flat
                                colors[i] = LOW_COLOR
                            else:
                                ratio = (height - min_z) / (mid_z - min_z)
                                colors[i] = LOW_COLOR * (1 - ratio) + MID_COLOR * ratio
                        else:
                            if max_z == mid_z: # Avoid division by zero if upper half is flat
                                colors[i] = HIGH_COLOR
                            else:
                                ratio = (height - mid_z) / (max_z - mid_z)
                                colors[i] = MID_COLOR * (1 - ratio) + HIGH_COLOR * ratio
                
                # PyVista expects 0-255 for `rgb=True` if scalars are used directly as colors
                # or 0-1 if mapped through a colormap. Here we're using RGB directly.
                # Let's keep 0-1 and let PyVista handle it.
                self.pyvista_mesh['colors'] = colors
                self.current_mesh_actor = self.plotter.add_mesh(self.pyvista_mesh, scalars='colors', rgb=True, show_edges=False)
            else:
                # If no points, add a default mesh or handle gracefully
                self.current_mesh_actor = self.plotter.add_mesh(self.pyvista_mesh, color=self.last_custom_color, show_edges=False)

        elif color_type == 'solid':
            color_to_apply = custom_rgb if custom_rgb is not None else self.last_custom_color
            # Set all point colors to the solid color (PyVista expects RGB 0-1 floats or 0-255 ints)
            # if self.pyvista_mesh.n_points > 0: # This check is redundant, pv_mesh.points handles it
            self.current_mesh_actor = self.plotter.add_mesh(self.pyvista_mesh, color=color_to_apply, show_edges=False)

        self.last_applied_color_type = color_type
        self.plotter.render() # Re-render the scene to show changes

    def on_color_selection_changed(self, index):
        """Handles changes in the color selection combo box."""
        selected_option = self.color_combo_box.currentText()
        if selected_option == "Original (Height-based)":
            self.apply_color_to_mesh('original')
            self.select_custom_color_btn.setEnabled(False)
            self.select_low_color_btn.setEnabled(True)
            self.select_mid_color_btn.setEnabled(True)
            self.select_high_color_btn.setEnabled(True)
        elif selected_option == "Solid Color":
            self.apply_color_to_mesh('solid', self.last_custom_color)
            self.select_custom_color_btn.setEnabled(True)
            self.select_low_color_btn.setEnabled(False)
            self.select_mid_color_btn.setEnabled(False)
            self.select_high_color_btn.setEnabled(False)

    def select_custom_color(self):
        """Opens a color dialog for the user to select a custom solid color."""
        # Convert stored RGB (0-1) to QColor (0-255) for dialog
        initial_color = QColor(int(self.last_custom_color[0]*255), int(self.last_custom_color[1]*255), int(self.last_custom_color[2]*255))
        color = QColorDialog.getColor(initial_color, self, "Select Solid Color")

        if color.isValid():
            new_rgb = (color.redF(), color.greenF(), color.blueF()) # Get RGB as floats 0-1
            self.last_custom_color = new_rgb
            # No need to set currentText if it's already "Solid Color"
            # Explicitly call apply_color_to_mesh for consistency
            self.apply_color_to_mesh('solid', new_rgb)


    def _select_gradient_color(self, color_level):
        """Helper to open color dialog for height gradient colors."""
        current_rgb = (0,0,0) # Default to black
        title = ""

        if color_level == 'low':
            current_rgb = self.low_height_color
            title = "Select Low Height Color"
        elif color_level == 'mid':
            current_rgb = self.mid_height_color
            title = "Select Mid Height Color"
        elif color_level == 'high':
            current_rgb = self.high_height_color
            title = "Select High Height Color"
        else:
            return

        initial_color = QColor(int(current_rgb[0]*255), int(current_rgb[1]*255), int(current_rgb[2]*255))
        color = QColorDialog.getColor(initial_color, self, title)

        if color.isValid():
            new_rgb = (color.redF(), color.greenF(), color.blueF())
            if color_level == 'low':
                self.low_height_color = new_rgb
            elif color_level == 'mid':
                self.mid_height_color = new_rgb
            elif color_level == 'high':
                self.high_height_color = new_rgb

            # If the current display mode is 'Original (Height-based)', re-apply colors
            if self.last_applied_color_type == 'original':
                self.apply_color_to_mesh('original')

    # Method to save the current STL mesh data to a file.
    def save_stl(self):
        if self.stl_mesh_data is None and self.pyvista_mesh is None:
            QMessageBox.warning(self, "No STL Data", "No STL data to save. Load an STL or convert an image first.")
            return

        file_dialog = QFileDialog()

        suggested_filename = "output.stl"
        if self.image_path and not self.loaded_stl_path: # Converted from image
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            suggested_filename = f"{base_name}_heightmap.stl"
        elif self.loaded_stl_path: # Loaded STL
            base_name = os.path.splitext(os.path.basename(self.loaded_stl_path))[0]
            suggested_filename = f"{base_name}_reprocessed.stl" # Suggest a reprocessed name
        
        file_path, _ = file_dialog.getSaveFileName(
            self, 'Save STL', suggested_filename,
            'STL Files (*.stl);;All Files (*.*)'
        )
        if file_path:
            try:
                # Prefer saving numpy-stl data if available as it's more direct for STL format
                if self.stl_mesh_data is not None:
                    self.stl_mesh_data.save(file_path)
                elif self.pyvista_mesh is not None:
                    # As a fallback, save using PyVista if numpy-stl object wasn't created (e.g., if loaded STL had issues)
                    self.pyvista_mesh.save(file_path)
                else:
                    raise ValueError("No mesh data available to save.")

                self.status_label.setText(f'STL saved to: {file_path}')
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"An error occurred while saving the STL file: {e}")
                self.status_label.setText(f'Error saving STL: {e}')

    def save_screenshot(self):
        if self.pyvista_mesh is None:
            QMessageBox.warning(self, "No Model", "No 3D model is currently displayed to take a screenshot.")
            return

        file_dialog = QFileDialog()
        suggested_filename = "screenshot.png"
        if self.image_path and not self.loaded_stl_path:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            suggested_filename = f"{base_name}_screenshot.png"
        elif self.loaded_stl_path:
            base_name = os.path.splitext(os.path.basename(self.loaded_stl_path))[0]
            suggested_filename = f"{base_name}_screenshot.png"

        file_path, _ = file_dialog.getSaveFileName(
            self, 'Save Screenshot', suggested_filename,
            'PNG Files (*.png);;All Files (*.*)'
        )

        if file_path:
            try:
                # Use PyVista's screenshot functionality
                self.plotter.screenshot(file_path)
                self.status_label.setText(f'Screenshot saved to: {file_path}')
            except Exception as e:
                QMessageBox.critical(self, "Screenshot Error", f"An error occurred while saving the screenshot: {e}")
                self.status_label.setText(f'Error saving screenshot: {e}')

    def reset_view(self):
        """Resets the 3D viewer camera to the initial position captured when the model was loaded/converted."""
        if self.plotter and self.pyvista_mesh: # Only reset if a mesh is actually displayed
            if self.initial_camera_position:
                self.plotter.camera_position = self.initial_camera_position
                self.status_label.setText('View reset to initial loaded position.')
            else:
                # Fallback to default reset if initial position wasn't captured
                self.plotter.reset_camera()
                self.status_label.setText('No initial position saved, reset to default view.')
        else:
            QMessageBox.warning(self, "No Model Displayed", "No 3D model is currently displayed to reset the view.")

    def update_performance_stats(self):
        """Updates the FPS and memory usage display in the status bar and diagnostics tab."""
        # Get FPS from PyVista's internal calculation
        fps = self.performance_monitor.get_average_fps()
        self.performance_label.setText(f'FPS: {fps:.1f}')

        # Get memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_mb = mem_info.rss / (1024 * 1024) # Resident Set Size in MB
            self.memory_label.setText(f'Memory: {memory_mb:.2f} MB')

            # Update diagnostics tab
            perf_text = f"FPS: {fps:.1f}\n"
            perf_text += f"Memory Usage (RSS): {memory_mb:.2f} MB\n"
            if self.pyvista_mesh:
                perf_text += f"PyVista Mesh Vertices: {self.pyvista_mesh.n_points:,}\n"
                perf_text += f"PyVista Mesh Faces: {self.pyvista_mesh.n_cells:,}\n"
            self.performance_display.setText(perf_text)

        except ImportError:
            self.memory_label.setText('Memory: N/A (psutil not installed)')
            self.performance_display.setText("Memory monitoring requires 'psutil' library.")
        except Exception as e:
            self.memory_label.setText(f'Memory: Error ({e})')
            self.performance_display.setText(f"Error getting memory info: {e}")

    def update_system_info(self):
        """Populates the system information in the diagnostics tab."""
        info_text = "--- System Information ---\n"
        info_text += f"Python Version: {sys.version.split(' ')[0]}\n"
        info_text += f"Platform: {sys.platform}\n"
        info_text += f"Operating System: {os.name}\n"
        
        try:
            import platform
            info_text += f"OS Name: {platform.system()} {platform.release()} ({platform.version()})\n"
            info_text += f"Machine: {platform.machine()}\n"
            info_text += f"Processor: {platform.processor()}\n"
            
            import psutil
            info_text += f"Total CPU Cores: {psutil.cpu_count(logical=True)} (Logical), {psutil.cpu_count(logical=False)} (Physical)\n"
            info_text += f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n"
        except ImportError:
            info_text += "Additional system info requires 'psutil' library.\n"
        except Exception as e:
            info_text += f"Error retrieving system info: {e}\n"

        self.system_info_display.setText(info_text)

    def validate_current_mesh(self):
        """Triggers validation for the currently loaded/converted mesh and displays results."""
        self.validation_results.clear()
        report_text = "--- Mesh Validation Results ---\n"

        if self.pyvista_mesh:
            pv_issues, pv_warnings = self.mesh_validator.validate_pyvista_mesh(self.pyvista_mesh)
            report_text += "\n--- PyVista Mesh Validation ---\n"
            if pv_issues:
                report_text += "Issues:\n" + "\n".join([f"- {issue}" for issue in pv_issues]) + "\n"
            if pv_warnings:
                report_text += "Warnings:\n" + "\n".join([f"- {warning}" for warning in pv_warnings]) + "\n"
            if not pv_issues and not pv_warnings:
                report_text += "PyVista mesh appears healthy.\n"
        else:
            report_text += "No PyVista mesh loaded for validation.\n"

        if self.stl_mesh_data:
            stl_issues, stl_warnings = self.mesh_validator.validate_stl_structure(self.stl_mesh_data)
            report_text += "\n--- NumPy-STL Mesh Validation (for saving) ---\n"
            if stl_issues:
                report_text += "Issues:\n" + "\n".join([f"- {issue}" for issue in stl_issues]) + "\n"
            if stl_warnings:
                report_text += "Warnings:\n" + "\n".join([f"- {warning}" for warning in stl_warnings]) + "\n"
            if not stl_issues and not stl_warnings:
                report_text += "NumPy-STL mesh appears healthy.\n"
        else:
            report_text += "No NumPy-STL mesh available for validation (may not be generated for loaded STLs).\n"

        if not self.pyvista_mesh and not self.stl_mesh_data:
            report_text = "No mesh currently loaded or converted to validate."
            
        self.validation_results.setText(report_text)


# This block ensures the code runs only when the script is executed directly.
if __name__ == '__main__':
    # Ensure HighDPI scaling is enabled for better appearance on high-res screens
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    ex = ImageToSTLApp()
    ex.show()
    sys.exit(app.exec_())