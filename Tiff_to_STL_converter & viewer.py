import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QLabel, QLineEdit, QCheckBox, QMessageBox, QFrame,
    QHBoxLayout, QSizePolicy, QComboBox, QColorDialog
)
from PyQt5.QtGui import QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt, QSize
from PIL import Image
import numpy as np
from stl import mesh

import pyvista as pv
from pyvistaqt import QtInteractor

# Define the main application class, inheriting from QMainWindow for a standard window structure.
class ImageToSTLApp(QMainWindow):
    # Constructor for the ImageToSTLApp class.
    def __init__(self):
        # Call the constructor of the parent class (QMainWindow).
        super().__init__()
        # Initialize variables to store the path of the loaded image,
        # the generated STL mesh data (using numpy-stl), and the PyVista mesh object.
        self.image_path = None
        self.stl_mesh_data = None  # Stores the mesh in numpy-stl format for saving
        self.pyvista_mesh = None   # Stores the mesh in PyVista format for visualization
        self.loaded_stl_path = None # Store the path of the loaded STL file
        self.initial_camera_position = None # Store the initial camera position
        self.current_mesh_actor = None # Store the PyVista actor for the current mesh

        # Initialize color-related variables
        self.last_applied_color_type = 'original' # 'original' for height-based, 'solid' for uniform color
        self.last_custom_color = (1.0, 0.843, 0.0) # Default gold color for solid mode (RGB normalized)

        # Default colors for height gradient (RGB normalized)
        self.low_height_color = (0.0, 0.0, 0.0)    # Black
        self.mid_height_color = (1.0, 0.843, 0.0) # Gold
        self.high_height_color = (1.0, 0.0, 0.0)  # Red


        # Initialize variables for the PyVista viewer widget and its plotter.
        self.viewer_widget = None
        self.plotter = None

        # Call the method to set up the user interface.
        self.init_ui()

    # Method to initialize and set up the graphical user interface.
    def init_ui(self):
        # Set the title of the main window.
        self.setWindowTitle('Image to STL Converter & Viewer')
        # Set the initial position and size of the window (x, y, width, height).
        self.setGeometry(100, 100, 1200, 900)
        # Set the window icon using a resource path.
        self.setWindowIcon(QIcon(self.get_resource_path('icon.png')))

        # Create a central widget that will hold all other UI elements.
        central_widget = QWidget()
        # Set the central widget for the QMainWindow.
        self.setCentralWidget(central_widget)

        # Create a horizontal layout for the main window to arrange controls and the viewer side by side.
        main_h_layout = QHBoxLayout()
        # Set the main layout for the central widget.
        central_widget.setLayout(main_h_layout)

        # Create a vertical layout for the control panel on the left side.
        controls_v_layout = QVBoxLayout()
        # Create a widget to encapsulate the controls layout.
        controls_widget = QWidget()
        controls_widget.setLayout(controls_v_layout)
        # Set a fixed width for the control panel.
        controls_widget.setFixedWidth(400)
        # Add the control panel widget to the main horizontal layout.
        main_h_layout.addWidget(controls_widget)

        # --- Image Loading/STL Loading Section ---
        # Create a frame for the image/STL loading section for visual grouping.
        image_section_frame = QFrame()
        image_section_frame.setFrameShape(QFrame.StyledPanel) # Give it a styled border.
        image_layout = QVBoxLayout()
        image_section_frame.setLayout(image_layout)
        controls_v_layout.addWidget(image_section_frame) # Add frame to controls layout.

        # Add a section title.
        image_layout.addWidget(QLabel("<h2>1. Load Image or STL</h2>"))

        # Button to load an existing STL file.
        self.load_stl_btn = QPushButton('Load Existing STL...')
        self.load_stl_btn.clicked.connect(self.load_stl) # Connect button click to load_stl method.
        image_layout.addWidget(self.load_stl_btn)

        # Button to browse for an image file for conversion.
        self.load_image_btn = QPushButton('Browse Image for Conversion...')
        self.load_image_btn.clicked.connect(self.load_image) # Connect button click to load_image method.
        image_layout.addWidget(self.load_image_btn)

        # Label to display the loaded image preview.
        self.image_display_label = QLabel('No image loaded.')
        self.image_display_label.setAlignment(Qt.AlignCenter) # Center the text/image.
        self.image_display_label.setFixedSize(250, 250) # Set a fixed size for the preview area.
        self.image_display_label.setScaledContents(True) # Scale image to fit the label.
        image_layout.addWidget(self.image_display_label, alignment=Qt.AlignCenter) # Center label in layout.

        # --- Conversion Settings Section ---
        # Create a frame for the conversion settings section.
        settings_section_frame = QFrame()
        settings_section_frame.setFrameShape(QFrame.StyledPanel)
        settings_layout = QVBoxLayout()
        settings_section_frame.setLayout(settings_layout)
        controls_v_layout.addWidget(settings_section_frame)

        # Add a section title.
        settings_layout.addWidget(QLabel("<h2>2. Adjust Conversion Settings</h2>"))

        # Layout for Max Height input.
        max_height_layout = QHBoxLayout()
        max_height_layout.addWidget(QLabel("Max Height (mm):"))
        self.max_height_input = QLineEdit('10.0') # Default value.
        self.max_height_input.setToolTip("Sets the maximum height of the 3D model.")
        max_height_layout.addWidget(self.max_height_input)
        settings_layout.addLayout(max_height_layout)

        # Layout for Pixels per Unit (scale) input.
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Pixels per Unit (mm):"))
        self.scale_input = QLineEdit('0.1') # Default value.
        self.scale_input.setToolTip("Determines the size of each pixel in millimeters (e.g., 0.1 means 10 pixels = 1mm).")
        scale_layout.addWidget(self.scale_input)
        settings_layout.addLayout(scale_layout)

        # Checkbox to invert height mapping.
        self.invert_height_checkbox = QCheckBox('Invert Height (darker pixels = taller)')
        self.invert_height_checkbox.setToolTip("Check this to make darker areas of the image appear taller in the 3D model.")
        settings_layout.addWidget(self.invert_height_checkbox)

        # --- Convert & Save Section ---
        # Create a frame for the actions section.
        actions_section_frame = QFrame()
        actions_section_frame.setFrameShape(QFrame.StyledPanel)
        actions_layout = QVBoxLayout()
        actions_section_frame.setLayout(actions_layout)
        controls_v_layout.addWidget(actions_section_frame)

        # Add a section title.
        actions_layout.addWidget(QLabel("<h2>3. Convert & Save</h2>"))
        # Button to initiate image to STL conversion.
        self.convert_btn = QPushButton('Convert Image to STL')
        self.convert_btn.clicked.connect(self.convert_image_to_stl) # Connect to conversion method.
        self.convert_btn.setEnabled(False) # Initially disabled until an image is loaded.
        actions_layout.addWidget(self.convert_btn)

        # Button to save the current STL file.
        self.save_stl_btn = QPushButton('Save Current STL File...')
        self.save_stl_btn.clicked.connect(self.save_stl) # Connect to save method.
        self.save_stl_btn.setEnabled(False) # Initially disabled until a mesh is generated/loaded.
        actions_layout.addWidget(self.save_stl_btn)

        # Button to save a screenshot of the 3D model
        self.save_screenshot_btn = QPushButton('Save Screenshot (PNG)')
        self.save_screenshot_btn.clicked.connect(self.save_screenshot)
        self.save_screenshot_btn.setEnabled(False) # Initially disabled
        actions_layout.addWidget(self.save_screenshot_btn)

        # Button to reset the view
        self.reset_view_btn = QPushButton('Reset View to Initial Position')
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setEnabled(False) # Initially disabled
        actions_layout.addWidget(self.reset_view_btn)

        # --- Color Settings Section ---
        color_section_frame = QFrame()
        color_section_frame.setFrameShape(QFrame.StyledPanel)
        color_layout = QVBoxLayout()
        color_section_frame.setLayout(color_layout)
        controls_v_layout.addWidget(color_section_frame)

        color_layout.addWidget(QLabel("<h2>4. Change Model Color</h2>"))

        self.color_combo_box = QComboBox()
        self.color_combo_box.addItem("Original (Height-based)")
        self.color_combo_box.addItem("Solid Color")
        self.color_combo_box.currentIndexChanged.connect(self.on_color_selection_changed)
        self.color_combo_box.setEnabled(False) # Disable until model is loaded
        color_layout.addWidget(self.color_combo_box)

        self.select_custom_color_btn = QPushButton('Select Custom Solid Color...')
        self.select_custom_color_btn.clicked.connect(self.select_custom_color)
        self.select_custom_color_btn.setEnabled(False) # Disable until model is loaded
        color_layout.addWidget(self.select_custom_color_btn)

        # New: Height Gradient Color Selection
        color_layout.addWidget(QLabel("<br><b>Height Gradient Colors:</b>"))

        self.select_low_color_btn = QPushButton('Select Low Height Color (e.g., Black)')
        self.select_low_color_btn.clicked.connect(lambda: self._select_gradient_color('low'))
        self.select_low_color_btn.setEnabled(False) # Enabled with "Original" mode
        color_layout.addWidget(self.select_low_color_btn)

        self.select_mid_color_btn = QPushButton('Select Mid Height Color (e.g., Gold)')
        self.select_mid_color_btn.clicked.connect(lambda: self._select_gradient_color('mid'))
        self.select_mid_color_btn.setEnabled(False) # Enabled with "Original" mode
        color_layout.addWidget(self.select_mid_color_btn)

        self.select_high_color_btn = QPushButton('Select High Height Color (e.g., Red)')
        self.select_high_color_btn.clicked.connect(lambda: self._select_gradient_color('high'))
        self.select_high_color_btn.setEnabled(False) # Enabled with "Original" mode
        color_layout.addWidget(self.select_high_color_btn)


        # Add a stretch to push all control elements to the top.
        controls_v_layout.addStretch()

        # --- 3D Viewer Section ---
        # Create a PyVistaQt interactor widget for 3D visualization.
        self.viewer_widget = QtInteractor(self)
        # Get the PyVista plotter object from the interactor.
        self.plotter = self.viewer_widget.interactor
        # Set the background color of the 3D viewer.
        self.plotter.set_background('white')
        # Show coordinate axes.
        self.plotter.show_axes()
        # Show a grid on the plane.
        self.plotter.show_grid()
        # Add the 3D viewer widget to the main horizontal layout.
        main_h_layout.addWidget(self.viewer_widget)

        # --- Status Bar ---
        # Create a label for the status bar to display messages to the user.
        self.status_label = QLabel('Ready: Load an image or an STL file.')
        # Add the label to the QMainWindow's status bar.
        self.statusBar().addWidget(self.status_label)

    # Helper method to get resource paths, especially useful for PyInstaller bundles.
    def get_resource_path(self, relative_path):
        try:
            # If running as a bundled executable, sys._MEIPASS holds the path to temp files.
            base_path = sys._MEIPASS
        except Exception:
            # If running as a script, use the current directory.
            base_path = os.path.abspath(".")
        # Join the base path with the relative path to the resource.
        return os.path.join(base_path, relative_path)

    def _reset_viewer_state(self):
        """Resets all viewer-related internal states and UI elements."""
        self.stl_mesh_data = None
        self.pyvista_mesh = None
        self.loaded_stl_path = None
        self.initial_camera_position = None
        self.current_mesh_actor = None
        self.plotter.clear()
        self.plotter.reset_camera()
        
        self.convert_btn.setEnabled(False)
        self.save_stl_btn.setEnabled(False)
        self.save_screenshot_btn.setEnabled(False)
        self.reset_view_btn.setEnabled(False)
        
        self.color_combo_box.setEnabled(False)
        self.select_custom_color_btn.setEnabled(False)
        self.select_low_color_btn.setEnabled(False) # Disable gradient color buttons
        self.select_mid_color_btn.setEnabled(False)
        self.select_high_color_btn.setEnabled(False)
        self.color_combo_box.setCurrentIndex(0) # Reset to 'Original'

    # Method to handle loading an image file.
    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Open Image', '',
            'Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff)'
        )
        if file_path:
            self.image_path = file_path
            self.status_label.setText(f'Loaded Image: {os.path.basename(file_path)}')
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(self.image_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display_label.setPixmap(scaled_pixmap)

            self._reset_viewer_state() # Reset state for a new image conversion
            self.convert_btn.setEnabled(True) # Enable convert button specifically for image

    # Method to handle loading an existing STL file.
    def load_stl(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Open STL File', '',
            'STL Files (*.stl)'
        )
        if file_path:
            try:
                self._reset_viewer_state() # Reset state before loading new STL

                self.pyvista_mesh = pv.read(file_path)
                self.loaded_stl_path = file_path

                if not self.pyvista_mesh.faces.any() or self.pyvista_mesh.n_cells == 0:
                    raise ValueError("Loaded STL has no faces or cells.")
                # PyVista's face array usually has 4 elements per triangle (3, idx1, idx2, idx3)
                if self.pyvista_mesh.faces.shape[0] % 4 != 0:
                     QMessageBox.warning(self, "STL Format Warning", "The loaded STL's face data might be in an unusual format. Visualization might be affected.")

                # Convert PyVista mesh data to numpy-stl Mesh format for saving
                # This conversion logic should be robust enough for valid PyVista PolyData
                faces = self.pyvista_mesh.faces.reshape(-1, 4)[:, 1:]
                vertices = self.pyvista_mesh.points

                if faces.size > 0 and vertices.size > 0:
                    self.stl_mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                    for i, f in enumerate(faces):
                        if not (np.all(f >= 0) and np.all(f < len(vertices))):
                            raise ValueError(f"Face indices {f} out of bounds for vertices array of size {len(vertices)}.")
                        self.stl_mesh_data.vectors[i][0] = vertices[f[0]]
                        self.stl_mesh_data.vectors[i][1] = vertices[f[1]]
                        self.stl_mesh_data.vectors[i][2] = vertices[f[2]]
                else:
                    self.stl_mesh_data = None


                self.apply_color_to_mesh('original') # Apply initial height-based coloring
                self.plotter.reset_camera()
                self.initial_camera_position = self.plotter.camera_position
                self.plotter.show_grid()

                self.status_label.setText(f'Loaded STL: {os.path.basename(file_path)}')
                self.save_stl_btn.setEnabled(True if self.stl_mesh_data is not None else False)
                self.save_screenshot_btn.setEnabled(True)
                self.reset_view_btn.setEnabled(True)
                
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


    # Method to convert the loaded image into an STL mesh.
    def convert_image_to_stl(self):
        if not self.image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            max_height = float(self.max_height_input.text().strip())
            unit_per_pixel = float(self.scale_input.text().strip())
            invert_height = self.invert_height_checkbox.isChecked()

            img = Image.open(self.image_path).convert('L')
            img_array = np.array(img)

            height_map = img_array / 255.0 * max_height
            if invert_height:
                height_map = max_height - height_map

            rows, cols = height_map.shape

            vertices = []
            for r in range(rows):
                for c in range(cols):
                    x = c * unit_per_pixel
                    y = r * unit_per_pixel
                    z = height_map[r, c]
                    vertices.append([x, y, z])
            vertices = np.array(vertices)

            faces = []
            for r in range(rows - 1):
                for c in range(cols - 1):
                    p1 = r * cols + c
                    p2 = r * cols + (c + 1)
                    p3 = (r + 1) * cols + c
                    p4 = (r + 1) * cols + (c + 1)

                    faces.append([p1, p3, p2])
                    faces.append([p2, p3, p4])
            faces = np.array(faces)

            # Ensure vertices and faces are not empty before creating PyVista mesh
            if vertices.size == 0 or faces.size == 0:
                raise ValueError("Image conversion resulted in no vertices or faces.")

            pyvista_faces = np.hstack((np.full((len(faces), 1), 3), faces)).flatten()
            self.pyvista_mesh = pv.PolyData(vertices, pyvista_faces)

            # Create a numpy-stl Mesh object for saving
            self.stl_mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                self.stl_mesh_data.vectors[i][0] = vertices[f[0]]
                self.stl_mesh_data.vectors[i][1] = vertices[f[1]]
                self.stl_mesh_data.vectors[i][2] = vertices[f[2]]

            self.apply_color_to_mesh('original') # Apply initial height-based coloring
            self.plotter.reset_camera()
            self.initial_camera_position = self.plotter.camera_position
            self.plotter.show_grid()

            self.status_label.setText('Conversion complete! Mesh displayed in viewer. You can now save the STL.')
            self.save_stl_btn.setEnabled(True)
            self.save_screenshot_btn.setEnabled(True)
            self.reset_view_btn.setEnabled(True)
            
            self.color_combo_box.setEnabled(True)
            self.color_combo_box.setCurrentIndex(0) # Ensure 'Original' is selected
            self.select_custom_color_btn.setEnabled(False) # Disabled for 'Original'
            self.select_low_color_btn.setEnabled(True) # Enabled for 'Original'
            self.select_mid_color_btn.setEnabled(True)
            self.select_high_color_btn.setEnabled(True)

            self.convert_btn.setEnabled(True) # Keep enabled for re-conversion

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Error in input or conversion: {e}\nPlease enter valid numerical values for height and scale.")
            self.status_label.setText(f'Error: {e}')
            self._reset_viewer_state()
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"An unexpected error occurred during conversion: {e}")
            self.status_label.setText(f'Error during conversion: {e}')
            self._reset_viewer_state()

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

                # Use instance variables for gradient colors
                LOW_COLOR = np.array(self.low_height_color)
                MID_COLOR = np.array(self.mid_height_color)
                HIGH_COLOR = np.array(self.high_height_color)

                if max_z == min_z:
                    colors = np.array([MID_COLOR] * len(z_values)) # If flat, use mid color
                else:
                    mid_z = (min_z + max_z) / 2
                    colors = []
                    for height in z_values:
                        if height < mid_z:
                            if mid_z == min_z: # Avoid division by zero if range is zero
                                color = MID_COLOR
                            else:
                                ratio = (height - min_z) / (mid_z - min_z)
                                color = LOW_COLOR * (1 - ratio) + MID_COLOR * ratio
                        else:
                            if max_z == mid_z: # Avoid division by zero if range is zero
                                color = MID_COLOR
                            else:
                                ratio = (height - mid_z) / (max_z - mid_z)
                                color = MID_COLOR * (1 - ratio) + HIGH_COLOR * ratio
                        colors.append(color)
                    colors = np.array(colors)

                self.pyvista_mesh['colors'] = colors
                self.current_mesh_actor = self.plotter.add_mesh(self.pyvista_mesh, scalars='colors', rgb=True, show_edges=False)
            else:
                # If no points, add a default mesh or handle gracefully
                self.current_mesh_actor = self.plotter.add_mesh(self.pyvista_mesh, color=self.last_custom_color, show_edges=False)

        elif color_type == 'solid':
            color_to_apply = custom_rgb if custom_rgb is not None else self.last_custom_color
            # Set all point colors to the solid color
            if self.pyvista_mesh.n_points > 0:
                self.pyvista_mesh['colors'] = np.array([color_to_apply] * self.pyvista_mesh.n_points)
                self.current_mesh_actor = self.plotter.add_mesh(self.pyvista_mesh, scalars='colors', rgb=True, show_edges=False)
            else:
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
        initial_color = QColor(int(self.last_custom_color[0]*255), int(self.last_custom_color[1]*255), int(self.last_custom_color[2]*255))
        color = QColorDialog.getColor(initial_color, self, "Select Solid Color")

        if color.isValid():
            new_rgb = (color.redF(), color.greenF(), color.blueF())
            self.last_custom_color = new_rgb
            self.color_combo_box.setCurrentText("Solid Color") # This will trigger on_color_selection_changed if not already solid
            if self.last_applied_color_type == 'solid': # Only re-apply if currently in solid mode
                self.apply_color_to_mesh('solid', new_rgb)

    def _select_gradient_color(self, color_level):
        """Helper to open color dialog for height gradient colors."""
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
        if self.image_path:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            suggested_filename = f"{base_name}_heightmap.stl"
        elif self.loaded_stl_path:
            base_name = os.path.splitext(os.path.basename(self.loaded_stl_path))[0]
            suggested_filename = f"{base_name}_loaded.stl"

        file_path, _ = file_dialog.getSaveFileName(
            self, 'Save STL', suggested_filename,
            'STL Files (*.stl)'
        )
        if file_path:
            try:
                if self.stl_mesh_data is not None:
                    self.stl_mesh_data.save(file_path)
                elif self.pyvista_mesh is not None:
                    # If for some reason stl_mesh_data is None but pyvista_mesh exists
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
        if self.image_path:
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
        if self.plotter:
            if self.initial_camera_position:
                self.plotter.camera_position = self.initial_camera_position
                self.status_label.setText('View reset to initial loaded position.')
            else:
                QMessageBox.information(self, "No Initial View",
                                        "No specific initial camera position was saved for this model. "
                                        "Clicking 'Load Existing STL...' or 'Convert Image to STL' "
                                        "will set a new initial position.")
                self.status_label.setText('No initial position saved.')
        else:
            QMessageBox.warning(self, "No Viewer", "3D viewer not initialized.")


# This block ensures the code runs only when the script is executed directly.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageToSTLApp()
    ex.show()
    sys.exit(app.exec_())