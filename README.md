# `Tiff_to_STL_converter & viewer.py`
* **Purpose:** This Python application provides a user-friendly Graphical User Interface (GUI) for converting 2D images (like TIFFs, PNGs, JPEGs) into 3D STL models based on heightmap data. It also includes a built-in 3D viewer to visualise both converted models and existing STL files.


* **Key Features:**
    * **Image to STL Conversion:**  Transform grayscale images into 3D heightmap models. Users can choose whether darker or lighter pixels correspond to taller features
    * **STL File Viewer:** Load and inspect existing .stl files directly within the application.
    * **Configurable Conversion Settings:** Allows setting of voltage, current, and stop threshold; supports above/below stop condition.
         * **Max Height (mm):** Define the maximum Z-height of the generated 3D model.
         * **Pixels per Unit (mm):** Control the real-world scale of the 3D model relative to image pixels.
         * **Invert Height:*** Option to invert the height mapping, making darker pixels taller instead of brighter ones.  
    * **Real-time 3D Visualisation:** Utilizes PyVista for interactive 3D rendering of the converted or loaded STL models.
    * **Automatic Z-Height Colouring:** Models are automatically coloured in the viewer based on their Z-axis height, using a gradient from black (low) to white (high) with a gold midpoint.
    * **Save Functionality:"" Save the generated or loaded 3D models as new .stl files.
* **Usage:**
    1.  Ensure all dependencies are installed.
    2.  Run the script: `Tiff_to_STL_converter & viewer.py"`
    3.  **Load Image or STL:**
        * Click the "Browse Image for Conversion..." button to select a 2D image file (e.g., .png, .jpg, .tif). A preview of the selected image will appear
        * Alternatively, click "Load Existing STL..." to open and display a pre-existing .stl file directly in the 3D viewer 
    4.  **Adjust Conversion Settings (for image conversion):**
        * Max Height (mm): Input the desired maximum height for the 3D model in millimetres
        * Pixels per Unit (mm): Specify the real-world dimension that each pixel in your image represents (e.g., 0.1 means 10 pixels will equate to 1 millimetre in the 3D model).
        * Invert Height: Check this box if you want darker areas of your image to correspond to taller features in the generated 3D model 
    5.  **Convert Image to STL:** After loading an image and configuring the settings, click "Convert Image to STL". The application will process the image and render the resulting 3D model in the integrated viewer
    6.  Save STL File: Once a model is displayed in the viewer (either newly converted or loaded), click "Save Current STL File" to save it to your chosen location on your system 
    7.   Click "Stop Logging" to end the experiment. Data will be exported according to the selected format and setting
    8.    3D Viewer Interaction: Use your mouse to freely rotate, pan, and zoom the 3D model within the viewer.

* **Dependencies:**
    * `PyQt5:` The framework used for building the graphical user interface.
    * `Pillow` (PIL) For opening and processing various image formats
    * `numpy`  Essential for numerical operations and array manipulations, especially for heightmap and mesh data.
    * `numpy-stl`  For creating and saving STL mesh files.
    * `pyvista` A powerful library for 3D plotting and mesh analysis, used for rendering the STL models.
    * `pyvistaqt` Integrates PyVista's 3D plotting capabilities with PyQt5 applications.
```bash
pip install PyQt5 Pillow numpy numpy-stl pyvista pyvistaqt
```
