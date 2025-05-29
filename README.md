# Rigol Power Supply Automation and Logging Script

This repository contains Python scripts designed for use in manufacturing tip etching, specifically for controlling a Rigol DP811A power supply (though the core logic can be adapted for other VISA-compatible instruments).

## Scripts

### 1. Power_Supply_Stopper.py

This script provides a comprehensive Tkinter-based graphical user interface (GUI) for controlling a programmable power supply. It includes advanced features such as preset management, configurable export formats, detailed logging, and an interactive user experience for electrochemical etching experiments.

## Key Features

- **GUI Interface:** User-friendly interface built with Tkinter and ttkthemes
- **VISA Resource Scanning:** Automatically scans for and lists available VISA instruments
- **Configurable Parameters:** Allows setting of voltage, current, and stop threshold; supports above or below stop conditions
- **Real-time Plotting:** Live plots for Voltage, Current, Power, and Resistance with configurable styles
- **Data Logging & Export:**
  - Logs all data points in memory during the session
  - Exports data to CSV, XLSX (Excel), or JSON formats
  - Includes configuration settings and user notes in exported files
- **Preset Management:** Save and load configurations for different experimental setups
- **Simulation Mode:** Enables running the application without a physical instrument for testing or demonstration
- **Notes Section:** Add and save experimental notes, which are included in data exports
- **Electrochemical Cell Information:** Fields for Anode, Cathode, Electrolyte, and Molarity details, saved with the data
- **Status & Log Tabs:** Provides real-time status updates and a detailed event log
- **Customizable Appearance:** Selectable GUI themes and plot styles

## Installation

### Dependencies

Install all required dependencies using pip:

```bash
pip install pyvisa pandas openpyxl numpy matplotlib ttkthemes Pillow psutil zeroconf pyvisa-py
```

**Note:** `tkinter` and `winsound` are usually included with Python installations.

### Required Dependencies:
- `tkinter` (usually part of Python's standard library)
- `pyvisa` (for instrument communication)
- `pandas` (for XLSX and JSON export)
- `openpyxl` (for XLSX export, used by pandas)
- `numpy` (for numerical operations, especially in plotting)
- `matplotlib` (for plotting)

### Optional Dependencies:
- `ttkthemes` (for enhanced GUI styling; falls back to default Tkinter styles if not found)
- `Pillow (PIL)` (for using .ico window icons; falls back if not found)
- `winsound` (for beep sound on Windows when threshold is met; script runs on other OS without it)
- `psutil` (System monitoring, not directly used in the provided script but listed in original dependencies)
- `zeroconf` (For auto-discovery of networked instruments, not directly used in the provided script but listed in original dependencies)
- `pyvisa-py` (Backend for pyvisa, pure Python implementation, no NI-VISA needed)

## Usage

1. Ensure all dependencies are installed
2. Run the script: 
   ```bash
   python "Power Supply Stopper.py"
   ```

### Control Tab:
- Scan for VISA resources
- Set desired Voltage, Current, and Stop Threshold
- Choose the save location and export format (CSV, XLSX, JSON, or All)
- Fill in electrochemical cell information as needed

### Settings Tab:
- Configure plot update interval, max plot points, GUI theme, and plot style
- Enable/disable simulation mode
- Set the stop condition (current below or above threshold)
- Manage presets (save, load, delete)

### Notes Tab:
- Add any relevant experimental notes, which will be saved with the configuration and included in data exports

### Operation:
1. Click "Start Logging" - The application will switch to the "Plots" tab
2. Click "Stop Logging" to end the experiment. Data will be exported according to the selected format and settings
3. The "Log" tab shows a history of operations and events

## Code Architecture

### Core Components

#### 1. Imports and Third-Party Libraries
The script uses try-except blocks to handle optional dependencies gracefully, importing:
- **Standard Library Modules:** `tkinter`, `messagebox`, `filedialog`, `ttk`, `StringVar`, `BooleanVar`, `os`, `sys`, `json`, `csv`, `queue`, `threading`, `time`, `traceback`, `datetime`, `collections.deque`, `concurrent.futures.ThreadPoolExecutor`, and `typing`
- **Third-Party Libraries:** `winsound`, `pyvisa`, `pandas`, `openpyxl`, `numpy`, `matplotlib`, `ttkthemes`, `Pillow (PIL)`

#### 2. Constants
Comprehensive set of constants defining application parameters, file names, UI settings, and operational modes.

#### 3. Tooltip Class
Helper class creating transient pop-up windows with descriptive text when hovering over widgets.

#### 4. ConfigManager Class
Manages persistent storage and retrieval of application settings and user-defined presets:
- **Initialisation:** Sets up default configuration values and loads existing settings
- **Configuration Management:** `load_config()` and `save_config()` methods
- **Preset Management:** Methods for saving, loading, and managing groups of settings
- **Notes Management:** Handles user notes stored within the main configuration

#### 5. DataManager Class
Efficiently manages collected data points using `collections.deque` for optimised real-time plotting performance.

#### 6. DataLogger Class
Critical component for power supply interaction and data acquisition:
- **Connection Management:** Establishes a VISA connection or simulation mode
- **Data Acquisition:** Runs in a separate thread for GUI responsiveness
- **Export Functions:** Static methods for CSV, Excel, and JSON export with embedded metadata

#### 7. PowerLoggerApp Class
Main Tkinter application coordinating GUI and all other components:
- **Multi-tab Interface:** Control, Settings, Plots, Log, Notes, and About tabs
- **Real-time Plotting:** Integrated matplotlib with animation
- **Event Handling:** User interactions and application logic
- **Threading:** Maintains GUI responsiveness during data acquisition
- **Theme Management:** Dynamic GUI and plot style changes

## Screenshots

*Note: Success is assured only with the RIGOL DP 811A power supply, as it was the only one available for testing.*

- Power Supply Control Tab 1
- Power Supply Control Tab 2  
- Power Supply Settings Tab
- Real-time Plotting

## Limitations

When decreasing the logging time down to 10ms, plotting time increases, but this does not affect the logging or export process for JSON, CSV, or XLSX formats.

## Experimental Notes

These programs are beneficial for electrochemical etching projects. Below are some successful parameters found during testing:

### Nickel Etching
- **Constant Voltage:** 0.5 M HCl, 4V, 0.5 amps, with a threshold of 0.09 amps
  - *Result:* Sub-100nm radius of curvature achieved
- **Constant Current:** 0.5 M HCl, 12V, 0.1 amps, with a Voltage limit of 6 Volts

### Tungsten Etching
- **Constant Voltage:** 2 M NaOH, 9V, 1 amp with a threshold of 0.032 amps
  - *Result:* Sub-100nm radius of curvature achieved
- **Constant Current:** 2 M NaOH, 15V, 0.1 amp with a voltage limit of 12.004 Volts

**Recommendation:** Further experimentation with the Constant Current setting is recommended, as it showed potential for producing smoother tips.

## Final Note

Don't get better tips than me!
