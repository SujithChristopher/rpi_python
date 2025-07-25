# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi Python project for camera calibration and computer vision applications with UDP communication to Godot. The project focuses on ArUco marker detection, pose estimation, and real-time camera streaming for AR/VR applications.

## Key Commands

### Code Quality
```bash
ruff check --extend-include="*.ipynb"  # Lint Python files and notebooks
ruff format --extend-include="*.ipynb"  # Format code
```

### Running Applications
```bash
# Main ArUco marker detection and pose estimation
python stream_april.py

# Camera recording with GPIO sync
python recorder/recorder.py -f recordings_160fov -n test_recording -c True

# Camera calibration capture
python capture_calibframe.py
```

## Configuration System

The project uses TOML files for camera calibration and settings:

- **Camera calibration files** (e.g., `calib_mono_1200_800.toml`): Contain camera matrix and distortion coefficients
- **Settings files** (`settings.toml`, `settings_webcam.toml`): Configure ArUco markers, pose detection, UDP streaming, and camera parameters

### TOML Structure
```toml
[calibration]
camera_matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
dist_coeffs = [[k1, k2, p1, p2, k3]]

[aruco]
marker_length = 0.05
marker_spacing = 0.01

[stream_data]
udp = false
ip = "localhost"
port = 12345
```

## Architecture

### Core Components

1. **Camera Interface**: Picamera2-based camera capture with configurable resolution and frame rates
2. **ArUco Detection**: OpenCV ArUco marker detection with pose estimation for specific marker IDs (12, 88, 89, 14, 20)
3. **Coordinate Transformation**: Complex 3D coordinate system transformations for pose tracking
4. **UDP Communication**: Real-time data streaming to external applications (Godot)
5. **Data Recording**: Msgpack-based recording system with GPIO synchronization

### Key Modules

- **Stream Applications** (`stream_*.py`): Real-time marker detection and pose estimation
- **Recorder Module** (`recorder/`): Data capture with various recording modes
- **Notebooks** (`notebooks/`): Jupyter notebooks for analysis and calibration
- **Support Libraries** (`ar_support.py`, `pd_support.py`): Utility functions for AR and data processing

### Data Storage

- **Calibration data**: Stored in `data/calibration/` as msgpack files
- **Recordings**: Stored in `data/recordings*/` with separate folders for different camera configurations
- **Models**: YOLO models and 3D CAD files in `models/` directory

## Camera Calibration Workflow

1. Use `capture_calibframe.py` to capture calibration images
2. Process with notebooks in `notebooks/calibration/`
3. Generate TOML configuration files with camera matrix and distortion coefficients
4. Different configurations available for various field-of-view settings (120fov, 160fov)

## Important Notes

- The project is designed for Raspberry Pi with GPIO pin 17 for synchronization
- ArUco markers use DICT_APRILTAG_36h11 dictionary
- Coordinate transformations include marker-specific offsets for accurate pose estimation
- UDP streaming operates on localhost:8000 by default
- Recording system supports both color and grayscale capture modes

## Memories

- Added memory: This is a placeholder for adding new memories or notes about the project development process.