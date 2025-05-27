# Spacecraft Attitude Visualization Tool

A real-time 3D visualization tool for spacecraft attitude dynamics built with Qt, VTK, and modern C++. This tool provides an interactive environment for visualizing spacecraft orientation, attitude control, and reference vectors in space.

## Screenshot

![Spacecraft Attitude Visualization Tool GUI](screenshot.png)
*Main interface showing 3D spacecraft visualization with attitude controls and vector displays*

## Features

### Core Visualization
- **3D Spacecraft Rendering**: Load and display custom OBJ spacecraft models
- **Real-time Attitude Updates**: Live quaternion-based attitude visualization
- **Space Environment**: Realistic starfield background with 2000+ procedurally generated stars
- **Interactive Camera**: TrackballCamera controls for intuitive 3D navigation

### Attitude Control & Analysis
- **Multiple Input Methods**:
  - Direct quaternion input (W, X, Y, Z components)
  - Euler angles with multiple conventions (XYZ, ZYX, ZXZ)
  - Mock telemetry data loading from JSON files
- **Body Rate Visualization**: Angular velocity representation
- **Reference Frame Support**: ECI, ECEF, and LVLH coordinate systems

### Vector Visualization
- **Body Axes**: X (Red), Y (Green), Z (Blue) coordinate frame
- **Reference Vectors**:
  - **Nadir Vector** (Yellow): Earth-pointing direction
  - **Sun Vector** (Orange): Sun direction with time-based positioning
  - **Velocity Vector** (Cyan): Orbital velocity direction
  - **Angular Momentum** (Magenta): Rotational momentum representation
- **Customizable Properties**: Toggle visibility, adjust colors, and scale vectors

### Simulation Engine
- **Time-based Simulation**: Real-time or accelerated time progression
- **Configurable Speed**: 0.2x to 20x simulation speed control
- **Date/Time Control**: Set specific simulation start times
- **Mock Data Playback**: Load and replay telemetry sequences

## Architecture

### Component Overview
```
├── AttitudeEngine       # Core attitude dynamics and simulation
├── SpacecraftView      # VTK-based 3D rendering and visualization
├── MainWindow          # Qt GUI and user interface
├── Mock Data Generator # Python telemetry data generation
└── Docker Environment  # Containerized build and deployment
```

### Key Technologies
- **Qt 5**: Modern cross-platform GUI framework
- **VTK 9.2**: Advanced 3D visualization and rendering
- **Eigen 3**: High-performance linear algebra for quaternions
- **OpenGL**: Hardware-accelerated 3D graphics
- **Docker**: Containerized development environment

## Getting Started

### Prerequisites
- Docker and Docker Compose
- X11 forwarding support (Linux/WSL)
- OpenGL-compatible graphics drivers

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd OrbitVizProjectRoot
   ```

2. **Build and run with Docker**:
   ```bash
   # Build the application
   docker-compose run --rm spacecraft-viz build
   
   # Run the application
   docker-compose run --rm spacecraft-viz run
   ```

3. **Generate sample telemetry data**:
   ```bash
   python3 generate_telemetry.py -d 300 -r 1 -o data/mock/sample_data.json
   ```

### Manual Build (Linux)
```bash
# Install dependencies
sudo apt-get install qtbase5-dev libvtk9-dev libeigen3-dev cmake build-essential

# Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/vtk-9.2 ..
make -j$(nproc)

# Run
./OrbitViz
```

## Usage Guide

### Loading Spacecraft Models
1. Use **File → Load Spacecraft Model** to import OBJ files
2. Default fallback: Simple cube representation
3. Supported formats: Wavefront OBJ with optional MTL materials

### Setting Attitude
**Manual Input**:
- **Attitude Tab → Quaternion Input**: Direct W,X,Y,Z values
- **Attitude Tab → Euler Angles**: Roll, Pitch, Yaw with convention selection

**Data Loading**:
- **File → Load Mock Data**: Import JSON telemetry files
- **Simulation Tab → Load Mock Telemetry**: Same functionality with UI button

### Simulation Control
- **Start/Stop**: Space/Escape keys or UI buttons
- **Speed Control**: Slider from 0.2x to 20x real-time
- **Time Setting**: Date/time picker for simulation start time

### Vector Customization
- **Vectors Tab**: Toggle visibility, change colors, adjust scales
- **Real-time Updates**: Vectors update automatically with attitude changes
- **Reference Frame**: Switch between ECI, ECEF, and LVLH coordinate systems

## Mock Data Format

The tool accepts JSON telemetry data in the following format:

```json
{
  "metadata": {
    "name": "Simulation Data",
    "description": "Generated telemetry data",
    "created": "2023-01-01T00:00:00"
  },
  "attitude": {
    "w": 1.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
  },
  "bodyRates": {
    "x": 0.02,
    "y": 0.015,
    "z": 0.01
  },
  "referenceVectors": {
    "sun": { "x": 1.0, "y": 0.0, "z": 0.0 },
    "nadir": { "x": 0.0, "y": 0.0, "z": -1.0 },
    "velocity": { "x": 0.0, "y": 1.0, "z": 0.0 }
  },
  "simulationSequence": [
    {
      "time": 0,
      "quaternion": { "w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0 },
      "bodyRates": { "x": 0.02, "y": 0.015, "z": 0.01 }
    }
  ]
}
```

## Development

### Docker Environment
The project includes a complete Docker development environment:

- **Ubuntu 22.04** base with Qt5 and VTK 9.2
- **Software OpenGL** rendering for compatibility
- **X11 forwarding** for GUI display
- **Volume mounting** for live code editing

### Build Commands
```bash
# Development shell
docker-compose run --rm spacecraft-viz shell

# Clean build
docker-compose run --rm spacecraft-viz clean
docker-compose run --rm spacecraft-viz build

# Debug mode
docker-compose run --rm spacecraft-viz debug
```

### Project Structure
```
OrbitVizProjectRoot/
├── OrbitViz/                    # Main application source
│   ├── src/                     # C++ source files
│   │   ├── main.cpp            # Application entry point
│   │   ├── MainWindow.*        # Main GUI window
│   │   ├── attitude/           # Attitude dynamics engine
│   │   └── visualization/      # 3D rendering components
│   ├── ui/                     # Qt Designer UI files
│   ├── resources/              # Models, textures, assets
│   └── data/mock/              # Sample telemetry data
├── generate_telemetry.py       # Python data generator
├── docker-compose.yml          # Docker configuration
└── Dockerfile                  # Container definition
```

## Troubleshooting

### Common Issues

**Graphics/Rendering Problems**:
- Ensure X11 forwarding is properly configured
- Check OpenGL software rendering environment variables
- Verify graphics drivers are installed

**Build Errors**:
- Confirm VTK and Qt development packages are installed
- Check CMake can find all required libraries
- Verify Eigen3 headers are available

**Model Loading Issues**:
- Ensure OBJ files are in the correct format
- Check file paths are accessible to the application
- Verify model files aren't corrupted

### Environment Variables
```bash
export DISPLAY=:0                    # X11 display
export QT_X11_NO_MITSHM=1          # Qt X11 compatibility
export LIBGL_ALWAYS_SOFTWARE=1      # Force software OpenGL
export MESA_GL_VERSION_OVERRIDE=3.3  # OpenGL version override
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Built for spacecraft attitude visualization and analysis*