#!/bin/bash

# Make sure proper execution permission is set
chmod +x $(dirname $0)/entrypoint.sh

# Create runtime directory
mkdir -p /tmp/runtime-root
export XDG_RUNTIME_DIR=/tmp/runtime-root
chmod 0700 /tmp/runtime-root

# Setup environment for software OpenGL rendering
export QT_XCB_NO_MITSHM=1
export QT_OPENGL=software
export QT_NO_GLYPH_CACHE=1

# Configure MESA software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export GALLIUM_DRIVER=llvmpipe

# Set up VTK environment
export VTK_RENDERER=MESA

# Make sure the necessary directories exist
mkdir -p /app/OrbitViz/data/mock
mkdir -p /app/OrbitViz/build
mkdir -p /app/OrbitViz/resources/textures
mkdir -p /app/OrbitViz/resources/models
mkdir -p /app/OrbitViz/build/resources/textures
mkdir -p /app/OrbitViz/build/resources/models
mkdir -p /app/OrbitViz/build/data/mock

# Run ldconfig to update the shared library cache
ldconfig

# Print a welcome message
echo "=== Spacecraft Attitude Visualization Tool ==="
echo "QT_OPENGL: $QT_OPENGL"
echo "LIBGL_ALWAYS_SOFTWARE: $LIBGL_ALWAYS_SOFTWARE"
echo "MESA_GL_VERSION_OVERRIDE: $MESA_GL_VERSION_OVERRIDE"
echo "=============================================="

if [ "$1" = "build" ]; then
  mkdir -p /app/OrbitViz/build
  cd /app/OrbitViz/build
  cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/vtk-9.2 ..
  make -j$(nproc)
elif [ "$1" = "run" ]; then
  if [ ! -f "/app/OrbitViz/build/OrbitViz" ]; then
    echo "Application not built yet. Building now..."
    mkdir -p /app/OrbitViz/build
    cd /app/OrbitViz/build
    cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/vtk-9.2 ..
    make -j$(nproc)
  fi
  cd /app/OrbitViz/build
  # Run with proper environment settings
  ./OrbitViz
elif [ "$1" = "debug" ]; then
  # Print diagnostic info
  echo "===== System Information ====="
  uname -a
  echo "===== OpenGL Support ====="
  glxinfo | grep "OpenGL" || echo "glxinfo not available"
  echo "===== VTK Information ====="
  ls -la /usr/local/lib | grep -E 'vtk' || echo "No VTK libraries found"
  echo "===== Environment Variables ====="
  env | grep -E "QT_|VTK_|LD_|CMAKE|MESA|LIBGL"
  
  # Build with debug symbols
  mkdir -p /app/OrbitViz/build
  cd /app/OrbitViz/build
  cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/vtk-9.2 ..
  make -j$(nproc)
  
  echo "===== Running with debug settings ====="
  QT_LOGGING_RULES="*.debug=true" ./OrbitViz
elif [ "$1" = "clean" ]; then
  echo "Cleaning build directory..."
  rm -rf /app/OrbitViz/build/*
  echo "Done."
elif [ "$1" = "shell" ]; then
  /bin/bash
else
  echo "Usage: $0 {build|run|debug|clean|shell}"
  exit 1
fi