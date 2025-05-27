# Dockerfile for Qt with VTK for Spacecraft Attitude Visualization Tool
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Build tools for our application
    build-essential \
    cmake \
    git \
    # Qt dependencies
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtdeclarative5-dev \
    libqt5svg5-dev \
    # OpenGL dependencies
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    # VTK dependencies
    libxt-dev \
    libqt5x11extras5-dev \
    # X11 dependencies
    libxkbcommon-x11-dev \
    libxcb-icccm4-dev \
    libxcb-image0-dev \
    libxcb-keysyms1-dev \
    libxcb-render-util0-dev \
    libxcb-xinerama0-dev \
    libxcb-xkb-dev \
    # Math libraries
    libeigen3-dev \
    # Utilities
    wget \
    unzip \
    vim \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build VTK with Qt support
WORKDIR /tmp
RUN wget -q https://www.vtk.org/files/release/9.2/VTK-9.2.6.tar.gz && \
    tar -xzf VTK-9.2.6.tar.gz && \
    mkdir VTK-9.2.6/build && \
    cd VTK-9.2.6/build && \
    cmake -DVTK_GROUP_ENABLE_Qt=YES \
          -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES \
          -DVTK_MODULE_ENABLE_VTK_GUISupportQtQuick=NO \
          -DVTK_MODULE_ENABLE_VTK_RenderingQt=YES \
          -DVTK_QT_VERSION=5 \
          -DVTK_BUILD_QT_DESIGNER_PLUGIN=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DVTK_USE_X=ON \
          -DVTK_USE_OFFSCREEN=OFF \
          -DVTK_RENDERING_BACKEND=OpenGL2 \
          -DBUILD_SHARED_LIBS=ON \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && \
    rm -rf VTK-9.2.6*

# Create app directory
WORKDIR /app

# Set up environment variables
ENV QT_XCB_NO_MITSHM=1
ENV QT_OPENGL=software
ENV QT_GRAPHICSSYSTEM=native
ENV CMAKE_PREFIX_PATH=/usr/local/lib/cmake/vtk-9.2:$CMAKE_PREFIX_PATH

# Copy scripts
COPY entrypoint.sh /app/

# Set executable permissions
RUN chmod +x /app/entrypoint.sh

# Create directory structure
RUN mkdir -p /app/OrbitViz/data/mock && \
    mkdir -p /app/OrbitViz/resources/textures && \
    mkdir -p /app/OrbitViz/resources/models

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["shell"]