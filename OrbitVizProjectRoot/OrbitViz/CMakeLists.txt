cmake_minimum_required(VERSION 3.16)
project(OrbitViz VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Set the UI file search paths
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required Qt packages
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find all required VTK packages with specific components
find_package(VTK REQUIRED COMPONENTS 
    CommonCore
    CommonDataModel
    CommonTransforms
    CommonColor
    FiltersCore
    FiltersGeneral
    FiltersGeometry
    FiltersSources
    FiltersTexture      
    IOImage
    IOLegacy
    IOXML
    IOGeometry       # For OBJ model loading
    ImagingCore
    InteractionStyle
    RenderingCore
    RenderingOpenGL2
    RenderingFreeType
    RenderingContextOpenGL2
    GUISupportQt
    ChartsCore
    ViewsCore
    ViewsQt
)

# Print debug info
message(STATUS "VTK include dirs: ${VTK_INCLUDE_DIRS}")
message(STATUS "VTK libraries: ${VTK_LIBRARIES}")

# Include VTK and Eigen modules directly
include_directories(
    ${VTK_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/attitude
    ${CMAKE_CURRENT_SOURCE_DIR}/src/visualization
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# Define a preprocessor directive for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

# Add UI file with full path
set(UI_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ui/mainwindow.ui
)

# Add source files
add_executable(OrbitViz
    src/main.cpp
    src/MainWindow.h
    src/MainWindow.cpp
    src/attitude/AttitudeEngine.h
    src/attitude/AttitudeEngine.cpp
    src/visualization/SpacecraftView.h
    src/visualization/SpacecraftView.cpp
    ${UI_FILES}
)

# Include directories
target_include_directories(OrbitViz PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR} # For generated UI headers
    ${VTK_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(OrbitViz PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    ${VTK_LIBRARIES}
)

vtk_module_autoinit(
    TARGETS OrbitViz
    MODULES ${VTK_LIBRARIES}
)

# Create resources directories in build
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/resources/textures)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/resources/models)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data/mock)

# Copy resource files to build directory
file(GLOB TEXTURE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/textures/*.jpg ${CMAKE_CURRENT_SOURCE_DIR}/resources/textures/*.png)
file(GLOB MODEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/models/*.obj ${CMAKE_CURRENT_SOURCE_DIR}/resources/models/*.mtl)
file(GLOB MOCK_DATA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/mock/*.json)

foreach(TEXTURE_FILE ${TEXTURE_FILES})
    file(COPY ${TEXTURE_FILE} DESTINATION ${CMAKE_BINARY_DIR}/resources/textures/)
endforeach()

foreach(MODEL_FILE ${MODEL_FILES})
    file(COPY ${MODEL_FILE} DESTINATION ${CMAKE_BINARY_DIR}/resources/models/)
endforeach()

foreach(DATA_FILE ${MOCK_DATA_FILES})
    file(COPY ${DATA_FILE} DESTINATION ${CMAKE_BINARY_DIR}/data/mock/)
endforeach()

# Install instructions
install(TARGETS OrbitViz
    RUNTIME DESTINATION bin
)

# Install resource files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/
    DESTINATION share/OrbitViz/resources
    FILES_MATCHING PATTERN "*.jpg" PATTERN "*.png" PATTERN "*.obj" PATTERN "*.mtl"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
    DESTINATION share/OrbitViz/data
    FILES_MATCHING PATTERN "*.json"
)