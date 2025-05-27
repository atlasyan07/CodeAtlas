#include "SpacecraftView.h"
#include "AttitudeEngine.h"

#include <QDebug>
#include <QDir>
#include <QResizeEvent>
#include <ctime>
#include <cmath>

// VTK includes
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkOBJReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkArrowSource.h>
#include <vtkSphereSource.h>
#include <vtkCubeSource.h>
#include <vtkAxisActor.h>
#include <vtkTubeFilter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLight.h>
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkLine.h>
#include <vtkNamedColors.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkCaptionActor2D.h>

SpacecraftView::SpacecraftView(QWidget* parent)
    : QVTKOpenGLNativeWidget(parent)
    , animationTimer(new QTimer(this))
    , currentDateTime(QDateTime::currentDateTime())
    , animationSpeed(1.0)
    , animating(false)
    , isInitialized(false)
    , initializationAttempted(false)
    , renderer(nullptr)
    , spacecraftActor(nullptr)
    , spacecraftMapper(nullptr)
    , spacecraftReader(nullptr)
    , spacecraftTransform(nullptr)
    , refFrameXActor(nullptr)
    , refFrameYActor(nullptr)
    , refFrameZActor(nullptr)
    , bodyRateXActor(nullptr)
    , bodyRateYActor(nullptr)
    , bodyRateZActor(nullptr)
    , starsActor(nullptr)
    , mainLight(nullptr)
    , attitudeEngine(nullptr)
    , spacecraftModelPath("resources/models/cubesat.obj") // Default model
{
    // Connect the animation timer
    connect(animationTimer, &QTimer::timeout, this, &SpacecraftView::onTimerUpdate);
    animationTimer->setInterval(16); // ~60 FPS
    
    // Configure OpenGL parameters for better rendering
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    
    // Configure the render window
    renderWindow->SetMultiSamples(0);  // Disable multisampling for software rendering
    renderWindow->SetPointSmoothing(true);
    renderWindow->SetLineSmoothing(true);
    renderWindow->SetPolygonSmoothing(true);
    
    this->setRenderWindow(renderWindow);
    
    // Create renderer
    renderer = vtkRenderer::New();
    renderer->SetBackground(0.0, 0.0, 0.0); // Pure black for space
    
    // Add renderer to window
    renderWindow->AddRenderer(renderer);
    
    // Create interactor style for camera manipulation
    vtkNew<vtkInteractorStyleTrackballCamera> style;
    this->renderWindow()->GetInteractor()->SetInteractorStyle(style);
    
    // Setup lighting
    setupLighting();
    
    // Start timer for delayed initialization - wait longer for window to properly size
    QTimer::singleShot(300, this, &SpacecraftView::delayedInitialization);
    
    // Initialize random seed for star generation
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    // Start animation timer
    animationTimer->start();
}

SpacecraftView::~SpacecraftView()
{
    // Stop timer if active
    if (animationTimer->isActive()) {
        animationTimer->stop();
    }
    
    // Clean up VTK objects
    if (renderer) {
        renderer->Delete();
        renderer = nullptr;
    }

    if (spacecraftActor) {
        spacecraftActor->Delete();
        spacecraftActor = nullptr;
    }

    if (spacecraftMapper) {
        spacecraftMapper->Delete();
        spacecraftMapper = nullptr;
    }

    if (spacecraftReader) {
        spacecraftReader->Delete();
        spacecraftReader = nullptr;
    }

    if (spacecraftTransform) {
        spacecraftTransform->Delete();
        spacecraftTransform = nullptr;
    }
    
    // Clean up reference frame axes
    if (refFrameXActor) {
        refFrameXActor->Delete();
        refFrameXActor = nullptr;
    }
    
    if (refFrameYActor) {
        refFrameYActor->Delete();
        refFrameYActor = nullptr;
    }
    
    if (refFrameZActor) {
        refFrameZActor->Delete();
        refFrameZActor = nullptr;
    }
    
    // Clean up body rate actors
    if (bodyRateXActor) {
        bodyRateXActor->Delete();
        bodyRateXActor = nullptr;
    }
    
    if (bodyRateYActor) {
        bodyRateYActor->Delete();
        bodyRateYActor = nullptr;
    }
    
    if (bodyRateZActor) {
        bodyRateZActor->Delete();
        bodyRateZActor = nullptr;
    }
    
    // Clean up stars actor
    if (starsActor) {
        starsActor->Delete();
        starsActor = nullptr;
    }
    
    // Clean up light
    if (mainLight) {
        mainLight->Delete();
        mainLight = nullptr;
    }
    
    // Clean up vector visualizations
    for (auto it = vectorVisualizations.begin(); it != vectorVisualizations.end(); ++it) {
        VectorVisualization& vis = it.value();
        if (vis.actor) {
            vis.actor->Delete();
        }
        if (vis.source) {
            vis.source->Delete();
        }
        if (vis.mapper) {
            vis.mapper->Delete();
        }
        if (vis.transform) {
            vis.transform->Delete();
        }
    }
    vectorVisualizations.clear();
}

void SpacecraftView::setAttitudeEngine(AttitudeEngine* engine)
{
    // Store the pointer to the attitude engine
    attitudeEngine = engine;
    
    // Connect attitude engine signals
    if (attitudeEngine) {
        connect(attitudeEngine, &AttitudeEngine::attitudeUpdated,
                this, &SpacecraftView::onAttitudeUpdated);
        
        connect(attitudeEngine, &AttitudeEngine::bodyRatesUpdated,
                this, &SpacecraftView::onBodyRatesUpdated);
        
        connect(attitudeEngine, &AttitudeEngine::vectorUpdated,
                this, &SpacecraftView::onVectorUpdated);
    }
}

void SpacecraftView::setupLighting()
{
    // Create main light
    mainLight = vtkLight::New();
    mainLight->SetLightTypeToSceneLight();
    mainLight->SetPositional(false);  // Directional light
    mainLight->SetColor(1.0, 1.0, 1.0);  // White light
    mainLight->SetIntensity(1.0);  
    
    // Position the light
    mainLight->SetPosition(10.0, 10.0, 10.0);
    mainLight->SetFocalPoint(0.0, 0.0, 0.0);
    
    // Add the light to the renderer
    renderer->AddLight(mainLight);
    
    // Add ambient light to ensure objects aren't too dark
    vtkSmartPointer<vtkLight> ambientLight = vtkSmartPointer<vtkLight>::New();
    ambientLight->SetLightTypeToHeadlight();  // Follows camera
    ambientLight->SetIntensity(0.3);         // Low intensity ambient
    ambientLight->SetColor(0.6, 0.6, 0.6);    // Gray ambient light
    renderer->AddLight(ambientLight);
}

void SpacecraftView::setupSpaceBackground()
{
    // Set the renderer's background to pure black
    renderer->SetBackground(0.0, 0.0, 0.0);
    qDebug() << "Creating starfield background...";
    
    // Create a set of points for stars
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    
    // Create cells (vertices) for each point
    vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
    
    // Create colors for the points
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3); // RGB
    colors->SetName("Colors");
    
    // Create stars
    const int numStars = 2000;
    const double distance = 100.0;  // Distance from center
    
    for (int i = 0; i < numStars; i++) {
        // Generate random position on a sphere with radius 'distance'
        double theta = 2.0 * PI * (static_cast<double>(rand()) / RAND_MAX);
        double phi = acos(2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0);
        
        double x = distance * sin(phi) * cos(theta);
        double y = distance * sin(phi) * sin(theta);
        double z = distance * cos(phi);
        
        // Add the point
        vtkIdType pointId = points->InsertNextPoint(x, y, z);
        
        // Add a vertex cell for this point
        vertices->InsertNextCell(1, &pointId);
        
        // Determine star brightness and color
        double randValue = static_cast<double>(rand()) / RAND_MAX;
        unsigned char r, g, b;
        
        if (randValue > 0.97) {
            // Brightest stars (3% of stars)
            r = g = b = 255;
        } else if (randValue > 0.85) {
            // Very bright stars (12% of stars)
            r = g = b = 220;
        } else if (randValue > 0.65) {
            // Medium bright stars (20% of stars)
            r = g = b = 180;
        } else {
            // Dim stars (65% of stars)
            r = g = b = 120;
        }
        
        // Apply color variations to some stars
        if (randValue > 0.985) {
            // Blue stars
            r = static_cast<unsigned char>(r * 0.7);
            g = static_cast<unsigned char>(g * 0.8);
            // b remains the same (full blue component)
        } else if (randValue > 0.97 && randValue <= 0.985) {
            // Red stars
            // r remains the same (full red component)
            g = static_cast<unsigned char>(g * 0.7);
            b = static_cast<unsigned char>(b * 0.7);
        } else if (randValue > 0.955 && randValue <= 0.97) {
            // Yellow stars
            // r and g remain the same (full red and green components)
            b = static_cast<unsigned char>(b * 0.7);
        }
        
        // Add the color to the array
        colors->InsertNextTuple3(r, g, b);
    }
    
    // Create a polydata to store the points
    vtkSmartPointer<vtkPolyData> starsPolyData = vtkSmartPointer<vtkPolyData>::New();
    starsPolyData->SetPoints(points);
    starsPolyData->SetVerts(vertices);
    starsPolyData->GetPointData()->SetScalars(colors);
    
    // Create a mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(starsPolyData);
    
    // Create the actor
    starsActor = vtkActor::New();
    starsActor->SetMapper(mapper);
    
    // Increase point size for more visible stars
    starsActor->GetProperty()->SetPointSize(2.0);
    
    // Disable lighting for stars
    starsActor->GetProperty()->SetLighting(false);
    
    // Add to renderer
    renderer->AddActor(starsActor);
    
    qDebug() << "Created starfield with" << numStars << "points";
}

void SpacecraftView::setupCamera()
{
    if (renderer) {
        vtkCamera* camera = renderer->GetActiveCamera();
        
        // Set initial camera position
        camera->SetPosition(0, -10, 5);
        camera->SetFocalPoint(0, 0, 0);
        camera->SetViewUp(0, 0, 1);
        camera->SetViewAngle(30);  // Narrower angle for a better perspective
        
        // Set clipping ranges
        camera->SetClippingRange(0.1, 1000);
        
        // Enable camera to use view angle (perspective)
        camera->ParallelProjectionOff();
        
        // Set ambient sampling for better lighting
        renderer->SetUseFXAA(true);  // Enable FXAA antialiasing if available
        renderer->SetTwoSidedLighting(true);
        renderer->SetAutomaticLightCreation(false);
    }
}

void SpacecraftView::createSpacecraft()
{
    qDebug() << "Creating spacecraft model...";
    
    // Try to find the spacecraft model
    QStringList modelPaths = {
        spacecraftModelPath,
        "/app/OrbitViz/resources/models/cubesat.obj",
        "/app/OrbitViz/build/resources/models/cubesat.obj",
        "resources/models/cubesat.obj"
    };
    
    bool modelFound = false;
    QString modelFile;
    
    for (const QString& path : modelPaths) {
        if (QFile::exists(path)) {
            modelFile = path;
            modelFound = true;
            qDebug() << "Found spacecraft model:" << path;
            break;
        }
    }
    
    if (!modelFound) {
        qWarning() << "Could not find spacecraft model! Using a cube instead.";
        
        // Create a simple cube as fallback
        vtkSmartPointer<vtkCubeSource> cubeSource = vtkSmartPointer<vtkCubeSource>::New();
        cubeSource->SetXLength(1.0);
        cubeSource->SetYLength(2.0);
        cubeSource->SetZLength(0.5);
        cubeSource->SetCenter(0.0, 0.0, 0.0);
        
        spacecraftMapper = vtkPolyDataMapper::New();
        spacecraftMapper->SetInputConnection(cubeSource->GetOutputPort());
    } else {
        // Load the OBJ model
        spacecraftReader = vtkOBJReader::New();
        spacecraftReader->SetFileName(modelFile.toStdString().c_str());
        spacecraftReader->Update();
        
        spacecraftMapper = vtkPolyDataMapper::New();
        spacecraftMapper->SetInputConnection(spacecraftReader->GetOutputPort());
    }
    
    // Create spacecraft actor
    spacecraftActor = vtkActor::New();
    spacecraftActor->SetMapper(spacecraftMapper);
    
    // Set material properties
    spacecraftActor->GetProperty()->SetColor(0.7, 0.7, 0.7); // Light gray
    spacecraftActor->GetProperty()->SetAmbient(0.3);
    spacecraftActor->GetProperty()->SetDiffuse(0.7);
    spacecraftActor->GetProperty()->SetSpecular(0.5);
    spacecraftActor->GetProperty()->SetSpecularPower(20);
    
    // Create transform for attitude changes
    spacecraftTransform = vtkTransform::New();
    spacecraftTransform->PostMultiply(); // Set to post-multiply mode
    spacecraftActor->SetUserTransform(spacecraftTransform);
    
    // Add to renderer
    renderer->AddActor(spacecraftActor);
    
    qDebug() << "Spacecraft model created";
}

void SpacecraftView::createCoordinateAxes()
{
    qDebug() << "Creating coordinate reference axes...";
    
    // Create X axis (red)
    vtkSmartPointer<vtkArrowSource> xAxisSource = vtkSmartPointer<vtkArrowSource>::New();
    xAxisSource->SetShaftRadius(0.03);
    xAxisSource->SetTipRadius(0.1);
    xAxisSource->SetTipLength(0.2);
    
    vtkSmartPointer<vtkPolyDataMapper> xAxisMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    xAxisMapper->SetInputConnection(xAxisSource->GetOutputPort());
    
    refFrameXActor = vtkActor::New();
    refFrameXActor->SetMapper(xAxisMapper);
    refFrameXActor->GetProperty()->SetColor(1.0, 0.0, 0.0); // Red
    
    // Position/orient the X axis
    vtkSmartPointer<vtkTransform> xAxisTransform = vtkSmartPointer<vtkTransform>::New();
    xAxisTransform->Identity();
    xAxisTransform->Scale(3.0, 3.0, 3.0); // Scale the arrow
    refFrameXActor->SetUserTransform(xAxisTransform);
    
    // Create Y axis (green)
    vtkSmartPointer<vtkArrowSource> yAxisSource = vtkSmartPointer<vtkArrowSource>::New();
    yAxisSource->SetShaftRadius(0.03);
    yAxisSource->SetTipRadius(0.1);
    yAxisSource->SetTipLength(0.2);
    
    vtkSmartPointer<vtkPolyDataMapper> yAxisMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    yAxisMapper->SetInputConnection(yAxisSource->GetOutputPort());
    
    refFrameYActor = vtkActor::New();
    refFrameYActor->SetMapper(yAxisMapper);
    refFrameYActor->GetProperty()->SetColor(0.0, 1.0, 0.0); // Green
    
    // Position/orient the Y axis
    vtkSmartPointer<vtkTransform> yAxisTransform = vtkSmartPointer<vtkTransform>::New();
    yAxisTransform->Identity();
    yAxisTransform->RotateZ(90); // Rotate to point along Y axis
    yAxisTransform->Scale(3.0, 3.0, 3.0); // Scale the arrow
    refFrameYActor->SetUserTransform(yAxisTransform);
    
    // Create Z axis (blue)
    vtkSmartPointer<vtkArrowSource> zAxisSource = vtkSmartPointer<vtkArrowSource>::New();
    zAxisSource->SetShaftRadius(0.03);
    zAxisSource->SetTipRadius(0.1);
    zAxisSource->SetTipLength(0.2);
    
    vtkSmartPointer<vtkPolyDataMapper> zAxisMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    zAxisMapper->SetInputConnection(zAxisSource->GetOutputPort());
    
    refFrameZActor = vtkActor::New();
    refFrameZActor->SetMapper(zAxisMapper);
    refFrameZActor->GetProperty()->SetColor(0.0, 0.0, 1.0); // Blue
    
    // Position/orient the Z axis
    vtkSmartPointer<vtkTransform> zAxisTransform = vtkSmartPointer<vtkTransform>::New();
    zAxisTransform->Identity();
    zAxisTransform->RotateY(-90); // Rotate to point along Z axis
    zAxisTransform->Scale(3.0, 3.0, 3.0); // Scale the arrow
    refFrameZActor->SetUserTransform(zAxisTransform);
    
    // Add to renderer
    renderer->AddActor(refFrameXActor);
    renderer->AddActor(refFrameYActor);
    renderer->AddActor(refFrameZActor);
    
    qDebug() << "Coordinate axes created";
}

void SpacecraftView::createVectorVisualizations()
{
    qDebug() << "Creating vector visualizations...";
    
    // Initialize vector visualizations for different vector types
    auto createVectorVis = [this](AttitudeEngine::VectorType type, const QColor& color, 
                                 double scale, bool visible) {
        VectorVisualization vis;
        
        // Create arrow source
        vis.source = vtkArrowSource::New();
        vis.source->SetShaftRadius(0.03);
        vis.source->SetTipRadius(0.1);
        vis.source->SetTipLength(0.2);
        
        // Create mapper
        vis.mapper = vtkPolyDataMapper::New();
        vis.mapper->SetInputConnection(vis.source->GetOutputPort());
        
        // Create actor
        vis.actor = vtkActor::New();
        vis.actor->SetMapper(vis.mapper);
        vis.actor->GetProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
        
        // Create transform
        vis.transform = vtkTransform::New();
        vis.transform->Identity();
        vis.transform->Scale(scale, scale, scale);
        vis.actor->SetUserTransform(vis.transform);
        
        // Set properties
        vis.visible = visible;
        vis.scale = scale;
        vis.color = color;
        
        // Set initial visibility
        vis.actor->SetVisibility(visible);
        
        // Add to renderer
        renderer->AddActor(vis.actor);
        
        // Store in map
        vectorVisualizations[type] = vis;
    };
    
    // Create visualizations for different vector types
    createVectorVis(AttitudeEngine::BODY_X, QColor(255, 0, 0), 2.0, true);     // Body X axis (red)
    createVectorVis(AttitudeEngine::BODY_Y, QColor(0, 255, 0), 2.0, true);     // Body Y axis (green)
    createVectorVis(AttitudeEngine::BODY_Z, QColor(0, 0, 255), 2.0, true);     // Body Z axis (blue)
    createVectorVis(AttitudeEngine::NADIR, QColor(255, 255, 0), 2.5, true);    // Nadir vector (yellow)
    createVectorVis(AttitudeEngine::SUN, QColor(255, 165, 0), 3.0, true);      // Sun vector (orange)
    createVectorVis(AttitudeEngine::VELOCITY, QColor(0, 255, 255), 2.5, false); // Velocity vector (cyan)
    createVectorVis(AttitudeEngine::ANGULAR_MOMENTUM, QColor(255, 0, 255), 2.0, false); // Angular momentum (magenta)
    
    qDebug() << "Vector visualizations created";
}

void SpacecraftView::updateVectorActor(int vectorType, const Eigen::Vector3d& vector)
{
    if (!vectorVisualizations.contains(vectorType)) {
        return;
    }
    
    VectorVisualization& vis = vectorVisualizations[vectorType];
    
    // Only update if the vector is visible
    if (!vis.visible) {
        return;
    }
    
    // Normalize the vector
    Eigen::Vector3d normalizedVector = vector.normalized();
    
    // Calculate rotation to align arrow with vector
    // The arrow initially points along the x-axis
    Eigen::Vector3d xAxis(1.0, 0.0, 0.0);
    
    // Calculate angle and rotation axis
    double angle = acos(xAxis.dot(normalizedVector));
    
    // Check if vectors are nearly parallel or anti-parallel
    if (std::abs(angle) < 1e-6) {
        // No rotation needed (vectors are nearly parallel)
        vis.transform->Identity();
    } else if (std::abs(angle - PI) < 1e-6) {
        // Vectors are nearly anti-parallel, rotate 180 degrees around any perpendicular axis
        vis.transform->Identity();
        vis.transform->RotateY(180.0);
    } else {
        // Normal case - calculate rotation axis using cross product
        Eigen::Vector3d rotAxis = xAxis.cross(normalizedVector).normalized();
        
        // Convert angle to degrees
        double angleDegrees = angle * 180.0 / PI;
        
        // Set the transform
        vis.transform->Identity();
        vis.transform->RotateWXYZ(angleDegrees, rotAxis.x(), rotAxis.y(), rotAxis.z());
    }
    
    // Scale arrow based on the vector magnitude and the visualization scale
    vis.transform->Scale(vector.norm() * vis.scale, vis.scale, vis.scale);
    
    // Force update
    vis.actor->Modified();
}

void SpacecraftView::delayedInitialization()
{
    // Prevent multiple initialization attempts
    if (initializationAttempted) {
        return;
    }
    
    initializationAttempted = true;
    
    // Check if the render window has valid dimensions now
    int* windowSize = this->renderWindow()->GetSize();
    
    if (windowSize[0] > 0 && windowSize[1] > 0) {
        qDebug() << "Initializing with valid window dimensions:" 
                 << windowSize[0] << "x" << windowSize[1];
        
        // Setup background (stars)
        setupSpaceBackground();
        
        // Setup camera
        setupCamera();
        
        // Create the spacecraft model
        createSpacecraft();
        
        // Create coordinate axes
        createCoordinateAxes();
        
        // Create vector visualizations
        createVectorVisualizations();
        
        // Reset the camera to frame the spacecraft correctly
        renderer->ResetCamera();
        
        // Force a render update
        this->renderWindow()->Render();
        
        isInitialized = true;
    } else {
        qDebug() << "Window size still invalid, retrying initialization in 200ms";
        // Reset the flag to allow another attempt
        initializationAttempted = false;
        // Try again in a short while
        QTimer::singleShot(200, this, &SpacecraftView::delayedInitialization);
    }
}

void SpacecraftView::resizeEvent(QResizeEvent* event)
{
    QVTKOpenGLNativeWidget::resizeEvent(event);
    
    // If we haven't initialized yet, try now
    if (!isInitialized && !initializationAttempted) {
        delayedInitialization();
    }
    
    // Otherwise adjust camera properly
    if (renderer && isInitialized) {
        // Update the aspect ratio of the camera
        vtkCamera* camera = renderer->GetActiveCamera();
        if (camera) {
            double aspectRatio = static_cast<double>(event->size().width()) / 
                                 static_cast<double>(event->size().height());
                                 
            // Preserve the camera's position but update its aspect ratio
            double position[3], focalPoint[3], viewUp[3];
            camera->GetPosition(position);
            camera->GetFocalPoint(focalPoint);
            camera->GetViewUp(viewUp);
            
            // Update the aspect ratio
            camera->SetViewAngle(30);  // Maintain a consistent field of view
            renderer->ResetCameraClippingRange();
            
            // Force a render update
            this->renderWindow()->Render();
        }
    }
}

void SpacecraftView::onTimerUpdate()
{
    if (animating && isInitialized) {
        // Update simulation time based on speed
        currentDateTime = currentDateTime.addSecs(static_cast<qint64>(animationSpeed));
        
        // Request a render update
        if (this->renderWindow()) {
            this->renderWindow()->Render();
        }
    }
}

void SpacecraftView::onAttitudeUpdated(const Eigen::Quaterniond& quat)
{
    if (!isInitialized || !spacecraftTransform) {
        return;
    }
    
    // Convert quaternion to axis-angle representation for VTK
    double angle = 2.0 * acos(quat.w()) * 180.0 / PI;  // Convert to degrees for VTK
    
    // Avoid division by zero or very small numbers
    double norm = quat.vec().norm();
    if (norm < 1e-10) {
        // Identity rotation
        spacecraftTransform->Identity();
    } else {
        // Get rotation axis
        Eigen::Vector3d axis = quat.vec() / norm;
        
        // Apply the rotation
        spacecraftTransform->Identity();
        spacecraftTransform->RotateWXYZ(angle, axis.x(), axis.y(), axis.z());
    }
    
    // Force update
    if (spacecraftActor) {
        spacecraftActor->Modified();
    }
    
    // Request a render update
    if (this->renderWindow()) {
        this->renderWindow()->Render();
    }
}

void SpacecraftView::onBodyRatesUpdated(double wx, double wy, double wz)
{
    // Update body rate visualization if needed
    // This is a placeholder for body rate visualization
}

void SpacecraftView::onVectorUpdated(int vectorType, const Eigen::Vector3d& vector)
{
    // Update the corresponding vector visualization
    updateVectorActor(vectorType, vector);
    
    // Request a render update
    if (this->renderWindow()) {
        this->renderWindow()->Render();
    }
}

void SpacecraftView::startAnimation()
{
    animating = true;
    
    if (!animationTimer->isActive()) {
        animationTimer->start();
    }
}

void SpacecraftView::stopAnimation()
{
    animating = false;
}

void SpacecraftView::setAnimationSpeed(double speed)
{
    animationSpeed = speed;
}

void SpacecraftView::setDateTime(const QDateTime& dateTime)
{
    currentDateTime = dateTime;
}

bool SpacecraftView::loadSpacecraftModel(const QString& modelPath)
{
    if (!QFile::exists(modelPath)) {
        qWarning() << "Spacecraft model file not found:" << modelPath;
        return false;
    }
    
    spacecraftModelPath = modelPath;
    
    // If already initialized, reload the model
    if (isInitialized && spacecraftActor) {
        // Remove old model
        renderer->RemoveActor(spacecraftActor);
        
        if (spacecraftMapper) {
            spacecraftMapper->Delete();
            spacecraftMapper = nullptr;
        }
        
        if (spacecraftReader) {
            spacecraftReader->Delete();
            spacecraftReader = nullptr;
        }
        
        // Load new model
        spacecraftReader = vtkOBJReader::New();
        spacecraftReader->SetFileName(modelPath.toStdString().c_str());
        spacecraftReader->Update();
        
        spacecraftMapper = vtkPolyDataMapper::New();
        spacecraftMapper->SetInputConnection(spacecraftReader->GetOutputPort());
        
        spacecraftActor->SetMapper(spacecraftMapper);
        
        // Add back to renderer
        renderer->AddActor(spacecraftActor);
        
        // Force render update
        this->renderWindow()->Render();
    }
    
    return true;
}

void SpacecraftView::showVector(int vectorType, bool visible)
{
    if (!vectorVisualizations.contains(vectorType)) {
        return;
    }
    
    VectorVisualization& vis = vectorVisualizations[vectorType];
    vis.visible = visible;
    vis.actor->SetVisibility(visible);
    
    // Update the attitude engine if connected
    if (attitudeEngine) {
        attitudeEngine->toggleVector(static_cast<AttitudeEngine::VectorType>(vectorType), visible);
    }
    
    // Force render update
    this->renderWindow()->Render();
}

void SpacecraftView::setVectorColor(int vectorType, const QColor& color)
{
    if (!vectorVisualizations.contains(vectorType)) {
        return;
    }
    
    VectorVisualization& vis = vectorVisualizations[vectorType];
    vis.color = color;
    vis.actor->GetProperty()->SetColor(color.redF(), color.greenF(), color.blueF());
    
    // Force render update
    this->renderWindow()->Render();
}

void SpacecraftView::setVectorScale(int vectorType, double scale)
{
    if (!vectorVisualizations.contains(vectorType)) {
        return;
    }
    
    VectorVisualization& vis = vectorVisualizations[vectorType];
    vis.scale = scale;
    
    // Update transform
    if (vis.transform) {
        // Get rotation part from current transform
        double wxyz[4]; // Array to hold the orientation values
        vis.transform->GetOrientationWXYZ(wxyz); // Use the correct API with array
        
        // Recreate transform with new scale
        vis.transform->Identity();
        vis.transform->RotateWXYZ(wxyz[0], wxyz[1], wxyz[2], wxyz[3]);
        vis.transform->Scale(scale, scale, scale);
    }
    
    // Force render update
    this->renderWindow()->Render();
}