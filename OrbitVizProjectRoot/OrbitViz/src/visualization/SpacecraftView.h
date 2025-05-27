#ifndef SPACECRAFTVIEW_H
#define SPACECRAFTVIEW_H

#include <QWidget>
#include <QTimer>
#include <QDateTime>
#include <QMap>
#include <QString>
#include <Eigen/Geometry>

// VTK forward declarations
class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
class vtkActor;
class vtkPolyDataMapper;
class vtkOBJReader;
class vtkTransform;
class vtkArrowSource;
class vtkLight;
class vtkCamera;
class vtkPoints;
class vtkCellArray;
class vtkPolyData;

// QVTKOpenGLNativeWidget
#include <QVTKOpenGLNativeWidget.h>

// Forward declaration
class AttitudeEngine;

class SpacecraftView : public QVTKOpenGLNativeWidget
{
    Q_OBJECT

public:
    explicit SpacecraftView(QWidget* parent = nullptr);
    ~SpacecraftView();

    // Connect attitude engine
    void setAttitudeEngine(AttitudeEngine* engine);

    // Control methods
    void startAnimation();
    void stopAnimation();
    void setAnimationSpeed(double speed);
    void setDateTime(const QDateTime& dateTime);
    
    // Load spacecraft model
    bool loadSpacecraftModel(const QString& modelPath);
    
    // Vector visualization control
    void showVector(int vectorType, bool visible);
    void setVectorColor(int vectorType, const QColor& color);
    void setVectorScale(int vectorType, double scale);

signals:
    void frameSwapped();

protected:
    // Override QWidget event handlers
    void resizeEvent(QResizeEvent* event) override;
    
private slots:
    void onTimerUpdate();        // Timer-driven update function
    void delayedInitialization(); // Delayed initialization function
    
    // Update slot for attitude changes
    void onAttitudeUpdated(const Eigen::Quaterniond& quat);
    
    // Update slot for body rates
    void onBodyRatesUpdated(double wx, double wy, double wz);
    
    // Update slot for vector changes
    void onVectorUpdated(int vectorType, const Eigen::Vector3d& vector);
    
private:
    // Setup and creation methods
    void setupLighting();
    void setupCamera();
    void setupSpaceBackground();
    void createSpacecraft();
    void createVectorVisualizations();
    void createCoordinateAxes();
    
    // Vector visualization helper
    void updateVectorActor(int vectorType, const Eigen::Vector3d& vector);
    
    // Animation properties
    QTimer* animationTimer;
    QDateTime currentDateTime;
    double animationSpeed;
    bool animating;
    bool isInitialized;         // Flag to track if the spacecraft has been created
    bool initializationAttempted; // Flag to prevent repeated initialization attempts
    
    // VTK objects
    vtkRenderer* renderer;
    vtkActor* spacecraftActor;
    vtkPolyDataMapper* spacecraftMapper;
    vtkOBJReader* spacecraftReader;
    vtkTransform* spacecraftTransform;
    
    // Reference frame axes
    vtkActor* refFrameXActor;
    vtkActor* refFrameYActor;
    vtkActor* refFrameZActor;
    
    // Vector actors for different vector types
    struct VectorVisualization {
        vtkActor* actor;
        vtkArrowSource* source;
        vtkPolyDataMapper* mapper;
        vtkTransform* transform;
        bool visible;
        double scale;
        QColor color;
    };
    
    QMap<int, VectorVisualization> vectorVisualizations;
    
    // Body rate visualization
    vtkActor* bodyRateXActor;
    vtkActor* bodyRateYActor;
    vtkActor* bodyRateZActor;
    
    // Stars background
    vtkActor* starsActor;
    
    // Lighting
    vtkLight* mainLight;
    
    // Pointer to attitude engine
    AttitudeEngine* attitudeEngine;
    
    // Spacecraft model path
    QString spacecraftModelPath;
    
    // Constants
    const double PI = 3.14159265358979323846;
};

#endif // SPACECRAFTVIEW_H