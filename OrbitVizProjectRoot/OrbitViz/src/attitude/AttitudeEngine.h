#ifndef ATTITUDEENGINE_H
#define ATTITUDEENGINE_H

#include <QObject>
#include <QTimer>
#include <QDateTime>
#include <QString>
#include <QMap>
#include <Eigen/Geometry>

// The AttitudeEngine handles spacecraft attitude calculations and dynamics
class AttitudeEngine : public QObject
{
    Q_OBJECT

public:
    explicit AttitudeEngine(QObject *parent = nullptr);
    ~AttitudeEngine();

    // Vector types for visualization
    enum VectorType {
        BODY_X,
        BODY_Y,
        BODY_Z,
        NADIR,
        SUN,
        VELOCITY,
        ANGULAR_MOMENTUM
    };

public slots:
    void startSimulation();
    void stopSimulation();
    void setSimulationSpeed(double speed);
    void setSimulationDate(const QDateTime &dateTime);
    void loadMockData(const QString &dataPath);
    void setQuaternion(const Eigen::Quaterniond &quat);
    void setBodyRates(double wx, double wy, double wz);
    void setReferenceFrame(const QString &frameName);
    void toggleVector(VectorType vectorType, bool enabled);

signals:
    void attitudeUpdated(const Eigen::Quaterniond &quat);
    void bodyRatesUpdated(double wx, double wy, double wz);
    void vectorUpdated(VectorType vectorType, const Eigen::Vector3d &vector);
    void simulationTimeChanged(const QString &timeString);

private slots:
    void updateSimulation();

private:
    QTimer *simulationTimer;
    QDateTime currentSimTime;
    double simSpeed;
    bool isRunning;

    // Attitude state
    Eigen::Quaterniond attitude;    // Current spacecraft attitude as a quaternion
    Eigen::Vector3d bodyRates;      // Angular velocity in body frame (rad/s)
    Eigen::Vector3d sunDirection;   // Direction to sun in reference frame
    Eigen::Vector3d nadirDirection; // Direction to nadir in reference frame
    Eigen::Vector3d velocityVector; // Velocity vector in reference frame
    
    // Map of enabled vectors
    QMap<VectorType, bool> enabledVectors;
    
    // Reference frame name (e.g., "ECI", "ECEF")
    QString referenceFrame;
    
    // Calculate and update vectors based on current attitude
    void updateVectors();
    
    // Generate sun direction based on time
    Eigen::Vector3d calculateSunDirection();
    
    // Generate nadir direction based on orbit model
    Eigen::Vector3d calculateNadirDirection();
    
    // Generate velocity vector based on orbit model
    Eigen::Vector3d calculateVelocityVector();
};

#endif // ATTITUDEENGINE_H